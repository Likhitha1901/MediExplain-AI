import io
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from starlette.testclient import TestClient
import httpx2
import openai

from backend.app import app
from rag.safety import MedicalSafety
from rag.reranker import MedicalReranker
from rag.citations import extract_supporting_passage, map_claims_to_sources
from rag.generator import RateLimitExceededError, parse_retry_duration, format_duration


class MediExplainMasterTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.safety = MedicalSafety()

    # ========================================================
    # 1. MEDICAL SAFETY TESTS
    # ========================================================

    def test_safety_01_normal_educational_question(self):
        """Normal questions must not trigger false warnings."""
        res = self.safety.analyze("What are the common symptoms of diabetes?")
        self.assertEqual(res["level"], "Normal")
        self.assertEqual(len(res["matched_terms"]), 0)

    def test_safety_02_warning_high_fever(self):
        """High fever or persistent symptoms must trigger Warning level."""
        res = self.safety.analyze("I have had a high fever and dizziness for 2 days")
        self.assertEqual(res["level"], "Warning")
        self.assertIn("high fever", res["matched_terms"])

    def test_safety_03_emergency_chest_pain(self):
        """Severe chest pain symptoms must trigger Emergency level."""
        res = self.safety.analyze("I have severe chest pain and trouble breathing")
        self.assertEqual(res["level"], "Emergency")
        self.assertTrue(any("chest pain" in t or "breathing" in t for t in res["matched_terms"]))

    def test_safety_04_emergency_overdose(self):
        """Accidental overdose / poisoning must trigger Emergency level."""
        res = self.safety.analyze("I accidentally took extra pills of my medicine")
        self.assertEqual(res["level"], "Emergency")

    def test_safety_05_educational_emergency_condition(self):
        """Educational inquiries into acute conditions (e.g. 'What is anaphylaxis?') should downgrade appropriately."""
        res = self.safety.analyze("What is anaphylaxis and what causes it?")
        # An educational question on emergency topic should be Warning/Educational, not Emergency panic
        self.assertEqual(res["level"], "Warning")
        self.assertTrue(res.get("is_educational", False))

    # ========================================================
    # 2. CORS & HEALTH ENDPOINTS
    # ========================================================

    def test_cors_01_health_origin_header(self):
        """Verify /health returns 200 with proper Access-Control-Allow-Origin."""
        resp = self.client.get("/health", headers={"Origin": "http://127.0.0.1:5500"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "healthy"})
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "http://127.0.0.1:5500")

    def test_cors_02_preflight_options(self):
        """Verify CORS OPTIONS preflight returns 200 and allowed methods."""
        resp = self.client.options(
            "/api/chat",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "http://127.0.0.1:5500")
        self.assertIn("POST", resp.headers.get("access-control-allow-methods", ""))

    def test_cors_03_unhandled_500_preserves_cors(self):
        """Unexpected internal 500 error must still retain Access-Control-Allow-Origin."""
        with patch("backend.routes.vector_store.search", side_effect=RuntimeError("Test crash")):
            resp = self.client.post(
                "/api/chat",
                json={"message": "What is hypertension?"},
                headers={"Origin": "http://127.0.0.1:5500"}
            )
            self.assertEqual(resp.status_code, 500)
            self.assertEqual(resp.headers.get("access-control-allow-origin"), "http://127.0.0.1:5500")

    # ========================================================
    # 3. API RATE-LIMIT & ERROR HANDLING
    # ========================================================

    def test_api_01_empty_question_handling(self):
        """Empty questions return structured guidance without LLM call."""
        resp = self.client.post(
            "/api/chat",
            json={"message": "   ", "language": "English"},
            headers={"Origin": "http://127.0.0.1:5500"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Please enter a medical question", data["answer"])

    def test_api_02_simulated_rate_limit_error(self):
        """Simulated 429 RateLimitError must return HTTP 429 with CORS and detail."""
        req = httpx2.Request("POST", "https://api.openai.com/v1/responses")
        resp = httpx2.Response(
            429,
            headers={"retry-after": "9420", "x-ratelimit-reset-tokens": "2h37m"},
            request=req
        )
        fake_rate_limit = openai.RateLimitError(
            message="Rate limit reached for model gpt-6-luna. Please try again in 2h37m.",
            response=resp,
            body={"error": {"code": "rate_limit_exceeded", "message": "Rate limit reached"}}
        )

        with patch("backend.routes.generator.client.responses.create", side_effect=fake_rate_limit):
            chat_resp = self.client.post(
                "/api/chat",
                json={"message": "What is asthma?"},
                headers={"Origin": "http://127.0.0.1:5500"}
            )
            self.assertEqual(chat_resp.status_code, 429)
            self.assertEqual(chat_resp.headers.get("access-control-allow-origin"), "http://127.0.0.1:5500")
            detail = chat_resp.json().get("detail", {})
            self.assertEqual(detail.get("error_type"), "rate_limit")
            self.assertIn("2h37m", detail.get("retry_after", ""))

    # ========================================================
    # 4. EVIDENCE CITATION & RAG REFINEMENTS
    # ========================================================

    def test_rag_01_extract_supporting_passage(self):
        """Verify excerpt extraction extracts query-relevant sentences."""
        content = (
            "Metformin is an oral biguanide medicine used to treat type 2 diabetes. "
            "Common side effects of metformin include nausea, upset stomach, and diarrhea. "
            "Patients should take it with meals to minimize gastrointestinal discomfort."
        )
        passage = extract_supporting_passage("side effects of metformin", content)
        self.assertIn("side effects", passage.lower())
        self.assertIn("nausea", passage.lower())

    def test_rag_02_claim_to_source_mapping(self):
        """Verify factual claims map to the most relevant source."""
        answer = "Metformin lowers blood glucose levels. Common side effects include nausea and diarrhea."
        sources = [
            {"title": "Metformin Information", "category": "medication", "url": "https://medlineplus.gov/metformin.html", "content": "Metformin side effects include nausea and diarrhea."}
        ]
        mapped = map_claims_to_sources(answer, sources)
        self.assertEqual(len(mapped), 1)
        self.assertTrue(len(mapped[0]["supported_claims"]) > 0)

    # ========================================================
    # 5. PDF DOCUMENT UPLOAD & PARSING
    # ========================================================

    def test_doc_01_upload_pdf_validation(self):
        """Uploading non-PDF should return 400."""
        import base64
        resp = self.client.post(
            "/api/upload-doc",
            json={"filename": "notes.txt", "file_base64": base64.b64encode(b"Not a PDF").decode("utf-8")}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Only PDF", str(resp.json()))

    @patch("backend.routes.PdfReader")
    def test_doc_02_upload_pdf_valid(self, mock_pdf_reader):
        """Uploading a valid PDF with text should return doc_id and page count."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Patient diagnosis indicates mild asthma. Prescribed Albuterol inhaler as needed."
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader

        resp = self.client.post(
            "/api/upload-doc",
            json={"filename": "sample_report.pdf", "file_base64": "JVBERi0xLjQK"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("doc_id", data)
        self.assertEqual(data["filename"], "sample_report.pdf")
        self.assertEqual(data["pages"], 1)
        self.assertGreater(data["chunks_count"], 0)

    # ========================================================
    # 6. REPORT & TREATMENT COMPARISON ENDPOINTS
    # ========================================================

    def test_report_01_generate_report_structure(self):
        """Verify /api/generate-report creates a complete structured report."""
        resp = self.client.post(
            "/api/generate-report",
            json={
                "question": "What is hypertension?",
                "answer": "Hypertension is high blood pressure inside the arteries. It forces your heart to work harder.",
                "sources": [{"title": "High Blood Pressure", "sources": "MedlinePlus", "category": "disease"}],
                "confidence": "High",
                "safety_level": "Normal",
                "safety_message": "For educational purposes only."
            }
        )
        self.assertEqual(resp.status_code, 200)
        report = resp.json()
        self.assertIn("report_title", report)
        self.assertIn("key_facts", report)
        self.assertIn("precautions", report)
        self.assertIn("disclaimer", report)

    def test_treatment_01_compare_treatments(self):
        """Verify /api/treatment-compare returns structured treatment options."""
        resp = self.client.post(
            "/api/treatment-compare",
            json={"condition": "diabetes"}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["condition"], "Diabetes")
        self.assertTrue(len(data["options"]) >= 2)
        self.assertIn("benefits", data["options"][0])
        self.assertIn("risks", data["options"][0])

    def test_doctor_01_summary_checklist(self):
        """Verify /api/doctor-summary returns questions for the doctor."""
        resp = self.client.post(
            "/api/doctor-summary",
            json={
                "question": "What are the symptoms of asthma?",
                "answer": "Asthma symptoms include wheezing, breathlessness, and coughing.",
                "sources": []
            }
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(len(data["questions_to_ask"]) >= 3)
        self.assertTrue(len(data["critical_symptoms_to_mention"]) >= 2)


if __name__ == "__main__":
    unittest.main()
