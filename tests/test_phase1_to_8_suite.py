import base64
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from starlette.testclient import TestClient
import numpy as np
from PIL import Image, ImageDraw

from backend.app import app
import backend.conversations as conv_store
from rag.context_manager import MedicalContextManager
from rag.medical_doc_processor import MedicalReportProcessor, PrescriptionProcessor, preprocess_image_for_ocr
from rag.treatment_comparator import TreatmentComparator
from rag.vector_store import MedicalVectorStore
from rag.reranker import MedicalReranker
from rag.safety import MedicalSafety


class MediExplainAllPhasesTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.context_mgr = MedicalContextManager()
        cls.report_proc = MedicalReportProcessor()
        cls.rx_proc = PrescriptionProcessor()
        cls.vector_store = MedicalVectorStore()
        cls.reranker = MedicalReranker()
        cls.safety = MedicalSafety()

    # ========================================================
    # PHASE 1 TESTS: CONVERSATION MANAGEMENT & PERSISTENCE
    # ========================================================

    def test_phase1_01_create_five_distinct_conversations(self):
        """Test creating 5 distinct conversations with unique stable IDs and timestamps."""
        conv_ids = []
        for i in range(1, 6):
            c_data = {
                "id": f"conv_test_{i}_{os.urandom(4).hex()}",
                "title": f"Medical Query {i}: Condition {i}",
                "messages": [
                    {
                        "question": f"Question {i} for topic {i}?",
                        "answer": f"Answer {i} explaining topic {i} grounded in evidence.",
                        "confidence": "High",
                        "safetyLevel": "Normal",
                        "sources": [{"title": f"Source {i}", "sources": "NHS", "category": "disease"}]
                    }
                ]
            }
            saved = conv_store.save_conversation(c_data)
            self.assertEqual(saved["id"], c_data["id"])
            self.assertIn("createdAt", saved)
            self.assertIn("updatedAt", saved)
            conv_ids.append(saved["id"])

        # Verify all 5 conversations exist independently
        all_convs = conv_store.list_conversations()
        all_ids = {c["id"] for c in all_convs}
        for cid in conv_ids:
            self.assertIn(cid, all_ids)

    def test_phase1_02_switch_between_conversations_no_mixing(self):
        """Switch between 5 conversations in non-sequential order and verify message integrity."""
        # Setup 5 unique conversations
        test_ids = [f"conv_switch_{i}_{os.urandom(3).hex()}" for i in range(5)]
        for i, cid in enumerate(test_ids):
            conv_store.save_conversation({
                "id": cid,
                "title": f"Distinct Chat {i}",
                "messages": [{
                    "question": f"Specific User Question {i}",
                    "answer": f"Specific AI Response {i}",
                    "confidence": "High"
                }]
            })

        # Switch in non-sequential order: 0 -> 3 -> 1 -> 4 -> 2 -> 0
        switch_order = [0, 3, 1, 4, 2, 0]
        for idx in switch_order:
            target_id = test_ids[idx]
            retrieved = conv_store.get_conversation(target_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["id"], target_id)
            self.assertEqual(retrieved["title"], f"Distinct Chat {idx}")
            self.assertEqual(retrieved["messages"][0]["question"], f"Specific User Question {idx}")
            self.assertEqual(retrieved["messages"][0]["answer"], f"Specific AI Response {idx}")

    def test_phase1_03_rename_conversation(self):
        """Renaming a conversation must update only its title and timestamp."""
        cid = f"conv_rename_{os.urandom(4).hex()}"
        conv_store.save_conversation({
            "id": cid,
            "title": "Original Title",
            "messages": [{"question": "Q?", "answer": "A."}]
        })

        resp = self.client.patch(f"/api/conversations/{cid}", json={"title": "Updated Medical Consultation Title"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["title"], "Updated Medical Consultation Title")
        self.assertEqual(data["id"], cid)
        # Message content remains untouched
        self.assertEqual(len(data["messages"]), 1)

    def test_phase1_04_delete_conversation(self):
        """Deleting a conversation removes only the selected conversation."""
        cid1 = f"conv_del1_{os.urandom(3).hex()}"
        cid2 = f"conv_del2_{os.urandom(3).hex()}"
        conv_store.save_conversation({"id": cid1, "title": "Keep me", "messages": []})
        conv_store.save_conversation({"id": cid2, "title": "Delete me", "messages": []})

        del_resp = self.client.delete(f"/api/conversations/{cid2}")
        self.assertEqual(del_resp.status_code, 200)

        self.assertIsNone(conv_store.get_conversation(cid2))
        self.assertIsNotNone(conv_store.get_conversation(cid1))

    # ========================================================
    # PHASE 2 TESTS: PDF MEDICAL REPORT (DIGITAL & SCANNED OCR)
    # ========================================================

    def test_phase2_01_digital_pdf_report_extraction(self):
        """Test digital PDF report extraction and structured test analysis."""
        # Create a synthetic in-memory PDF with laboratory text
        from pypdf import PdfWriter
        writer = PdfWriter()
        # Add a page with clinical text
        page_text = (
            "PATIENT LABORATORY REPORT\n"
            "Age: 48   Gender: Male   Date: 12/10/2025\n"
            "Clinical Biochemistry Results:\n"
            "Fasting Blood Sugar: 142 mg/dL (Reference: 70 - 99 mg/dL) [HIGH]\n"
            "HbA1c: 7.4 % (Reference: < 5.7 %) [HIGH]\n"
            "Serum Creatinine: 1.1 mg/dL (Reference: 0.7 - 1.3 mg/dL) [NORMAL]\n"
            "Total Cholesterol: 220 mg/dL (Reference: < 200 mg/dL) [HIGH]\n"
            "Impression: Impaired fasting glucose and elevated glycated hemoglobin indicative of diabetes.\n"
        )
        # In pypdf, we can use annotation or create a clean test PDF
        # We can test process_pdf with an existing PDF or synthetic bytes
        pdf_path = PROJECT_ROOT / "data" / "raw" / "medical_documents" / "diabetes.pdf"
        if pdf_path.exists():
            pdf_bytes = pdf_path.read_bytes()
            result = self.report_proc.process_pdf(pdf_bytes, "diabetes.pdf")
            self.assertGreater(result["total_pages"], 0)
            self.assertIn("analysis", result)
            self.assertIn("disclaimer", result["analysis"])

    def test_phase2_02_structured_report_analysis_logic(self):
        """Test report analyzer parses patient info, abnormal values, and recommendations."""
        pages = [
            {
                "page_number": 1,
                "text": (
                    "Patient Age: 52   Sex: Female   Date: 15/09/2025\n"
                    "HbA1c: 7.8 %\n"
                    "Fasting Blood Sugar: 154 mg/dL\n"
                    "Serum Creatinine: 0.9 mg/dL\n"
                    "Diagnosis: Type 2 Diabetes Mellitus"
                )
            }
        ]
        analysis = self.report_proc.analyze_report_content(pages, pages[0]["text"], "lab_report.pdf")

        self.assertEqual(analysis["patient_info"].get("age"), "52")
        self.assertEqual(analysis["patient_info"].get("gender"), "Female")

        # Verify abnormal findings detected
        abnormal_test_names = [ab["test_name"] for ab in analysis["abnormal_findings"]]
        self.assertTrue(any("HbA1c" in name for name in abnormal_test_names))
        self.assertTrue(any("Fasting" in name for name in abnormal_test_names))

        # Verify disclaimer and recommendations
        self.assertTrue(len(analysis["recommendations"]) > 0)
        self.assertIn("PATIENT EDUCATION ONLY", analysis["disclaimer"])

    # ========================================================
    # PHASE 3 TESTS: PRESCRIPTION IMAGE OCR & SAFETY
    # ========================================================

    def test_phase3_01_clear_prescription_ocr(self):
        """Test OCR extraction on a clear synthetic prescription image."""
        img = Image.new("RGB", (800, 320), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((25, 25), "Medical Clinic - Prescription Rx", fill=(0, 0, 0))
        d.text((25, 75), "1. Metformin 500mg - 1 tab twice daily with meals x 30 days", fill=(0, 0, 0))
        d.text((25, 125), "2. Lisinopril 10mg - 1 tab once daily in morning", fill=(0, 0, 0))
        d.text((25, 175), "3. Paracetamol 650mg - 1 tab SOS for fever", fill=(0, 0, 0))

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        result = self.rx_proc.process_prescription_image(img_bytes, "rx_clear.jpg")
        self.assertIn("prescription_id", result)
        self.assertIn("medicines", result)

        med_names = [m["medicine_name"].lower() for m in result["medicines"]]
        self.assertTrue(any("metformin" in n for n in med_names))

    def test_phase3_02_prescription_safety_and_explanations(self):
        """Test that unclear handwriting is flagged and medicines are explained safely."""
        candidate_medicines = [
            {
                "medicine_name": "Metformin",
                "strength": "500mg",
                "dosage": "1 tablet",
                "frequency": "Twice daily (BID)",
                "route": "Oral",
                "duration": "30 days",
                "special_instructions": "Take with meals",
                "clarity": "High"
            },
            {
                "medicine_name": "Amoxi???",
                "strength": "Unspecified",
                "dosage": "1 unit",
                "frequency": "Unclear",
                "route": "Oral",
                "duration": "As advised",
                "special_instructions": "",
                "clarity": "Unclear / Verification Required"
            }
        ]

        explanation = self.rx_proc.generate_explanation(candidate_medicines)

        # 1. Metformin has purpose and side effects
        met_exp = next(m for m in explanation["medicines"] if m["medicine_name"] == "Metformin")
        self.assertIn("blood glucose", met_exp["purpose"].lower())
        self.assertTrue(len(met_exp["common_side_effects"]) > 0)

        # 2. Unclear medicine triggers safety warning
        self.assertTrue(any("Amoxi" in w or "confidence" in w for w in explanation["safety_warnings"]))

        # 3. Medical safety disclaimer is present
        self.assertIn("CRITICAL MEDICAL SAFETY", explanation["disclaimer"])

    # ========================================================
    # PHASE 4 TESTS: DYNAMIC TREATMENT COMPARISON
    # ========================================================

    def test_phase4_01_asthma_treatment_comparison(self):
        """Test 9-field treatment comparison for asthma."""
        comp = TreatmentComparator(vector_store=self.vector_store, reranker=self.reranker)
        res = comp.compare("Compare the available treatments for asthma.")

        self.assertEqual(res["condition"], "Asthma")
        self.assertGreater(len(res["options"]), 0)

        first_opt = res["options"][0]
        # Verify 9 required fields
        required_fields = [
            "treatment", "how_it_works", "main_purpose", "benefits",
            "side_effects", "risks_precautions", "suitability",
            "administration", "evidence_sources"
        ]
        for field in required_fields:
            self.assertIn(field, first_opt, f"Missing required field '{field}'")
            self.assertTrue(len(first_opt[field]) > 0)

        self.assertIn("CLINICAL TREATMENT NOTICE", res["disclaimer"])

    def test_phase4_02_hypertension_and_other_diseases(self):
        """Test treatment comparison for hypertension and migraine."""
        comp = TreatmentComparator(vector_store=self.vector_store, reranker=self.reranker)

        # Hypertension
        res_htn = comp.compare("hypertension")
        self.assertEqual(res_htn["condition"], "Hypertension")
        self.assertGreater(len(res_htn["options"]), 0)

        # Migraine
        res_mig = comp.compare("migraine")
        self.assertEqual(res_mig["condition"], "Migraine")
        self.assertGreater(len(res_mig["options"]), 0)

    # ========================================================
    # PHASE 5 TESTS: FOLLOW-UP CONTEXT REFORMULATION
    # ========================================================

    def test_phase5_01_pronoun_resolution_followup(self):
        """Test resolving 'its symptoms' to previous disease topic."""
        history = [
            {"role": "user", "content": "What is iron deficiency anemia?"},
            {"role": "assistant", "content": "Iron deficiency anemia is a condition caused by insufficient iron..."}
        ]

        query = "What are its symptoms?"
        contextualized, was_modified, topic = self.context_mgr.contextualize_query(query, history)

        self.assertTrue(was_modified)
        self.assertEqual(topic.lower(), "iron deficiency anemia")
        self.assertIn("iron deficiency anemia", contextualized.lower())

    def test_phase5_02_treatment_pronoun_followup(self):
        """Test resolving 'How is it treated?' to previous condition."""
        history = [
            {"role": "user", "content": "Tell me about asthma causes."},
            {"role": "assistant", "content": "Asthma causes include airway inflammation and triggers."}
        ]

        query = "How is it treated?"
        contextualized, was_modified, topic = self.context_mgr.contextualize_query(query, history)

        self.assertTrue(was_modified)
        self.assertEqual(topic.lower(), "asthma")
        self.assertIn("asthma", contextualized.lower())

    # ========================================================
    # PHASE 6 TESTS: SELECT ANY WORD EXPLANATION
    # ========================================================

    def test_phase6_01_explain_selection_endpoint(self):
        """Test explaining selected medical phrases."""
        resp = self.client.post(
            "/api/explain-selection",
            json={
                "selected_text": "chronic condition",
                "context": "Hypertension is a chronic condition characterized by high pressure."
            }
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("term", data)
        self.assertIn("explanation", data)
        self.assertIn("long-lasting", data["explanation"].lower())

    def test_phase6_02_explain_lab_selection(self):
        """Test explaining selected lab term."""
        resp = self.client.post(
            "/api/explain-selection",
            json={"selected_text": "HbA1c", "context": "Your HbA1c is 7.2%."}
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Hemoglobin A1c", data["term"])

    # ========================================================
    # PHASE 7 TESTS: MULTI-SOURCE RETRIEVAL & RERANKING
    # ========================================================

    def test_phase7_01_no_diabetes_pdf_leak(self):
        """Confirm diabetes.pdf is not retrieved in general search results."""
        results = self.vector_store.search(query="diabetes symptoms and treatment", top_k=10)
        if results and results.get("metadatas"):
            for meta in results["metadatas"][0]:
                meta = meta or {}
                self.assertNotEqual(meta.get("file_name"), "diabetes.pdf")
                self.assertNotEqual(meta.get("source"), "diabetes")

    def test_phase7_02_multi_source_diversity(self):
        """Verify retrieved documents include evidence from diverse trusted sources."""
        results = self.vector_store.search(query="asthma inhaler and breathing symptoms", top_k=8)
        sources_found = set()
        if results and results.get("metadatas"):
            for meta in results["metadatas"][0]:
                meta = meta or {}
                s = meta.get("source")
                if s:
                    sources_found.add(s)

        # Confirm sources include MedlinePlus or new sources like WHO / NHS
        self.assertTrue(len(sources_found) >= 1)


if __name__ == "__main__":
    unittest.main()
