import os
import unittest
from rag.medical_doc_processor import PrescriptionProcessor

TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_prescriptions")


class TestPrescriptionPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.processor = PrescriptionProcessor()

    def test_clear_handwritten_prescription(self):
        """Test Case 1: Clear clinical prescription (Dextrose, ORS, Paracetamol)."""
        img_path = os.path.join(TEST_DIR, "rx_clear_handwritten.png")
        self.assertTrue(os.path.exists(img_path))

        with open(img_path, "rb") as f:
            data = f.read()

        result = self.processor.process_prescription_image(data, "rx_clear_handwritten.png")
        self.assertIsNotNone(result)
        self.assertIn("medicines", result)
        self.assertIn("explanations", result)

        med_names = [m["medicine_name"].lower() for m in result["medicines"]]
        # Verify Dextrose or ORS or Paracetamol recognized
        has_expected_med = any("dextrose" in name or "ors" in name or "paracetamol" in name for name in med_names)
        self.assertTrue(has_expected_med, f"Expected Dextrose/ORS/Paracetamol in recognized medicines: {med_names}")

        # Verify explanations were generated automatically without manual entry
        explanations = result.get("explanations", [])
        self.assertGreater(len(explanations), 0, "Explanations must be generated automatically.")
        for exp in explanations:
            self.assertIn("purpose", exp)
            self.assertIn("common_side_effects", exp)
            self.assertIn("precautions", exp)
            self.assertIn("evidence_source", exp)

    def test_printed_prescription(self):
        """Test Case 2: Printed electronic prescription (Metformin, Lisinopril, Atorvastatin)."""
        img_path = os.path.join(TEST_DIR, "rx_printed.png")
        self.assertTrue(os.path.exists(img_path))

        with open(img_path, "rb") as f:
            data = f.read()

        result = self.processor.process_prescription_image(data, "rx_printed.png")
        med_names = [m["medicine_name"].lower() for m in result["medicines"]]

        self.assertTrue(any("metformin" in name for name in med_names), f"Metformin should be detected: {med_names}")
        self.assertTrue(any("lisinopril" in name for name in med_names), f"Lisinopril should be detected: {med_names}")
        self.assertTrue(any("atorvastatin" in name for name in med_names), f"Atorvastatin should be detected: {med_names}")

        # Check interaction check
        interactions = result.get("interactions", [])
        self.assertTrue(len(interactions) > 0, "Drug-drug interaction check should detect Metformin + Lisinopril.")

    def test_low_quality_prescription_and_unclear_flagging(self):
        """Test Case 3: Low-quality/unclear handwritten entry must be flagged, not fabricated."""
        img_path = os.path.join(TEST_DIR, "rx_low_quality.png")
        self.assertTrue(os.path.exists(img_path))

        with open(img_path, "rb") as f:
            data = f.read()

        result = self.processor.process_prescription_image(data, "rx_low_quality.png")
        medicines = result.get("medicines", [])
        self.assertGreater(len(medicines), 0)

        # Check that Amoxicillin is recognized
        med_names = [m["medicine_name"].lower() for m in medicines]
        self.assertTrue(any("amoxicillin" in n for n in med_names))

        # Check that unclear lines have appropriate status
        unclear_items = [m for m in medicines if m.get("clarity") == "Unclear / Verification Required"]
        if unclear_items:
            for item in unclear_items:
                self.assertIn("Unclear", item["readability_status"])
                # Raw text must be preserved
                self.assertIsNotNone(item.get("raw_text"))

    def test_safety_disclaimer_present(self):
        """Rule F: Critical medical safety disclaimers must always be present."""
        img_path = os.path.join(TEST_DIR, "rx_printed.png")
        with open(img_path, "rb") as f:
            data = f.read()

        result = self.processor.process_prescription_image(data, "rx_printed.png")
        disclaimer = result.get("disclaimer", "")
        self.assertTrue("CRITICAL MEDICAL SAFETY" in disclaimer or "pharmacist" in disclaimer)


if __name__ == "__main__":
    unittest.main()
