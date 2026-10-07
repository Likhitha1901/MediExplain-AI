from typing import Literal


QueryCategory = Literal[
    "disease",
    "medication",
    "treatment",
    "medical_concept",
    "document",
    "general"
]


def classify_query(message: str) -> QueryCategory:

    text = message.lower().strip()

    # Document-related questions
    document_words = [
        "my report",
        "medical report",
        "uploaded",
        "document",
        "pdf",
        "this report",
        "my scan",
        "my prescription"
    ]

    # Medication-related questions
    medication_words = [
        "medicine",
        "medication",
        "drug",
        "tablet",
        "capsule",
        "dose",
        "dosage",
        "side effect",
        "side effects",
        "metformin",
        "ibuprofen",
        "paracetamol",
        "antibiotic"
    ]

    # Treatment-related questions
    treatment_words = [
        "treatment",
        "therapy",
        "treated",
        "treat",
        "management",
        "cure",
        "how to manage",
        "how is it treated"
    ]

    # Disease-related questions
    disease_words = [
        "disease",
        "disorder",
        "condition",
        "symptom",
        "symptoms",
        "cause",
        "causes",
        "risk factor",
        "risk factors",
        "diagnosis",
        "diagnose",
        "diabetes",
        "asthma",
        "hypertension",
        "migraine",
        "cancer",
        "arthritis",
        "pneumonia"
    ]

    # Medical concept questions
    medical_concept_words = [
        "what is",
        "what does",
        "meaning of",
        "define",
        "definition",
        "explain",
        "what does it mean",
        "medical term"
    ]

    # Priority order
    if any(word in text for word in document_words):
        return "document"

    if any(word in text for word in medication_words):
        return "medication"

    if any(word in text for word in treatment_words):
        return "treatment"

    if any(word in text for word in disease_words):
        return "disease"

    if any(word in text for word in medical_concept_words):
        return "medical_concept"

    return "general"