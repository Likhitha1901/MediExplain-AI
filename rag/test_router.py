from rag.query_router import classify_query


def test_routing():
    questions = [
        "What is diabetes?",
        "What are the symptoms of asthma?",
        "What is metformin used for?",
        "What are the side effects of ibuprofen?",
        "How is migraine treated?",
        "Explain my medical report",
        "What does HbA1c mean?"
    ]

    for question in questions:
        category = classify_query(question)
        print(f"{question} -> {category}")


if __name__ == "__main__":
    test_routing()
