from query_router import classify_query


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
    print(f"{question} → {category}")