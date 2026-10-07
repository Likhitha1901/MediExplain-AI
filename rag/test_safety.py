from safety import MedicalSafety


safety = MedicalSafety()


questions = [
    "What are the symptoms of diabetes?",
    "What is metformin used for?",
    "What dosage of metformin should I take?",
    "I have severe chest pain, what should I do?"
]


for question in questions:

    result = safety.analyze(question)

    print()
    print("QUESTION:", question)
    print("SAFETY:", result["level"])
    print("MESSAGE:")
    print(safety.get_safety_message(result))