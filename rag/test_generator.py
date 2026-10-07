from generator import MedicalGenerator


generator = MedicalGenerator()


question = "What are common symptoms of diabetes?"

context = """
Diabetes is a chronic condition that affects how the body
regulates blood glucose. Common symptoms may include increased
thirst, frequent urination and increased hunger.
"""


response = generator.generate(
    question=question,
    context=context,
    language="English"
)


print("\nQUESTION:")
print(question)

print("\nRESPONSE:")
print(response)