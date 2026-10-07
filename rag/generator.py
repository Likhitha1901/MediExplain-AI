import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class MedicalGenerator:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing from .env"
            )

        if not model:
            raise ValueError(
                "OPENAI_MODEL is missing from .env"
            )

        self.client = OpenAI(
            api_key=api_key
        )

        self.model_name = model

    def generate(
        self,
        question: str,
        context: str,
        language: str = "English"
    ):

        instructions = """
You are MediExplain AI, a medical information assistant.

Your job is to explain medical information clearly,
accurately, and in patient-friendly language.

Rules:

1. Use the provided medical evidence as the primary source.
2. Do not invent medical facts.
3. Do not diagnose the user.
4. Do not claim that a symptom definitely means a disease.
5. Explain uncertainty when appropriate.
6. Do not recommend changing prescription medicines.
7. For medication questions, explain purpose,
   side effects and dosage only when supported by
   the provided evidence.
8. If evidence is insufficient, say so.
9. Use simple language that patients can understand.
10. Recommend consulting a qualified healthcare
    professional for diagnosis or personalized treatment.
11. Answer in the requested language.
"""

        prompt = f"""
MEDICAL EVIDENCE:

{context}

----------------------------------------

USER QUESTION:

{question}

----------------------------------------

REQUESTED LANGUAGE:

{language}

Using the medical evidence above, provide a clear,
patient-friendly answer.
"""

        response = self.client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=prompt
        )

        return {
            "answer": response.output_text,
            "language": language,
            "model": self.model_name
        }