import os
import hashlib

from rag.vector_store import MedicalVectorStore


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MEDICAL_DATA_DIR = os.path.join(
    BASE_DIR,
    "medical_data"
)


# ============================================================
# MEDICAL CATEGORIES
# ============================================================

CATEGORIES = [
    "diseases",
    "medications",
    "symptoms",
    "treatments",
    "medical_tests",
    "procedures",
    "anatomy",
    "pregnancy",
    "mental_health",
    "nutrition",
    "emergency"
]


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    for category in CATEGORIES:

        category_path = os.path.join(
            MEDICAL_DATA_DIR,
            category
        )

        if not os.path.exists(category_path):
            continue

        for filename in os.listdir(category_path):

            if not filename.lower().endswith(".txt"):
                continue

            file_path = os.path.join(
                category_path,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read().strip()

            if not content:
                continue

            title = os.path.splitext(
                filename
            )[0]

            document_id = hashlib.md5(
                file_path.encode("utf-8")
            ).hexdigest()

            documents.append({

                "id": document_id,

                "content": content,

                "title": title,

                "source": "MediExplain Medical Knowledge Base",

                "category": category,

                "url": "",

                "file_name": filename,

                "chunk_index": 0

            })

    return documents


# ============================================================
# INGEST INTO CHROMADB
# ============================================================

def main():

    print("\n" + "=" * 60)

    print(
        "MediExplain AI - Medical Knowledge Ingestion"
    )

    print("=" * 60)

    documents = load_documents()

    print(
        f"\nDocuments found: {len(documents)}"
    )

    if not documents:

        print(
            "\nNo medical documents found."
        )

        print(
            "Please add .txt files inside the "
            "medical_data folders."
        )

        return

    vector_store = MedicalVectorStore()

    vector_store.add_documents(
        documents
    )

    print(
        f"\nSuccessfully added "
        f"{len(documents)} documents to ChromaDB."
    )

    print("\nIngestion completed successfully.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()