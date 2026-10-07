from vector_store import MedicalVectorStore


store = MedicalVectorStore()


documents = [

    {
        "id": "diabetes_001",
        "content": """
        Diabetes is a chronic condition that affects how the body
        regulates blood glucose. Common symptoms may include
        increased thirst, frequent urination and increased hunger.
        """,
        "source": "NIDDK",
        "title": "Diabetes Overview",
        "category": "disease",
        "url": "https://www.niddk.nih.gov/"
    },

    {
        "id": "metformin_001",
        "content": """
        Metformin is a medicine commonly used to help control
        blood glucose in people with type 2 diabetes. Common
        side effects can include diarrhea, nausea and stomach
        discomfort.
        """,
        "source": "Medical Reference",
        "title": "Metformin Information",
        "category": "medication",
        "url": ""
    },

    {
        "id": "asthma_001",
        "content": """
        Asthma is a chronic respiratory condition that can cause
        symptoms such as wheezing, coughing, chest tightness
        and difficulty breathing.
        """,
        "source": "Medical Reference",
        "title": "Asthma Overview",
        "category": "disease",
        "url": ""
    }
]


# Add documents to Chroma
store.add_documents(documents)


# Test semantic search
query = "Why am I feeling very thirsty and going to the bathroom frequently?"

results = store.search(
    query=query,
    top_k=3
)


print("\nUSER QUESTION:")
print(query)

print("\nRETRIEVED MEDICAL INFORMATION:")
print("--------------------------------")


for i, document in enumerate(
    results["documents"][0]
):

    metadata = results["metadatas"][0][i]

    print(f"\nResult {i + 1}")

    print("Title:", metadata["title"])
    print("Source:", metadata["source"])
    print("Category:", metadata["category"])
    print("URL:", metadata["url"])

    print("Content:")
    print(document)

    print("--------------------------------")