from rag.vector_store import MedicalVectorStore

store = MedicalVectorStore()

results = store.search_medlineplus(
    "iron deficiency anemia symptoms",
    top_k=5,
)

for item in results:
    print("\nTitle:", item["title"])
    print("Source:", item["source"])
    print("URL:", item["url"])
    print("Content:", item["content"][:250])