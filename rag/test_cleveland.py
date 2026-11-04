from rag.vector_store import MedicalVectorStore

vector_store = MedicalVectorStore()

results = vector_store.search_cleveland_clinic(
    "What are the symptoms of diabetes and how is it diagnosed?",
    top_k=5
)

print("\n========== FINAL CLEVELAND RESULTS ==========")
print("Documents retrieved:", len(results))

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}")
    print("Title:", doc["title"])
    print("Source:", doc["source"])
    print("URL:", doc["url"])
    print("Content length:", len(doc["content"]))
    print("Content preview:", doc["content"][:300])