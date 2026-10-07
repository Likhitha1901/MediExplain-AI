from reranker import MedicalReranker


reranker = MedicalReranker()


query = (
    "Why am I feeling very thirsty "
    "and going to the bathroom frequently?"
)


documents = [

    {
        "content": """
        Diabetes is a chronic condition that affects how the body
        regulates blood glucose. Common symptoms may include
        increased thirst, frequent urination and increased hunger.
        """,
        "source": "NIDDK",
        "title": "Diabetes Overview",
        "category": "disease"
    },

    {
        "content": """
        Asthma is a chronic respiratory condition that can cause
        wheezing, coughing, chest tightness and difficulty breathing.
        """,
        "source": "Medical Reference",
        "title": "Asthma Overview",
        "category": "disease"
    },

    {
        "content": """
        Metformin is a medicine commonly used to help control
        blood glucose in people with type 2 diabetes. Common
        side effects can include diarrhea, nausea and stomach
        discomfort.
        """,
        "source": "Medical Reference",
        "title": "Metformin Information",
        "category": "medication"
    }
]


results = reranker.rerank(
    query=query,
    documents=documents,
    top_k=3
)


print("\nUSER QUESTION:")
print(query)

print("\nRERANKED RESULTS:")
print("=" * 50)


for index, result in enumerate(results):

    document = result["document"]
    score = result["score"]

    print(f"\nRank {index + 1}")
    print("Title:", document["title"])
    print("Category:", document["category"])
    print("Source:", document["source"])
    print("Relevance Score:", round(score, 4))