from retriever import MedicalRetriever


retriever = MedicalRetriever()


retriever.add_document(
    content="""
    Diabetes is a chronic condition that affects how the body
    regulates blood glucose. Common symptoms may include increased
    thirst, frequent urination and increased hunger.
    """,
    source="NIDDK",
    title="Diabetes Overview",
    category="disease",
    url="https://www.niddk.nih.gov/"
)


retriever.add_document(
    content="""
    Metformin is a medicine commonly used to help control blood
    glucose in people with type 2 diabetes. Common side effects
    can include diarrhea, nausea and stomach discomfort.
    """,
    source="Medical Reference",
    title="Metformin Information",
    category="medication",
    url=None
)


results = retriever.search(
    query="What are symptoms of diabetes?",
    category="disease",
    top_k=3
)


for result in results:

    print("\n-------------------------")

    print("Title:", result.title)

    print("Source:", result.source)

    print("Category:", result.category)

    print("URL:", result.url)

    print("Content:", result.content.strip())