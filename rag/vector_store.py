import chromadb
from sentence_transformers import SentenceTransformer


class MedicalVectorStore:

    def __init__(self, collection_name="medical_knowledge"):

        # Create a persistent Chroma database
        self.client = chromadb.PersistentClient(
            path="data/chroma_db"
        )

        # Create or load our collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        # Embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def add_documents(self, documents):

        if not documents:
            return

        texts = []
        ids = []
        metadatas = []

        for index, document in enumerate(documents):

            texts.append(document["content"])

            ids.append(
                document.get(
                    "id",
                    f"doc_{index}"
                )
            )

            metadatas.append({
                "source": document.get("source", ""),
                "title": document.get("title", ""),
                "category": document.get("category", ""),
                "url": document.get("url", ""),
                "file_name": document.get("file_name", ""),
                "chunk_index":str(document.get("chunk_index", ""))
                    
            })

        # Convert documents into embeddings
        embeddings = self.embedding_model.encode(
            texts
        ).tolist()

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query, top_k=5):

        # Convert user question into embedding
        query_embedding = self.embedding_model.encode(
            query
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results