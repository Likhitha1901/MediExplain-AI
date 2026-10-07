from sentence_transformers import SentenceTransformer
import numpy as np


class MedicalReranker:

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def rerank(self, query, documents, top_k=3):

        if not documents:
            return []

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )

        scored_documents = []

        for document in documents:

            # Support both dictionary and MedicalDocument
            if isinstance(document, dict):
                content = document.get("content","")
                title=document.get("title","")
            else:
                content = document.content
                title=""

            document_embedding = self.model.encode(
                content,
                normalize_embeddings=True
            )

            semantic_score = float(
                np.dot(
                    query_embedding,
                    document_embedding
                )
            )
            query_words=set(
                word.lower().strip(".,!?;:\"'()[]{}")
                for word in query.split()
                if len(word)>2
            )
            title_words=set(
                word.lower().strip(".,!?;:\"'()[]{}")
                for word in title.split()
                if len(word)>2
            )
            content_words=set(
                word.lower().strip(".,!?;:\"'()[]{}")
                for word in content.split()
                if len(word)>2
            )
            title_overlap=(
                len(query_words&title_words)/len(query_words)
                if query_words else 0
            )
            content_overlap=(
                len(query_words&content_words)/len(query_words)
                if query_words else 0
            )
            keyword_score=(
                0.7*title_overlap+
                0.3*content_overlap
            )
            score=(
                0.70*semantic_score+
                0.30*keyword_score
            )
            
            content_overlap=(
            len(query_words&content_words)/len(query_words)
            if query_words else 0
            )
            quer_tokens=[
                word.lower().strip(".,?!:;()[]")
                for word in query.split()
                if len(word)>2
            ]
            title_lower=title.lower()
            content_lower=content.lower()
            phrase_score=0
            for i in range(len(quer_tokens)-1):
                phrase=(
                    quer_tokens[i]
                    +" "
                    + quer_tokens[i + 1]
                )
                if phrase in title_lower:
                    phrase_score+=1
                elif phrase in content_lower:
                    phrase_score+=0.5
            if query_tokens:
                phrase_score=min(
                    phrase_score/len(query_tokens),
                    1.0
                    
                )
            keyword_score=(
                0.40*title_overlap+
                0.20*content_overlap
                +0.40*phrase_score
            )
            score=(
                0.50*semantic_score+
                0.50*keyword_score
            )
            scored_documents.append(
                {
                    "document": document,
                    "score": score
                }
            )
            print(
                f"RERANK SCORE|"
                f"{document.get('title','Unknown')if isinstance(document,dict)else 'Unknown'}"
                f"|{score:.4f}"
            )
        scored_documents.sort(
            key=lambda x: x["score"],
            reverse=True
        )

# --------------------------------------------------------
# Keep only documents that are sufficiently relevant
# compared with the best matching document.
# --------------------------------------------------------

        best_score = scored_documents[0]["score"]

        filtered_documents = [
            item
            for item in scored_documents
            if item["score"] >= best_score * 0.45
        ]

        return filtered_documents[:top_k]