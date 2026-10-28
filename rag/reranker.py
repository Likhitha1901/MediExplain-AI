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
        query_words = {
            word.lower().strip(".,!?;:\"'()[]{}")
            for word in query.split()
            if len(word) > 2
        }
        query_tokens = [
            word.lower().strip(".,?!:;()[]")
            for word in query.split()
            if len(word) > 2
        ]

        seen_titles = set()

        for document in documents:
            # Support both dictionary and MedicalDocument
            if isinstance(document, dict):
                content = document.get("content", "")
                title = document.get("title", "")
            else:
                content = getattr(document, "content", "")
                title = getattr(document, "title", "")

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

            title_words = set(
                word.lower().strip(".,!?;:\"'()[]{}")
                for word in title.split()
                if len(word) > 2
            )
            content_words = set(
                word.lower().strip(".,!?;:\"'()[]{}")
                for word in content.split()
                if len(word) > 2
            )

            title_overlap = (
                len(query_words & title_words) / len(query_words)
                if query_words else 0
            )
            content_overlap = (
                len(query_words & content_words) / len(query_words)
                if query_words else 0
            )

            title_lower = title.lower()
            content_lower = content.lower()
            phrase_score = 0
            for i in range(len(query_tokens) - 1):
                phrase = (
                    query_tokens[i]
                    + " "
                    + query_tokens[i + 1]
                )
                if phrase in title_lower:
                    phrase_score += 1
                elif phrase in content_lower:
                    phrase_score += 0.5

            if query_tokens:
                phrase_score = min(
                    phrase_score / len(query_tokens),
                    1.0
                )

            keyword_score = (
                0.40 * title_overlap +
                0.20 * content_overlap +
                0.40 * phrase_score
            )

            score = (
                0.50 * semantic_score +
                0.50 * keyword_score
            )

            scored_documents.append(
                {
                    "document": document,
                    "score": score
                }
            )

        scored_documents.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        if not scored_documents:
            return []

        best_score = scored_documents[0]["score"]

        # Deduplicate and filter weak matches
        filtered_documents = []
        for item in scored_documents:
            if item["score"] < best_score * 0.40:
                continue

            doc = item["document"]
            title_key = (
                doc.get("title", "").strip().lower()
                if isinstance(doc, dict)
                else getattr(doc, "title", "").strip().lower()
            )

            if title_key and title_key in seen_titles:
                continue

            if title_key:
                seen_titles.add(title_key)

            filtered_documents.append(item)
            if len(filtered_documents) >= top_k:
                break

        return filtered_documents
