from typing import Any, Dict, List, Union
from annotated_types import doc
import numpy as np
from sentence_transformers import SentenceTransformer


class MedicalReranker:
    """
    Evidence-grounded medical reranker using all-MiniLM-L6-v2 embeddings
    combined with lexical and phrase overlap. Evaluates query-document relevance
    fairly across all trusted medical sources (MedlinePlus, NHS, Mayo Clinic,
    Cleveland Clinic, NIDDK, WHO).
    """

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def rerank(
        self,
        query: str,
        documents: List[Union[Dict[str, Any], Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate medical documents using:
        1. Query-document semantic cosine similarity (50%)
        2. Lexical keyword overlap (title 40%, content 20%, continuous phrase 40%) (50%)
        Preserves all source metadata, prevents duplicate passages, and filters weak matches.
        """
        if not documents:
            return []

        clean_query = query.strip()
        query_words = {
            w.lower().strip(".,!?;:\"'()[]{}")
            for w in clean_query.split()
            if len(w) > 2
        }
        query_tokens = [
            w.lower().strip(".,?!:;()[]")
            for w in clean_query.split()
            if len(w) > 2
        ]

        query_embedding = self.model.encode(
            clean_query,
            normalize_embeddings=True
        )

        scored_documents = []
        seen_content_signatures = set()

        for doc in documents:
            if isinstance(doc, dict):
                content = str(doc.get("content") or "")
                title = str(doc.get("title") or "")
                source = str(doc.get("source") or "MedlinePlus")
                url = str(doc.get("url") or "")
                category = str(doc.get("category") or "medical_information")
            else:
                content = str(getattr(doc, "content", "") or "")
                title = str(getattr(doc, "title", "") or "")
                source = str(getattr(doc, "source", "MedlinePlus") or "")
                url = str(getattr(doc, "url", "") or "")
                category = str(getattr(doc, "category", "medical_information") or "")

            if not content.strip():
                continue

            # Represent document with title and primary content for rich semantic encoding
            doc_text = f"{title}. {content[:1500]}" if title else content[:1500]
            doc_embedding = self.model.encode(
                doc_text,
                normalize_embeddings=True
            )

            semantic_score = float(np.dot(query_embedding, doc_embedding))
            # Bound cosine score between 0.0 and 1.0
            semantic_score = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))

            # Lexical & Phrase Overlap
            title_words = {
                w.lower().strip(".,!?;:\"'()[]{}")
                for w in title.split()
                if len(w) > 2
            }
            content_words = {
                w.lower().strip(".,!?;:\"'()[]{}")
                for w in content.split()
                if len(w) > 2
            }

            title_overlap = (
                len(query_words & title_words) / len(query_words)
                if query_words else 0.0
            )
            content_overlap = (
                len(query_words & content_words) / len(query_words)
                if query_words else 0.0
            )

            title_lower = title.lower()
            content_lower = content.lower()
            phrase_score = 0.0
            for i in range(len(query_tokens) - 1):
                phrase = f"{query_tokens[i]} {query_tokens[i + 1]}"
                if phrase in title_lower:
                    phrase_score += 1.0
                elif phrase in content_lower:
                    phrase_score += 0.5

            if query_tokens:
                phrase_score = min(phrase_score / max(len(query_tokens) - 1, 1), 1.0)

            keyword_score = (
                0.40 * title_overlap +
                0.20 * content_overlap +
                0.40 * phrase_score
            )

            # Balanced Final Relevance Score (50% Semantic, 50% Lexical)
            final_score = 0.50 * semantic_score + 0.50 * keyword_score

            # Standardized document dict preserving all metadata
            doc_dict = {
                "content": content,
                "title": title or "Medical Reference",
                "source": source,
                "url": url,
                "category": category,
            }
            if isinstance(doc, dict):
                for k, v in doc.items():
                    if k not in doc_dict:
                        doc_dict[k] = v

            scored_documents.append({
                "document": doc_dict,
                "score": float(final_score),
                "semantic_score": round(float(semantic_score), 4),
                "keyword_score": round(float(keyword_score), 4)
            })

        # Rank candidates by final score descending
        scored_documents.sort(key=lambda x: x["score"], reverse=True)
        print("\n========== ALL RERANKING SCORES ==========")

        for index, item in enumerate(scored_documents, start=1):
            doc = item["document"]

            print(f"\nRank: {index}")
            print(f"Title: {doc.get('title', 'Unknown')}")
            print(f"Source: {doc.get('source', 'Unknown')}")
            print(f"Score: {item['score']:.4f}")
            print(f"Semantic Score: {item['semantic_score']:.4f}")
            print(f"Keyword Score: {item['keyword_score']:.4f}")
            print(f"URL: {doc.get('url', '')}")

        print("==========================================\n")
        if not scored_documents:
            return []

        best_score = scored_documents[0]["score"]
        filtered_documents = []

        for item in scored_documents:
            # Filter matches with poor relative score
            if item["score"] < max(best_score * 0.40, 0.20):
                continue

            doc_obj = item["document"]
            # Deduplicate by content signature
            content_sig = (doc_obj.get("content") or "").strip().lower()[:100]
            if content_sig and content_sig in seen_content_signatures:
                continue
            if content_sig:
                seen_content_signatures.add(content_sig)

            filtered_documents.append(item)
            if len(filtered_documents) >= top_k:
                break

        return filtered_documents
