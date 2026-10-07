from dataclasses import dataclass
from typing import List


@dataclass
class MedicalDocument:
    content: str
    source: str
    title: str
    category: str
    url: str | None = None


class MedicalRetriever:

    def __init__(self):
        # Temporary knowledge base.
        # Later this will be replaced by our vector database.
        self.documents: List[MedicalDocument] = []

    def add_document(
        self,
        content: str,
        source: str,
        title: str,
        category: str,
        url: str | None = None
    ):
        document = MedicalDocument(
            content=content,
            source=source,
            title=title,
            category=category,
            url=url
        )

        self.documents.append(document)

    def search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5
    ) -> List[MedicalDocument]:

        query_words = set(query.lower().split())

        results = []

        for document in self.documents:

            # If a category is provided,
            # prefer documents belonging to that category.
            category_score = 0

            if category and document.category == category:
                category_score = 2

            content_words = set(
                document.content.lower().split()
            )

            keyword_score = len(
                query_words.intersection(content_words)
            )

            total_score = keyword_score + category_score

            if total_score > 0:
                results.append(
                    (total_score, document)
                )

        # Highest score first
        results.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            document
            for score, document in results[:top_k]
        ]