import re
from typing import Dict, List, Optional


def extract_supporting_passage(query: str, content: str, max_chars: int = 250) -> str:
    """
    Extracts the most relevant continuous sentence or excerpt from
    the medical content that matches key query terms.
    """
    if not content or not content.strip():
        return ""

    # Clean HTML tags if present in scraped XML/HTML
    clean_text = re.sub(r"<[^>]+>", " ", content)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    if not sentences:
        return clean_text[:max_chars]

    query_words = set(
        re.findall(r"\b[a-z]{3,}\b", query.lower())
    )

    best_sentence = ""
    best_score = -1

    for sent in sentences:
        sent_words = set(re.findall(r"\b[a-z]{3,}\b", sent.lower()))
        overlap = len(query_words.intersection(sent_words))
        if overlap > best_score:
            best_score = overlap
            best_sentence = sent

    if not best_sentence or best_score <= 0:
        return clean_text[:max_chars] + ("..." if len(clean_text) > max_chars else "")

    if len(best_sentence) > max_chars:
        return best_sentence[:max_chars].rsplit(" ", 1)[0] + "..."

    return best_sentence.strip()


def map_claims_to_sources(answer: str, sources: List[Dict]) -> List[Dict]:
    """
    Associates answer points or sections with the most applicable
    retrieved source based on semantic and keyword overlap.
    """
    if not answer or not sources:
        return []

    # Clean markdown formatting from answer sentences
    clean_answer = re.sub(r"[*#_`]", "", answer)
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", clean_answer)
        if len(s.strip()) > 20
    ]

    mapped_evidence = []

    for src in sources:
        src_title = src.get("title", "")
        src_cat = src.get("category", "")
        src_url = src.get("url", "")
        src_content = src.get("content") or src.get("passage", "")

        src_words = set(
            re.findall(r"\b[a-z]{3,}\b", (src_title + " " + src_content).lower())
        )

        matched_claims = []
        for sent in sentences:
            sent_words = set(re.findall(r"\b[a-z]{3,}\b", sent.lower()))
            if src_words and len(src_words.intersection(sent_words)) >= 2:
                matched_claims.append(sent)

        excerpt = extract_supporting_passage(src_title, src_content) if src_content else ""

        mapped_evidence.append({
            "title": src_title,
            "category": src_cat,
            "url": src_url,
            "excerpt": excerpt,
            "supported_claims": matched_claims[:2]
        })

    return mapped_evidence
