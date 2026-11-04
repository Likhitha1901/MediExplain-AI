import re
from typing import Dict, List, Optional, Tuple


# Common pronouns and demonstratives referring to prior medical entities
COREFERENCE_PATTERNS = [
    r"\b(it|its|this|that|these|those)\b",
    r"\b(this|the)\s+(disease|condition|illness|disorder|infection|syndrome)\b",
    r"\b(this|the|that)\s+(medicine|medication|drug|pill|tablet|treatment|therapy)\b",
    r"\b(its|the)\s+(symptoms|causes|treatment|side\s+effects|dosage|prevention|risks|complications)\b",
    r"\bhow\s+is\s+it\s+(treated|managed|diagnosed|prevented|taken)\b",
    r"\bwhat\s+are\s+(its|the)\s+(symptoms|causes|treatments|risks|side\s+effects)\b",
    r"\bcan\s+it\s+(cause|be\s+cured|kill|spread|worsen)\b",
    r"\bis\s+it\s+(contagious|curable|genetic|dangerous|fatal|hereditary)\b",
    r"\bthe\s+treatment\s+mentioned\s+earlier\b",
    r"\bthe\s+medicine\s+mentioned\s+earlier\b",
]

MEDICAL_ENTITY_STOPWORDS = {
    "what", "is", "are", "how", "why", "when", "where", "can", "does", "do",
    "the", "a", "an", "and", "or", "of", "in", "for", "with", "about", "tell",
    "me", "please", "explain", "describe", "detail", "details", "help", "i",
    "have", "my", "symptoms", "treatment", "treatments", "causes", "prevention"
}


def extract_primary_medical_subject(text: str) -> Optional[str]:
    """
    Extract the main medical disease, medication, or condition
    from a question or answer snippet.
    """
    if not text:
        return None

    clean = re.sub(r"[^\w\s-]", " ", text)
    # Common questions: "What is iron deficiency anemia?" -> "iron deficiency anemia"
    patterns = [
        r"(?:what\s+is|what\s+are|tell\s+me\s+about|explain|overview\s+of)\s+([a-zA-Z0-9\s-]{3,40})(?:\?|$)",
        r"(?:symptoms\s+of|causes\s+of|treatment\s+for|side\s+effects\s+of)\s+([a-zA-Z0-9\s-]{3,40})(?:\?|$)",
        r"([a-zA-Z0-9\s-]{3,35})\s+(?:symptoms|causes|treatment|medication|overview)",
    ]

    for p in patterns:
        m = re.search(p, clean, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            # Clean trailing words
            words = [w for w in candidate.split() if w.lower() not in MEDICAL_ENTITY_STOPWORDS]
            if words:
                return " ".join(words)

    # Fallback: find key capitalized or medical noun phrases
    words = clean.split()
    meaningful = [w for w in words if len(w) > 3 and w.lower() not in MEDICAL_ENTITY_STOPWORDS]
    if meaningful:
        return " ".join(meaningful[:3])

    return None


class MedicalContextManager:
    """
    Manages multi-turn conversation context, resolves pronouns and references,
    reformulates ambiguous follow-up questions for accurate retrieval,
    and formats history for LLM generation.
    """

    def has_coreference(self, query: str) -> bool:
        """Check if query contains ambiguous pronouns or references."""
        q_lower = query.lower().strip()
        for pattern in COREFERENCE_PATTERNS:
            if re.search(pattern, q_lower):
                return True
        # Very short queries like "symptoms?", "treatment?", "side effects?"
        words = q_lower.split()
        if len(words) <= 3 and any(w in ["symptoms", "treatment", "causes", "cure", "effects", "prevention", "dosage"] for w in words):
            return True
        return False

    def find_last_medical_topic(self, history: List[Dict[str, str]]) -> Optional[str]:
        """
        Traverse backward through conversation history to find the most recent
        specific disease, medication, or medical condition discussed.
        """
        if not history:
            return None

        # Check in reverse order (most recent turns first)
        for turn in reversed(history):
            role = turn.get("role", "")
            content = turn.get("content", "")

            # If user turn, extract subject from question
            if role == "user":
                subject = extract_primary_medical_subject(content)
                if subject and len(subject) > 2:
                    return subject

            # If assistant turn, check for first capitalized medical topic or title
            if role == "assistant":
                m = re.search(r"###?\s*([A-Za-z0-9\s-]{3,30})", content)
                if m:
                    subj = m.group(1).strip()
                    if subj.lower() not in ["medical explanation", "overview", "introduction"]:
                        return subj
                subject = extract_primary_medical_subject(content[:200])
                if subject and len(subject) > 2:
                    return subject

        return None

    def contextualize_query(
        self,
        current_query: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Reformulates follow-up question using previous conversation context.
        Returns:
            (contextualized_query, was_reformulated, resolved_topic)
        """
        clean_q = current_query.strip()
        if not history or not self.has_coreference(clean_q):
            return clean_q, False, None

        topic = self.find_last_medical_topic(history)
        if not topic:
            return clean_q, False, None

        # Perform pronoun substitution
        reformulated = clean_q
        replacements = [
            (r"\bhow\s+is\s+it\s+treated\b", f"How is {topic} treated"),
            (r"\bhow\s+is\s+it\s+managed\b", f"How is {topic} managed"),
            (r"\bhow\s+is\s+it\s+diagnosed\b", f"How is {topic} diagnosed"),
            (r"\bwhat\s+are\s+its\s+symptoms\b", f"What are the symptoms of {topic}"),
            (r"\bwhat\s+are\s+its\s+causes\b", f"What are the causes of {topic}"),
            (r"\bwhat\s+are\s+its\s+treatments\b", f"What are the treatments for {topic}"),
            (r"\bwhat\s+are\s+its\s+side\s+effects\b", f"What are the side effects of {topic}"),
            (r"\bcan\s+it\s+be\s+cured\b", f"Can {topic} be cured"),
            (r"\bis\s+it\s+curable\b", f"Is {topic} curable"),
            (r"\bis\s+it\s+contagious\b", f"Is {topic} contagious"),
            (r"\bits\s+symptoms\b", f"symptoms of {topic}"),
            (r"\bits\s+causes\b", f"causes of {topic}"),
            (r"\bits\s+treatment\b", f"treatment of {topic}"),
            (r"\bits\s+prevention\b", f"prevention of {topic}"),
            (r"\bthis\s+disease\b", topic),
            (r"\bthis\s+condition\b", topic),
            (r"\bthat\s+medicine\b", topic),
            (r"\bthis\s+medicine\b", topic),
            (r"\bthe\s+treatment\s+mentioned\s+earlier\b", f"treatments for {topic}"),
            (r"\bthe\s+medicine\s+mentioned\s+earlier\b", f"{topic}"),
            (r"\bthe\s+disease\b", topic),
            (r"\bit\b", topic),
        ]

        for pattern, repl in replacements:
            new_text = re.sub(pattern, repl, reformulated, flags=re.IGNORECASE)
            if new_text != reformulated:
                reformulated = new_text
                break

        # If query was very short like "Symptoms?" or "How to treat?"
        if len(clean_q.split()) <= 3 and topic.lower() not in reformulated.lower():
            reformulated = f"{clean_q} of {topic}"

        was_modified = (reformulated.lower().strip() != clean_q.lower().strip())
        return (reformulated if was_modified else clean_q), was_modified, topic

    def format_history_for_prompt(
        self,
        history: Optional[List[Dict[str, str]]],
        max_turns: int = 4,
        max_chars_per_turn: int = 350
    ) -> str:
        """
        Format recent conversation history into a concise, token-bounded
        string for LLM prompting.
        """
        if not history:
            return ""

        formatted_lines = []
        recent_turns = history[-max_turns:]
        for turn in recent_turns:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            truncated = content[:max_chars_per_turn]
            if len(content) > max_chars_per_turn:
                truncated += "..."
            formatted_lines.append(f"{role}: {truncated}")

        if not formatted_lines:
            return ""

        return "PRIOR CONVERSATION HISTORY:\n" + "\n".join(formatted_lines) + "\n\n"
