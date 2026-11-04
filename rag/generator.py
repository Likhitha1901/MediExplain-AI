import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import openai
from openai import OpenAI


load_dotenv()


# ============================================================
# CUSTOM GENERATOR EXCEPTIONS
# ============================================================

class MedicalGeneratorError(Exception):
    """Base exception for medical generator errors."""
    pass


class RateLimitExceededError(MedicalGeneratorError):
    """Raised when OpenAI rate limit or quota is exhausted."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        retry_after_text: Optional[str] = None,
        is_quota: bool = False,
        raw_error: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.retry_after_text = retry_after_text
        self.is_quota = is_quota
        self.raw_error = raw_error


class AuthenticationError(MedicalGeneratorError):
    """Raised when API key or authentication fails."""
    pass


# ============================================================
# HELPERS
# ============================================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds into a friendly human string."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m}m" if m else f"{h}h"
    elif seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s}s" if s else f"{m}m"
    elif seconds < 1:
        return f"{int(seconds * 1000)}ms"
    else:
        return f"{seconds:.1f}s" if (seconds % 1 != 0) else f"{int(seconds)}s"


def parse_retry_duration(text: str) -> Optional[Tuple[float, str]]:
    """Extract retry wait duration from provider error text."""
    if not text:
        return None

    pattern = r"try again in\s+((?:(\d+)h)?(?:(\d+)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+)ms)?)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None

    hours = int(match.group(2) or 0)
    minutes = int(match.group(3) or 0)
    seconds = float(match.group(4) or 0)
    milliseconds = float(match.group(5) or 0)

    total_seconds = hours * 3600 + minutes * 60 + seconds + (milliseconds / 1000.0)
    if total_seconds > 0:
        return total_seconds, match.group(1).strip()

    return None


def extract_rate_limit_info(exc: openai.RateLimitError) -> Tuple[Optional[float], Optional[str], bool]:
    """Inspect RateLimitError to determine wait duration and quota status."""
    error_msg = str(getattr(exc, "message", exc))
    body = getattr(exc, "body", None) or {}
    error_obj = body.get("error", {}) if isinstance(body, dict) else {}

    code = error_obj.get("code") or ""
    err_type = error_obj.get("type") or ""
    msg_lower = error_msg.lower()

    is_quota = (
        code == "insufficient_quota"
        or err_type == "insufficient_quota"
        or "quota" in msg_lower
        or "billing" in msg_lower
        or "credit balance" in msg_lower
        or "plan and billing" in msg_lower
    )

    retry_seconds: Optional[float] = None
    retry_text: Optional[str] = None

    parsed_msg = parse_retry_duration(error_msg)
    if parsed_msg:
        retry_seconds, retry_text = parsed_msg

    response = getattr(exc, "response", None)
    if response and hasattr(response, "headers"):
        headers = response.headers

        reset_tokens = headers.get("x-ratelimit-reset-tokens")
        if reset_tokens and not retry_text:
            parsed_hdr = parse_retry_duration(f"try again in {reset_tokens}")
            if parsed_hdr:
                retry_seconds, retry_text = parsed_hdr
            else:
                retry_text = reset_tokens

        raw_retry = headers.get("retry-after")
        if raw_retry and not retry_seconds:
            try:
                retry_seconds = float(raw_retry)
                if not retry_text:
                    retry_text = format_duration(retry_seconds)
            except (ValueError, TypeError):
                pass

    if retry_seconds is not None and not retry_text:
        retry_text = format_duration(retry_seconds)

    return retry_seconds, retry_text, is_quota


# ============================================================
# MEDICAL GENERATOR
# ============================================================

class MedicalGenerator:

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing from .env")

        if not model:
            raise ValueError("OPENAI_MODEL is missing from .env")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model

    def generate(
        self,
        question: str,
        context: str,
        language: str = "English",
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """
        Generate a patient-friendly medical explanation grounded in retrieved evidence.
        Supports multi-turn conversation history and handles provider rate-limits.
        """
        instructions = """You are MediExplain AI, a medical information assistant.
Your job is to explain medical information clearly, accurately, and in patient-friendly language.

Rules:
1. Use the provided medical evidence as the primary source.
2. Do not invent medical facts, statistics, or dosages.
3. Do not diagnose the user or make clinical declarations.
4. Do not claim that a symptom definitely means a disease.
5. Explain uncertainty clearly when evidence is limited.
6. Do not recommend changing or discontinuing prescription medicines.
7. For medication questions, explain purpose, side effects, precautions, and general dosage only when supported by evidence.
8. If evidence is insufficient, clearly say so.
9. Use simple, reassuring language that non-medical patients can understand.
10. Recommend consulting a qualified healthcare professional for personalized care.
11. Answer in the requested language."""

        # Format conversation history if provided (last 2-3 turns max)
        history_str = ""
        if history:
            valid_turns = []
            for item in history[-4:]:
                role = "User" if item.get("role") == "user" else "Assistant"
                content = str(item.get("content") or "").strip()
                if content:
                    valid_turns.append(f"{role}: {content[:300]}")
            if valid_turns:
                history_str = "PREVIOUS CONVERSATION CONTEXT:\n" + "\n".join(valid_turns) + "\n\n----------------------------------------\n\n"

        prompt = f"""MEDICAL EVIDENCE:
{context.strip()}

----------------------------------------

{history_str}USER QUESTION:
{question.strip()}

----------------------------------------

REQUESTED LANGUAGE:
{language}

Using the medical evidence above, provide a clear, patient-friendly explanation."""

        max_retries = 2
        initial_backoff = 1.0
        backoff_factor = 2.0
        max_wait_threshold = 8.0

        for attempt in range(max_retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    instructions=instructions,
                    input=prompt,
                    max_output_tokens=650
                )

                return {
                    "answer": response.output_text,
                    "language": language,
                    "model": self.model_name
                }

            except openai.RateLimitError as exc:
                retry_seconds, retry_text, is_quota = extract_rate_limit_info(exc)

                if is_quota:
                    raise RateLimitExceededError(
                        message=(
                            "The OpenAI API service is temporarily unavailable because the account "
                            "credit limit has been reached. Please check your account settings at platform.openai.com."
                        ),
                        retry_after=None,
                        retry_after_text=None,
                        is_quota=True,
                        raw_error=str(exc)
                    ) from exc

                # If wait time is excessive, do not hang the client
                if retry_seconds is not None and retry_seconds > max_wait_threshold:
                    wait_desc = retry_text or f"{int(retry_seconds)} seconds"
                    raise RateLimitExceededError(
                        message=(
                            f"The AI service has reached its token rate limit for model '{self.model_name}'. "
                            f"Provider wait estimate: {wait_desc}. Please try again later."
                        ),
                        retry_after=retry_seconds,
                        retry_after_text=wait_desc,
                        is_quota=False,
                        raw_error=str(exc)
                    ) from exc

                # Bounded backoff
                if attempt < max_retries:
                    sleep_time = (
                        retry_seconds + random.uniform(0.1, 0.4)
                        if (retry_seconds and retry_seconds <= max_wait_threshold)
                        else (initial_backoff * (backoff_factor ** attempt) + random.uniform(0.1, 0.4))
                    )
                    if sleep_time <= max_wait_threshold:
                        time.sleep(sleep_time)
                        continue

                wait_desc = retry_text or "a short while"
                raise RateLimitExceededError(
                    message=(
                        f"The AI service is temporarily busy (rate limit reached). "
                        f"Please try again in {wait_desc}."
                    ),
                    retry_after=retry_seconds,
                    retry_after_text=retry_text,
                    is_quota=False,
                    raw_error=str(exc)
                ) from exc

            except openai.AuthenticationError as exc:
                raise AuthenticationError(
                    "OpenAI API authentication failed. Please verify OPENAI_API_KEY in the backend configuration."
                ) from exc

            except openai.OpenAIError as exc:
                raise MedicalGeneratorError(
                    f"OpenAI API service error ({type(exc).__name__}): {str(exc)}"
                ) from exc

            except Exception as exc:
                raise MedicalGeneratorError(
                    f"Unexpected error during answer generation: {str(exc)}"
                ) from exc
