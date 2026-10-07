from fastapi import APIRouter
from pydantic import BaseModel

from rag.vector_store import MedicalVectorStore
from rag.reranker import MedicalReranker
from rag.generator import MedicalGenerator
from rag.safety import MedicalSafety


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["MediExplain AI"]
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str
    language: str = "English"


# ============================================================
# RESPONSE MODEL
# ============================================================

class SourceItem(BaseModel):
    title: str
    sources: str
    category: str
    url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    confidence: str
    safety_level: str
    safety_message: str


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

vector_store = MedicalVectorStore()
reranker = MedicalReranker()
generator = MedicalGenerator()
safety = MedicalSafety()


# ============================================================
# MULTILINGUAL FIXED MESSAGES
# ============================================================

MESSAGES = {

    "English": {
        "empty_question":
            "Please enter a medical question.",

        "empty_question_safety":
            "Please enter a question so MediExplain AI can provide relevant information.",

        "no_information":
            "I could not find reliable medical information relevant to your question in the MediExplain knowledge base.",

        "low_relevance":
            "I could not find sufficiently relevant medical information to answer this question.",

        "no_evidence":
            "I could not find sufficiently relevant medical evidence for this question."
    },

    "Kannada": {
        "empty_question":
            "ದಯವಿಟ್ಟು ವೈದ್ಯಕೀಯ ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ.",

        "empty_question_safety":
            "ಸಂಬಂಧಿತ ಮಾಹಿತಿಯನ್ನು ಒದಗಿಸಲು MediExplain AI ಗೆ ದಯವಿಟ್ಟು ಒಂದು ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ.",

        "no_information":
            "ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ವಿಶ್ವಾಸಾರ್ಹ ವೈದ್ಯಕೀಯ ಮಾಹಿತಿಯನ್ನು MediExplain ಜ್ಞಾನಕೋಶದಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",

        "low_relevance":
            "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಸಂಬಂಧಿತ ವೈದ್ಯಕೀಯ ಮಾಹಿತಿಯನ್ನು ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",

        "no_evidence":
            "ಈ ಪ್ರಶ್ನೆಗೆ ಸಾಕಷ್ಟು ಸಂಬಂಧಿತ ವೈದ್ಯಕೀಯ ಸಾಕ್ಷ್ಯವನ್ನು ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ."
    },

    "Hindi": {
        "empty_question":
            "कृपया एक चिकित्सा प्रश्न दर्ज करें।",

        "empty_question_safety":
            "प्रासंगिक जानकारी प्रदान करने के लिए कृपया एक प्रश्न दर्ज करें।",

        "no_information":
            "मुझे MediExplain के ज्ञान आधार में आपके प्रश्न से संबंधित विश्वसनीय चिकित्सा जानकारी नहीं मिली।",

        "low_relevance":
            "इस प्रश्न का उत्तर देने के लिए पर्याप्त प्रासंगिक चिकित्सा जानकारी नहीं मिली।",

        "no_evidence":
            "इस प्रश्न के लिए पर्याप्त प्रासंगिक चिकित्सा प्रमाण नहीं मिले।"
    },

    "Telugu": {
        "empty_question":
            "దయచేసి వైద్య ప్రశ్నను నమోదు చేయండి.",

        "empty_question_safety":
            "సంబంధిత సమాచారాన్ని అందించడానికి దయచేసి ఒక ప్రశ్నను నమోదు చేయండి.",

        "no_information":
            "మీ ప్రశ్నకు సంబంధించిన విశ్వసనీయ వైద్య సమాచారాన్ని MediExplain జ్ఞాన ఆధారంలో కనుగొనలేకపోయాను.",

        "low_relevance":
            "ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి తగిన సంబంధిత వైద్య సమాచారం కనుగొనబడలేదు.",

        "no_evidence":
            "ఈ ప్రశ్నకు తగిన సంబంధిత వైద్య ఆధారాలు కనుగొనబడలేదు."
    },

    "Tamil": {
        "empty_question":
            "தயவுசெய்து மருத்துவக் கேள்வியை உள்ளிடவும்.",

        "empty_question_safety":
            "தொடர்புடைய தகவலை வழங்க தயவுசெய்து ஒரு கேள்வியை உள்ளிடவும்.",

        "no_information":
            "உங்கள் கேள்விக்கு தொடர்புடைய நம்பகமான மருத்துவ தகவலை MediExplain அறிவுத் தளத்தில் கண்டறிய முடியவில்லை.",

        "low_relevance":
            "இந்தக் கேள்விக்கு பதிலளிக்க போதுமான தொடர்புடைய மருத்துவ தகவல் கிடைக்கவில்லை.",

        "no_evidence":
            "இந்தக் கேள்விக்கு போதுமான தொடர்புடைய மருத்துவ ஆதாரங்கள் கிடைக்கவில்லை."
    }
}


# ============================================================
# SAFETY MESSAGE TRANSLATION
# ============================================================

SAFETY_MESSAGES = {

    "English":
        "This information is for educational purposes and is not a substitute for professional medical advice.",

    "Kannada":
        "ಈ ಮಾಹಿತಿಯನ್ನು ಶೈಕ್ಷಣಿಕ ಉದ್ದೇಶಗಳಿಗಾಗಿ ಮಾತ್ರ ನೀಡಲಾಗಿದೆ ಮತ್ತು ಇದು ವೃತ್ತಿಪರ ವೈದ್ಯಕೀಯ ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ.",

    "Hindi":
        "यह जानकारी केवल शैक्षणिक उद्देश्यों के लिए है और पेशेवर चिकित्सा सलाह का विकल्प नहीं है।",

    "Telugu":
        "ఈ సమాచారం విద్యా ప్రయోజనాల కోసం మాత్రమే అందించబడింది మరియు ఇది వృత్తిపరమైన వైద్య సలహాకు ప్రత్యామ్నాయం కాదు.",

    "Tamil":
        "இந்தத் தகவல் கல்வி நோக்கங்களுக்காக மட்டுமே வழங்கப்படுகிறது மற்றும் இது தொழில்முறை மருத்துவ ஆலோசனைக்கு மாற்றாகாது."
}


def get_language(language: str) -> str:
    """
    Make sure the requested language matches
    one of our supported languages.
    """

    if language in MESSAGES:
        return language

    return "English"


def get_safety_message(language: str, safety_result: dict) -> str:
    """
    Return a localized safety message.
    """

    # For now, use our controlled educational safety message.
    # Later we can make urgent/high-risk messages multilingual too.
    return SAFETY_MESSAGES.get(
        language,
        SAFETY_MESSAGES["English"]
    )


# ============================================================
# CHAT API
# ============================================================

@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    # --------------------------------------------------------
    # 1. GET USER QUESTION
    # --------------------------------------------------------

    question = request.message.strip()

    language = get_language(request.language)

    messages = MESSAGES[language]

    if not question:

        return ChatResponse(
            answer=messages["empty_question"],
            sources=[],
            confidence="Low",
            safety_level="Normal",
            safety_message=messages["empty_question_safety"]
        )


    # --------------------------------------------------------
    # 2. MEDICAL SAFETY ANALYSIS
    # --------------------------------------------------------

    safety_result = safety.analyze(question)

    safety_level = safety_result["level"]

    safety_message = get_safety_message(
        language,
        safety_result
    )


    # --------------------------------------------------------
    # 3. RETRIEVE FROM CHROMADB
    # --------------------------------------------------------

    results = vector_store.search(
        query=question,
        top_k=10
    )


    # --------------------------------------------------------
    # 4. CONVERT CHROMADB RESULTS
    # --------------------------------------------------------

    documents = []

    if results and results.get("documents"):

        retrieved_documents = results["documents"][0]

        retrieved_metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        for index, content in enumerate(
            retrieved_documents
        ):

            metadata = {}

            if index < len(retrieved_metadatas):
                metadata = (
                    retrieved_metadatas[index]
                    or {}
                )

            documents.append({
                "content": content,
                "source": metadata.get(
                    "source",
                    ""
                ),
                "title": metadata.get(
                    "title",
                    ""
                ),
                "category": metadata.get(
                    "category",
                    ""
                ),
                "url": metadata.get(
                    "url",
                    ""
                )
            })


    # --------------------------------------------------------
    # 5. NO MEDICAL INFORMATION FOUND
    # --------------------------------------------------------

    if not documents:

        return ChatResponse(
            answer=messages["no_information"],
            sources=[],
            confidence="Low",
            safety_level=safety_level,
            safety_message=safety_message
        )


    # --------------------------------------------------------
    # 6. RERANK DOCUMENTS
    # --------------------------------------------------------

    reranked_results = reranker.rerank(
        query=question,
        documents=documents,
        top_k=3
    )


    if not reranked_results:

        return ChatResponse(
            answer=messages["low_relevance"],
            sources=[],
            confidence="Low",
            safety_level=safety_level,
            safety_message=safety_message
        )


    # --------------------------------------------------------
    # 7. BUILD MEDICAL CONTEXT
    # --------------------------------------------------------

    context_parts = []
    sources = []

    best_score = reranked_results[0]["score"]


    for result in reranked_results:

        document = result["document"]

        score = result["score"]

        # Ignore very weak documents
        #if score < best_score * 0.70:
            #continue


        title = document.get(
            "title",
            "Medical Information"
        )

        source = document.get(
            "source",
            "Unknown source"
        )

        category = document.get(
            "category",
            ""
        )

        url = document.get(
            "url",
            ""
        )

        content = document.get(
            "content",
            ""
        )


        # ----------------------------------------------------
        # MEDICAL EVIDENCE FOR LLM
        # ----------------------------------------------------

        context_parts.append(
            f"""
TITLE: {title}

SOURCE: {source}

CATEGORY: {category}

URL: {url}

MEDICAL CONTENT:
{content}

RELEVANCE SCORE:
{score:.4f}

----------------------------------------
"""
        )


        # ----------------------------------------------------
        # SOURCE CITATION
        # ----------------------------------------------------

        source_item = {
            "title": title,
            "sources": source,
            "category": category,
            "url": url,
        }

        if source_item not in sources:
            sources.append(source_item)


    # --------------------------------------------------------
    # 8. CHECK CONTEXT
    # --------------------------------------------------------

    if not context_parts:

        return ChatResponse(
            answer=messages["no_evidence"],
            sources=[],
            confidence="Low",
            safety_level=safety_level,
            safety_message=safety_message
        )


    context = "\n".join(context_parts)


    # --------------------------------------------------------
    # 9. GENERATE PATIENT-FRIENDLY ANSWER
    # --------------------------------------------------------

    llm_response = generator.generate(
        question=question,
        context=context,
        language=language
    )


    # --------------------------------------------------------
    # 10. CONFIDENCE
    # --------------------------------------------------------

    if best_score >= 0.60:

        confidence = "High"

    elif best_score >= 0.35:

        confidence = "Medium"

    else:

        confidence = "Low"


    # --------------------------------------------------------
    # 11. RETURN FINAL RESPONSE
    # --------------------------------------------------------

    return ChatResponse(
        answer=llm_response["answer"],
        sources=sources,
        confidence=confidence,
        safety_level=safety_level,
        safety_message=safety_message
    )