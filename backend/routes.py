import base64
import os
import re
import uuid
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

from rag.vector_store import MedicalVectorStore
from rag.reranker import MedicalReranker
from rag.generator import (
    MedicalGenerator,
    RateLimitExceededError,
    AuthenticationError,
    MedicalGeneratorError
)
from rag.safety import MedicalSafety
from rag.citations import extract_supporting_passage, map_claims_to_sources


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["MediExplain AI"]
)


# ============================================================
# IN-MEMORY STORE FOR UPLOADED USER DOCUMENTS
# ============================================================

UPLOADED_DOCUMENTS: Dict[str, Dict] = {}


# ============================================================
# REQUEST & RESPONSE MODELS
# ============================================================

class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    language: str = "English"
    history: Optional[List[ChatHistoryItem]] = None
    doc_id: Optional[str] = None


class SourceItem(BaseModel):
    title: str
    sources: str
    category: str
    url: Optional[str] = None
    excerpt: Optional[str] = None
    claims: Optional[List[str]] = None


class TerminologyItem(BaseModel):
    term: str
    definition: str
    importance: str
    example: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    confidence: str
    safety_level: str
    safety_message: str
    evidence_explorer: Optional[List[Dict]] = None
    terminology: Optional[List[TerminologyItem]] = None


class ReportRequest(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    confidence: str
    safety_level: str
    safety_message: str
    language: str = "English"


class ReportResponse(BaseModel):
    report_title: str
    question: str
    plain_explanation: str
    key_facts: List[str]
    medical_terms: List[TerminologyItem]
    precautions: List[str]
    sources: List[SourceItem]
    disclaimer: str


class TreatmentCompareRequest(BaseModel):
    condition: str
    language: str = "English"


class TreatmentOption(BaseModel):
    treatment: str
    category: str
    benefits: str
    limitations: str
    risks: str
    evidence_source: str


class TreatmentCompareResponse(BaseModel):
    condition: str
    overview: str
    options: List[TreatmentOption]
    disclaimer: str


class DoctorSummaryRequest(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    language: str = "English"


class DoctorSummaryResponse(BaseModel):
    topic: str
    summary_for_doctor: str
    questions_to_ask: List[str]
    critical_symptoms_to_mention: List[str]
    disclaimer: str


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
        "empty_question": "Please enter a medical question.",
        "empty_question_safety": "Please enter a question so MediExplain AI can provide relevant information.",
        "no_information": "I could not find reliable medical information relevant to your question in the MediExplain knowledge base.",
        "low_relevance": "I could not find sufficiently relevant medical information to answer this question accurately.",
        "no_evidence": "I could not find sufficiently relevant medical evidence for this question."
    },
    "Kannada": {
        "empty_question": "ದಯವಿಟ್ಟು ವೈದ್ಯಕೀಯ ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ.",
        "empty_question_safety": "ಸಂಬಂಧಿತ ಮಾಹಿತಿಯನ್ನು ಒದಗಿಸಲು MediExplain AI ಗೆ ದಯವಿಟ್ಟು ಒಂದು ಪ್ರಶ್ನೆಯನ್ನು ನಮೂದಿಸಿ.",
        "no_information": "ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ವಿಶ್ವಾಸಾರ್ಹ ವೈದ್ಯಕೀಯ ಮಾಹಿತಿಯನ್ನು MediExplain ಜ್ಞಾನಕೋಶದಲ್ಲಿ ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
        "low_relevance": "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಸಾಕಷ್ಟು ಸಂಬಂಧಿತ ವೈದ್ಯಕೀಯ ಮಾಹಿತಿಯನ್ನು ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ.",
        "no_evidence": "ಈ ಪ್ರಶ್ನೆಗೆ ಸಾಕಷ್ಟು ಸಂಬಂಧಿತ ವೈದ್ಯಕೀಯ ಸಾಕ್ಷ್ಯವನ್ನು ಕಂಡುಹಿಡಿಯಲಾಗಲಿಲ್ಲ."
    },
    "Hindi": {
        "empty_question": "कृपया एक चिकित्सा प्रश्न दर्ज करें।",
        "empty_question_safety": "प्रासंगिक जानकारी प्रदान करने के लिए कृपया एक प्रश्न दर्ज करें।",
        "no_information": "मुझे MediExplain के ज्ञान आधार में आपके प्रश्न से संबंधित विश्वसनीय चिकित्सा जानकारी नहीं मिली।",
        "low_relevance": "इस प्रश्न का उत्तर देने के लिए पर्याप्त प्रासंगिक चिकित्सा जानकारी नहीं मिली।",
        "no_evidence": "इस प्रश्न के लिए पर्याप्त प्रासंगिक चिकित्सा प्रमाण नहीं मिले।"
    }
}


def get_language(language: str) -> str:
    if language in MESSAGES:
        return language
    return "English"


# ============================================================
# MEDICAL TERMINOLOGY SIMPLIFIER KNOWLEDGE
# ============================================================

MEDICAL_TERMS_DB = {
    "hypertension": {
        "term": "Hypertension",
        "definition": "High blood pressure inside your arteries.",
        "importance": "Over time, it forces your heart to work harder and can damage blood vessels.",
        "example": "A blood pressure reading of 140/90 mmHg or higher."
    },
    "hyperglycemia": {
        "term": "Hyperglycemia",
        "definition": "An unusually high concentration of glucose (sugar) circulating in the blood.",
        "importance": "Common in diabetes; persistent high sugar levels can harm nerves, kidneys, and eyes.",
        "example": "Feeling extremely thirsty and having a fasting blood sugar above 126 mg/dL."
    },
    "hypoglycemia": {
        "term": "Hypoglycemia",
        "definition": "Abnormally low blood sugar level (typically below 70 mg/dL).",
        "importance": "Brain cells require glucose to function; severe drops can cause shakiness, confusion, or fainting.",
        "example": "Sweating and trembling after taking insulin without eating enough food."
    },
    "dyspnea": {
        "term": "Dyspnea",
        "definition": "The medical term for shortness of breath or laboured breathing.",
        "importance": "Can be an indicator of lung conditions (like asthma or pneumonia) or heart disease.",
        "example": "Feeling winded or unable to catch your breath after climbing stairs."
    },
    "edema": {
        "term": "Edema",
        "definition": "Swelling caused by excess fluid trapped in your body's tissues.",
        "importance": "Frequently occurs in feet, ankles, and legs, and may signal heart, kidney, or vein issues.",
        "example": "Puffy ankles that leave an indentation when pressed with a finger."
    },
    "anaphylaxis": {
        "term": "Anaphylaxis",
        "definition": "A severe, potentially life-threatening allergic reaction affecting the entire body.",
        "importance": "Requires immediate emergency treatment with epinephrine because airway swelling can block breathing.",
        "example": "Sudden hives, wheezing, and throat tightness within minutes of eating peanuts."
    },
    "arrhythmia": {
        "term": "Arrhythmia",
        "definition": "An irregular heartbeat — beating too fast, too slow, or erratically.",
        "importance": "Affects how efficiently blood is pumped to vital organs.",
        "example": "Feeling a sudden fluttering sensation or skipped beat in your chest."
    },
    "neuropathy": {
        "term": "Neuropathy",
        "definition": "Damage or dysfunction of one or more peripheral nerves.",
        "importance": "Common in long-term diabetes; reduces protective sensation in feet and fingers.",
        "example": "A tingling, burning 'pins and needles' feeling in your toes."
    },
    "bronchodilator": {
        "term": "Bronchodilator",
        "definition": "A medication that relaxes and widens the airway passages in the lungs.",
        "importance": "Makes breathing easier during asthma flare-ups or COPD exacerbations.",
        "example": "Using an albuterol inhaler during an asthma attack."
    },
    "analgesic": {
        "term": "Analgesic",
        "definition": "A medication used specifically to relieve pain without causing loss of consciousness.",
        "importance": "Helps manage acute or chronic discomfort.",
        "example": "Paracetamol, acetaminophen, or ibuprofen taken for a headache."
    }
}


def extract_terms(text: str) -> List[TerminologyItem]:
    found = []
    text_lower = text.lower()
    for key, data in MEDICAL_TERMS_DB.items():
        if key in text_lower:
            found.append(TerminologyItem(**data))
    return found[:3]


# ============================================================
# CHAT API
# ============================================================

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.message.strip()
    language = get_language(request.language)
    messages = MESSAGES[language]

    # 1. Validation
    if not question:
        return ChatResponse(
            answer=messages["empty_question"],
            sources=[],
            confidence="Low",
            safety_level="Normal",
            safety_message=safety.get_safety_message({"level": "Normal"}, language),
            evidence_explorer=[],
            terminology=[]
        )

    # 2. Medical Safety Analysis
    safety_result = safety.analyze(question)
    safety_level = safety_result["level"]
    safety_message = safety.get_safety_message(safety_result, language)

    documents = []

    # 3. Retrieve from Uploaded Document OR Persistent ChromaDB
    if request.doc_id and request.doc_id in UPLOADED_DOCUMENTS:
        doc_info = UPLOADED_DOCUMENTS[request.doc_id]
        chunks = doc_info.get("chunks", [])
        for chunk in chunks:
            documents.append({
                "content": chunk["text"],
                "source": f"Uploaded PDF ({doc_info['filename']})",
                "title": f"{doc_info['filename']} - Page {chunk['page']}",
                "category": "uploaded_document",
                "url": ""
            })
    else:
        results = vector_store.search(query=question, top_k=10)
        if results and results.get("documents"):
            retrieved_documents = results["documents"][0]
            retrieved_metadatas = results.get("metadatas", [[]])[0]
            for index, content in enumerate(retrieved_documents):
                metadata = retrieved_metadatas[index] if index < len(retrieved_metadatas) else {}
                metadata = metadata or {}
                documents.append({
                    "content": content,
                    "source": metadata.get("source", "MedlinePlus"),
                    "title": metadata.get("title", "Medical Reference"),
                    "category": metadata.get("category", "medical_information"),
                    "url": metadata.get("url", "")
                })

    if not documents:
        return ChatResponse(
            answer=messages["no_information"],
            sources=[],
            confidence="Low",
            safety_level=safety_level,
            safety_message=safety_message,
            evidence_explorer=[],
            terminology=[]
        )

    # 4. Rerank Documents with Hybrid Reranker
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
            safety_message=safety_message,
            evidence_explorer=[],
            terminology=[]
        )

    # 5. Build Medical Context and Sources
    context_parts = []
    sources: List[SourceItem] = []
    evidence_explorer = []
    best_score = reranked_results[0]["score"]

    for result in reranked_results:
        doc = result["document"]
        score = result["score"]

        title = doc.get("title", "Medical Reference")
        source = doc.get("source", "MedlinePlus")
        category = doc.get("category", "medical_information")
        url = doc.get("url", "")
        content = doc.get("content", "")[:2500]

        context_parts.append(
            f"TITLE: {title}\n"
            f"SOURCE: {source}\n"
            f"CATEGORY: {category}\n"
            f"MEDICAL CONTENT:\n{content.strip()}\n"
            "----------------------------------------"
        )

        excerpt = extract_supporting_passage(question, content)

        source_item = SourceItem(
            title=title,
            sources=source,
            category=category,
            url=url if url else None,
            excerpt=excerpt
        )

        if not any(s.title == title for s in sources):
            sources.append(source_item)

        evidence_explorer.append({
            "title": title,
            "category": category,
            "url": url,
            "relevance_score": round(score, 4),
            "excerpt": excerpt
        })

    if not context_parts:
        return ChatResponse(
            answer=messages["no_evidence"],
            sources=[],
            confidence="Low",
            safety_level=safety_level,
            safety_message=safety_message,
            evidence_explorer=[],
            terminology=[]
        )

    context = "\n".join(context_parts)

    # 6. Generate Patient-Friendly Answer
    history_dicts = [h.model_dump() for h in request.history] if request.history else None

    try:
        llm_response = generator.generate(
            question=question,
            context=context,
            language=language,
            history=history_dicts
        )
        answer_text = llm_response["answer"]
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail={
                "error_type": "quota_exceeded" if exc.is_quota else "rate_limit",
                "message": exc.message,
                "retry_after": exc.retry_after_text,
                "is_quota": exc.is_quota
            }
        ) from exc
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail={"error_type": "auth_error", "message": str(exc)}
        ) from exc
    except MedicalGeneratorError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error_type": "generator_error", "message": str(exc)}
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error_type": "server_error", "message": f"Answer generation error: {str(exc)}"}
        ) from exc

    # 7. Confidence Calculation
    if best_score >= 0.55:
        confidence = "High"
    elif best_score >= 0.35:
        confidence = "Medium"
    else:
        confidence = "Low"

    # 8. Extract Terminology & Claim Mapping
    terms = extract_terms(question + " " + answer_text)

    # Attach supported claims to sources
    claim_mappings = map_claims_to_sources(
        answer_text,
        [{"title": s.title, "category": s.category, "url": s.url or "", "content": s.excerpt or ""} for s in sources]
    )
    for src in sources:
        for cm in claim_mappings:
            if cm["title"] == src.title and cm.get("supported_claims"):
                src.claims = cm["supported_claims"]

    return ChatResponse(
        answer=answer_text,
        sources=sources,
        confidence=confidence,
        safety_level=safety_level,
        safety_message=safety_message,
        evidence_explorer=evidence_explorer,
        terminology=terms
    )


# ============================================================
# DOCUMENT UPLOAD API (PDF Q&A)
# ============================================================

class UploadDocRequest(BaseModel):
    filename: str
    file_base64: str


@router.post("/upload-doc")
def upload_document(req: UploadDocRequest):
    filename = req.filename or "uploaded_document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail={"error_type": "invalid_file", "message": "Only PDF (.pdf) medical documents are currently supported."}
        )

    try:
        content_bytes = base64.b64decode(req.file_base64)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={"error_type": "invalid_encoding", "message": "Invalid base64 document encoding."}
        )

    if len(content_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail={"error_type": "file_too_large", "message": "Uploaded file exceeds the maximum 10 MB limit."}
        )

    # Extract text with pypdf
    import io
    try:
        reader = PdfReader(io.BytesIO(content_bytes))
        num_pages = len(reader.pages)
        chunks = []
        for p_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = re.sub(r"\s+", " ", page_text).strip()
            if not page_text:
                continue

            # Split into ~600 character chunks
            paras = [p.strip() for p in page_text.split(". ") if len(p.strip()) > 20]
            current_chunk = ""
            for p in paras:
                if len(current_chunk) + len(p) > 600:
                    chunks.append({"page": p_idx + 1, "text": current_chunk.strip()})
                    current_chunk = p + ". "
                else:
                    current_chunk += p + ". "
            if current_chunk.strip():
                chunks.append({"page": p_idx + 1, "text": current_chunk.strip()})

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail={"error_type": "empty_text", "message": "Could not extract text from this PDF. It may be a scanned image without OCR."}
            )

        doc_id = str(uuid.uuid4())
        UPLOADED_DOCUMENTS[doc_id] = {
            "filename": filename,
            "pages": num_pages,
            "chunks": chunks
        }

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": filename,
            "pages": num_pages,
            "chunks_count": len(chunks),
            "message": f"Successfully parsed '{filename}' ({num_pages} pages). You can now ask questions about this document."
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error_type": "parse_error", "message": f"Failed to extract document contents: {str(exc)}"}
        )


# ============================================================
# AI-GENERATED STRUCTURED MEDICAL REPORT API
# ============================================================

@router.post("/generate-report", response_model=ReportResponse)
def generate_report(req: ReportRequest):
    # Extract plain explanation
    plain_explanation = req.answer.strip()
    if not plain_explanation:
        plain_explanation = "No explanation available for this report."

    # Extract key factual bullet points from answer
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"[*#_`]", "", req.answer)) if len(s.strip()) > 25]
    key_facts = sentences[:4] if sentences else ["Evidence retrieved from trusted medical sources."]

    # Extract medical terms
    terms = extract_terms(req.question + " " + req.answer)

    # General precautions based on safety level
    precautions = [
        "This report is generated for personal educational reference and health literacy only.",
        "Always discuss diagnosis, laboratory interpretations, or symptoms with a licensed doctor.",
        "Do not stop, start, or modify prescribed medications based on this automated report."
    ]
    if req.safety_level in ["Emergency", "Warning"]:
        precautions.insert(0, req.safety_message)

    report_title = f"MediExplain AI Educational Report: {req.question[:50].strip()}"

    disclaimer = (
        "CONFIDENTIAL PATIENT EDUCATION ONLY — NOT A CLINICAL DIAGNOSIS. "
        "This automated report synthesizes medical literature for informational purposes. "
        "It does not establish a doctor-patient relationship."
    )

    return ReportResponse(
        report_title=report_title,
        question=req.question,
        plain_explanation=plain_explanation,
        key_facts=key_facts,
        medical_terms=terms,
        precautions=precautions,
        sources=req.sources,
        disclaimer=disclaimer
    )


# ============================================================
# TREATMENT COMPARISON API
# ============================================================

TREATMENTS_DATABASE = {
    "diabetes": [
        TreatmentOption(
            treatment="Lifestyle Modification (Diet & Aerobic Exercise)",
            category="Non-pharmacological",
            benefits="Improves insulin sensitivity, supports weight management, reduces cardiovascular risk.",
            limitations="Requires continuous long-term adherence; often insufficient alone as disease progresses.",
            risks="Low risk; hypoglycemia risk if combined with certain medications without food adjustment.",
            evidence_source="MedlinePlus & NIDDK Diabetes Guidelines"
        ),
        TreatmentOption(
            treatment="Metformin (First-line Biguanide Medication)",
            category="Oral Medication",
            benefits="Lowers hepatic glucose production, does not cause hypoglycemia as monotherapy, weight neutral.",
            limitations="May take several weeks for optimal glycemic control; requires renal function monitoring.",
            risks="Common gastrointestinal upset (nausea, diarrhea); rare risk of lactic acidosis.",
            evidence_source="MedlinePlus Drug Information: Metformin"
        ),
        TreatmentOption(
            treatment="Insulin Therapy (Basal / Bolus)",
            category="Injectable Biological Therapy",
            benefits="Highly effective at reducing glucose levels regardless of underlying beta-cell function.",
            limitations="Requires frequent blood glucose monitoring, injections, and careful dose titration.",
            risks="Risk of hypoglycemia if dose exceeds carbohydrate intake; possible weight gain.",
            evidence_source="MedlinePlus: Insulin Information"
        )
    ],
    "asthma": [
        TreatmentOption(
            treatment="Short-Acting Beta2-Agonists (SABA Inhalers, e.g. Albuterol)",
            category="Quick-Relief Medication",
            benefits="Rapidly relaxes airway smooth muscles within minutes to relieve acute shortness of breath and wheezing.",
            limitations="Does not treat underlying chronic airway inflammation; effects wear off after 4-6 hours.",
            risks="Temporary rapid heartbeat (tachycardia), muscle shakiness, nervousness.",
            evidence_source="MedlinePlus & NHLBI Asthma Guidelines"
        ),
        TreatmentOption(
            treatment="Inhaled Corticosteroids (ICS, e.g. Fluticasone)",
            category="Long-Term Control Controller",
            benefits="Reduces airway swelling and mucus over time, preventing severe asthma attacks and emergency visits.",
            limitations="Must be taken daily as prescribed; does not provide immediate rescue during an acute attack.",
            risks="Oral thrush (fungal infection) and hoarseness if mouth is not rinsed after inhalation.",
            evidence_source="MedlinePlus: Inhaled Steroids"
        ),
        TreatmentOption(
            treatment="Allergen & Environmental Trigger Avoidance",
            category="Non-pharmacological Prevention",
            benefits="Reduces flare-up frequency by limiting exposure to smoke, pollen, dust mites, or cold air.",
            limitations="Difficult to achieve complete avoidance in outdoor or occupational settings.",
            risks="Negligible physical risk.",
            evidence_source="MedlinePlus: Asthma Overview"
        )
    ],
    "hypertension": [
        TreatmentOption(
            treatment="Dietary Approaches to Stop Hypertension (DASH Diet) & Sodium Reduction",
            category="Lifestyle Intervention",
            benefits="Proven reduction in systolic and diastolic blood pressure, enhances heart health.",
            limitations="Requires sustained dietary discipline; blood pressure reduction is moderate.",
            risks="Minimal; gradual dietary transition recommended.",
            evidence_source="MedlinePlus & American Heart Association"
        ),
        TreatmentOption(
            treatment="ACE Inhibitors / ARBs (e.g. Lisinopril, Losartan)",
            category="Prescription Medication",
            benefits="Relaxes blood vessels, protects kidney function especially in patients with diabetes.",
            limitations="Requires regular blood tests for potassium and kidney filtration monitoring.",
            risks="Dry cough (with ACE inhibitors), elevated blood potassium, dizziness upon standing.",
            evidence_source="MedlinePlus: Blood Pressure Medicines"
        ),
        TreatmentOption(
            treatment="Calcium Channel Blockers (e.g. Amlodipine)",
            category="Prescription Medication",
            benefits="Prevents calcium from entering heart and artery muscle cells, causing vessel relaxation.",
            limitations="May not be first choice in patients with specific forms of heart failure.",
            risks="Lower leg/ankle edema (swelling), flushing, headache.",
            evidence_source="MedlinePlus: Blood Pressure Medicines"
        )
    ]
}


@router.post("/treatment-compare", response_model=TreatmentCompareResponse)
def compare_treatments(req: TreatmentCompareRequest):
    cond_lower = req.condition.lower()
    matched_key = None
    for k in TREATMENTS_DATABASE:
        if k in cond_lower:
            matched_key = k
            break

    if not matched_key:
        matched_key = "diabetes"  # Default reference condition

    options = TREATMENTS_DATABASE[matched_key]
    overview = (
        f"Comparative overview of evidence-based treatments commonly evaluated for {matched_key.capitalize()}. "
        "Treatment choice depends on patient age, comorbidities, severity, and clinical evaluation."
    )
    disclaimer = (
        "Educational comparison only. Do not initiate or discontinue any medical treatment without "
        "personalized guidance from your physician."
    )

    return TreatmentCompareResponse(
        condition=matched_key.capitalize(),
        overview=overview,
        options=options,
        disclaimer=disclaimer
    )


# ============================================================
# DOCTOR DISCUSSION SUMMARY API
# ============================================================

@router.post("/doctor-summary", response_model=DoctorSummaryResponse)
def get_doctor_summary(req: DoctorSummaryRequest):
    clean_q = req.question.strip()

    questions = [
        f"What is the most likely cause of my specific symptoms regarding {clean_q}?",
        "Are there any diagnostic laboratory tests or imaging scans you recommend?",
        "What are the benefits, potential side effects, and monitoring requirements of treatment options?",
        "What lifestyle or dietary modifications should I follow alongside any prescribed regimen?",
        "What warning signs or 'red-flag' symptoms should prompt me to contact you or seek urgent care?"
    ]

    critical_symptoms = [
        "Sudden, severe chest pressure or shortness of breath",
        "Severe dizziness, sudden confusion, or loss of consciousness",
        "Uncontrolled bleeding or persistent inability to keep fluids down",
        "Signs of severe allergic reaction (lip/throat swelling, difficulty swallowing)"
    ]

    summary_text = (
        f"The patient inquired about: '{clean_q}'. Evidence-based medical education was reviewed covering "
        "established symptoms, precautions, and general clinical background. The patient is advised to discuss "
        "individualized clinical diagnosis and prescription management with their attending physician."
    )

    return DoctorSummaryResponse(
        topic=clean_q[:60],
        summary_for_doctor=summary_text,
        questions_to_ask=questions,
        critical_symptoms_to_mention=critical_symptoms,
        disclaimer="Prepared by MediExplain AI for patient-doctor communication. Not a medical chart or diagnosis."
    )
