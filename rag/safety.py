import re
from typing import Dict, List, Optional


class MedicalSafety:
    """
    Evaluates patient queries for medical urgency, warnings,
    and educational intent to safeguard users.
    """

    def __init__(self):
        # --------------------------------------------------
        # EMERGENCY INDICATORS (Urgent Attention Required)
        # --------------------------------------------------
        self.emergency_keywords = [
            # Cardiac & Circulatory
            "severe chest pain",
            "crushing chest pain",
            "pressure in chest",
            "tightness in chest",
            "heart attack",
            "cardiac arrest",
            "heart pain",

            # Respiratory Emergency
            "difficulty breathing",
            "trouble breathing",
            "cannot breathe",
            "can't breathe",
            "severe breathlessness",
            "stopped breathing",
            "choking",
            "gasping for air",
            "stridor",

            # Neurological Emergency / Stroke
            "sudden weakness",
            "sudden numbness",
            "face drooping",
            "facial drooping",
            "slurred speech",
            "cannot speak",
            "can't speak",
            "loss of consciousness",
            "unconscious",
            "passed out",
            "unresponsive",
            "seizure",
            "convulsions",

            # Severe Bleeding & Trauma
            "severe bleeding",
            "heavy bleeding",
            "bleeding heavily",
            "vomiting blood",
            "coughing blood",
            "uncontrolled bleeding",
            "severe injury",
            "major trauma",

            # Severe Allergic Reaction / Anaphylaxis
            "anaphylaxis",
            "severe allergic reaction",
            "throat swelling",
            "swelling of throat",
            "swollen tongue and cannot breathe",

            # Poisoning & Acute Overdose
            "drug overdose",
            "medication overdose",
            "medicine overdose",
            "took extra pills",
            "accidentally took extra",
            "extra dose of medicine",
            "swallowed poison",
            "suspected poisoning",
            "poisoning",
            "poisoned",
            "overdose",

            # Self-Harm & Crisis
            "want to die",
            "suicide",
            "kill myself",
            "end my life"
        ]

        # --------------------------------------------------
        # WARNING INDICATORS (Professional Assessment Needed)
        # --------------------------------------------------
        self.warning_keywords = [
            # Fevers and Infections
            "high fever",
            "persistent fever",
            "fever with stiff neck",
            "fever not going down",

            # Severe Non-Emergency Pains
            "severe headache",
            "worst headache of my life",
            "severe abdominal pain",
            "severe stomach pain",
            "persistent pain",

            # Gastrointestinal & Excretory
            "persistent vomiting",
            "unable to keep fluids down",
            "blood in urine",
            "blood in stool",
            "black tarry stool",

            # Neurological / Balance
            "fainting",
            "dizziness",
            "vertigo",
            "confusion",
            "disoriented",

            # Medication Concerns
            "wrong dose",
            "double dose",
            "missed dose",
            "drug interaction",
            "medicine interaction",
            "mixing medications",
            "mixing medicines",
            "adverse drug reaction",
            "side effects from medication",

            # Progression
            "worsening symptoms",
            "symptoms getting worse",
            "condition deteriorating"
        ]

        # Educational phrasing patterns that soften false-positive emergencies
        self.educational_patterns = [
            r"^what\s+is\b",
            r"^what\s+are\b",
            r"^define\b",
            r"^definition\s+of\b",
            r"^explain\b",
            r"^tell\s+me\s+about\b",
            r"^overview\s+of\b",
            r"^how\s+does\b",
            r"^how\s+do\b",
            r"^causes\s+of\b",
            r"^what\s+causes\b",
            r"^can\s+you\s+explain\b",
            r"^information\s+on\b",
            r"^history\s+of\b",
            r"^types\s+of\b"
        ]

        # First-person symptom indicators indicating an active situation
        self.active_situation_patterns = [
            r"\bi\s+(have|am\s+having|feel|felt|got|took|swallowed)\b",
            r"\bmy\s+(chest|heart|head|throat|arm|face|body)\b",
            r"\bhelp\s+me\b",
            r"\bwhat\s+should\s+i\s+do\s+now\b",
            r"\bhappening\s+right\s+now\b",
            r"\bi\s+can['’]?t\b",
            r"\bi['’]?m\s+scared\b"
        ]

    def _is_purely_educational(self, text: str) -> bool:
        """
        Check if the question is an abstract/educational query rather
        than an active personal emergency symptom.
        """
        has_educational_prefix = any(
            re.search(pat, text, re.IGNORECASE) for pat in self.educational_patterns
        )
        has_active_signal = any(
            re.search(pat, text, re.IGNORECASE) for pat in self.active_situation_patterns
        )

        return has_educational_prefix and not has_active_signal

    def analyze(self, question: str) -> Dict:
        """
        Classifies medical urgency into: Emergency, Warning, or Normal.
        """
        if not question or not question.strip():
            return {
                "level": "Normal",
                "reason": "No question provided.",
                "matched_terms": []
            }

        # Normalize whitespace and case
        text = re.sub(r"\s+", " ", question.lower()).strip()

        # Check for educational intent
        is_educational = self._is_purely_educational(text)

        # --------------------------------------------------
        # 1. EMERGENCY CHECK
        # --------------------------------------------------
        emergency_matches = []
        for kw in self.emergency_keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text):
                emergency_matches.append(kw)

        if emergency_matches:
            if is_educational:
                return {
                    "level": "Warning",
                    "reason": (
                        f"The question inquires about an urgent condition ({', '.join(emergency_matches)}). "
                        "Providing educational information with guidance on recognizing emergencies."
                    ),
                    "matched_terms": emergency_matches,
                    "is_educational": True
                }

            return {
                "level": "Emergency",
                "reason": (
                    "The question contains symptoms or situations that may require "
                    "urgent emergency medical attention."
                ),
                "matched_terms": emergency_matches,
                "is_educational": False
            }

        # --------------------------------------------------
        # 2. WARNING CHECK
        # --------------------------------------------------
        warning_matches = []
        for kw in self.warning_keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text):
                warning_matches.append(kw)

        if warning_matches:
            return {
                "level": "Warning",
                "reason": (
                    "The question mentions symptoms or medication factors that may require "
                    "professional healthcare evaluation."
                ),
                "matched_terms": warning_matches,
                "is_educational": is_educational
            }

        # --------------------------------------------------
        # 3. NORMAL
        # --------------------------------------------------
        return {
            "level": "Normal",
            "reason": "No acute emergency or high-risk warning indicators detected.",
            "matched_terms": [],
            "is_educational": is_educational
        }

    def get_safety_message(self, safety_result: Dict, language: str = "English") -> str:
        """
        Return patient-friendly safety advice in the chosen language.
        """
        level = safety_result.get("level", "Normal")

        messages = {
            "English": {
                "Emergency": (
                    "🚨 POTENTIAL MEDICAL EMERGENCY: This query describes symptoms or situations that may require "
                    "immediate emergency medical care. If this is happening now, call your local emergency services "
                    "(e.g., 911, 112) or proceed to the nearest emergency department right away. "
                    "MediExplain AI cannot provide emergency triage or diagnosis."
                ),
                "Warning": (
                    "⚠️ HEALTHCARE EVALUATION RECOMMENDED: These symptoms or medication concerns warrant evaluation "
                    "by a qualified healthcare provider, especially if severe, persistent, or worsening. "
                    "Do not alter prescribed medications without consulting your doctor or pharmacist."
                ),
                "Normal": (
                    "ℹ️ EDUCATIONAL INFORMATION: This explanation is for general health literacy and educational purposes only. "
                    "It is not a clinical diagnosis or individualized medical advice. Always consult a licensed healthcare "
                    "professional for personal medical concerns."
                )
            },
            "Kannada": {
                "Emergency": (
                    "🚨 ತುರ್ತು ವೈದ್ಯಕೀಯ ಗಮನದ ಅಗತ್ಯವಿದೆ: ಈ ಪ್ರಶ್ನೆಯು ತಕ್ಷಣದ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿರುವ ತುರ್ತು ಪರಿಸ್ಥಿತಿಯನ್ನು ವಿವರಿಸಬಹುದು. "
                    "ಇದು ಈಗ ಸಂಭವಿಸುತ್ತಿದ್ದರೆ, ತಕ್ಷಣ ನಿಮ್ಮ ಸ್ಥಳೀಯ ತುರ್ತು ಸೇವೆಗೆ ಕರೆ ಮಾಡಿ ಅಥವಾ ಹತ್ತಿರದ ಆಸ್ಪತ್ರೆಗೆ ಭೇಟಿ ನೀಡಿ."
                ),
                "Warning": (
                    "⚠️ ವೈದ್ಯಕೀಯ ಸಲಹೆ ಶಿಫಾರಸು ಮಾಡಲಾಗಿದೆ: ಈ ಲಕ್ಷಣಗಳು ಅಥವಾ ಔಷಧಿ ಕಾಳಜಿಗಳಿಗೆ ವೈದ್ಯರ ಪರಿಶೀಲನೆ ಅಗತ್ಯವಿದೆ. "
                    "ವೈದ್ಯರ ಸಲಹೆಯಿಲ್ಲದೆ ಔಷಧಿಗಳನ್ನು ಬದಲಾಯಿಸಬೇಡಿ."
                ),
                "Normal": (
                    "ℹ️ ಶೈಕ್ಷಣಿಕ ಮಾಹಿತಿ: ಈ ಮಾಹಿತಿಯನ್ನು ಶೈಕ್ಷಣಿಕ ಉದ್ದೇಶಗಳಿಗಾಗಿ ಮಾತ್ರ ನೀಡಲಾಗಿದೆ ಮತ್ತು ಇದು ವೃತ್ತಿಪರ ವೈದ್ಯಕೀಯ ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ."
                )
            },
            "Hindi": {
                "Emergency": (
                    "🚨 संभावित आपातकालीन स्थिति: इस प्रश्न में ऐसे लक्षण हैं जिनके लिए तत्काल आपातकालीन चिकित्सा देखभाल की आवश्यकता हो सकती है। "
                    "यदि यह अभी हो रहा है, तो तुरंत अपनी स्थानीय आपातकालीन सेवा को कॉल करें या निकटतम अस्पताल जाएं।"
                ),
                "Warning": (
                    "⚠️ चिकित्सीय परामर्श अनुशंसित: इन लक्षणों या दवा संबंधी चिंताओं के लिए योग्य चिकित्सक से परामर्श करना चाहिए। "
                    "बिना डॉक्टर की सलाह के दवाएं न बदलें।"
                ),
                "Normal": (
                    "ℹ️ शैक्षणिक जानकारी: यह जानकारी केवल शैक्षणिक उद्देश्यों के लिए है और पेशेवर चिकित्सा सलाह का विकल्प नहीं है।"
                )
            }
        }

        lang_dict = messages.get(language, messages["English"])
        return lang_dict.get(level, lang_dict["Normal"])
