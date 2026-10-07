import re


class MedicalSafety:

    def __init__(self):

        # --------------------------------------------------
        # EMERGENCY INDICATORS
        # --------------------------------------------------

        self.emergency_keywords = [

            # Cardiac
            "severe chest pain",
            "chest pain",
            "pressure in chest",
            "tightness in chest",
            "heart attack",
            "heart pain",

            # Breathing
            "difficulty breathing",
            "trouble breathing",
            "cannot breathe",
            "can't breathe",
            "shortness of breath",
            "severe breathlessness",

            # Neurological emergency
            "sudden weakness",
            "sudden numbness",
            "face drooping",
            "slurred speech",
            "cannot speak",
            "can't speak",
            "loss of consciousness",
            "unconscious",
            "seizure",

            # Severe bleeding
            "severe bleeding",
            "heavy bleeding",
            "bleeding heavily",
            "vomiting blood",
            "coughing blood",

            # Severe allergic reaction
            "severe allergic reaction",
            "anaphylaxis",
            "swelling of throat",
            "throat swelling",

            # Poisoning / overdose
            "overdose",
            "drug overdose",
            "poisoning",
            "poisoned",

            # Severe injury
            "severe injury",
            "major accident",
            "serious accident"
        ]


        # --------------------------------------------------
        # WARNING INDICATORS
        # --------------------------------------------------

        self.warning_keywords = [

            "high fever",
            "persistent fever",
            "severe headache",
            "persistent vomiting",
            "severe abdominal pain",
            "severe stomach pain",
            "blood in urine",
            "blood in stool",
            "fainting",
            "dizziness",
            "persistent pain",
            "worsening symptoms",
            "symptoms getting worse"
        ]


    # ======================================================
    # ANALYZE QUESTION
    # ======================================================

    def analyze(self, question: str):

        if not question:

            return {
                "level": "Normal",
                "reason": "No question provided."
            }


        text = question.lower().strip()


        # --------------------------------------------------
        # EMERGENCY CHECK
        # --------------------------------------------------

        emergency_matches = []

        for keyword in self.emergency_keywords:

            if keyword in text:
                emergency_matches.append(keyword)


        if emergency_matches:

            return {
                "level": "Emergency",
                "reason": (
                    "The question contains symptoms or "
                    "situations that may require urgent "
                    "medical attention."
                ),
                "matched_terms": emergency_matches
            }


        # --------------------------------------------------
        # WARNING CHECK
        # --------------------------------------------------

        warning_matches = []

        for keyword in self.warning_keywords:

            if keyword in text:
                warning_matches.append(keyword)


        if warning_matches:

            return {
                "level": "Warning",
                "reason": (
                    "The question contains symptoms that "
                    "may require medical evaluation."
                ),
                "matched_terms": warning_matches
            }


        # --------------------------------------------------
        # NORMAL
        # --------------------------------------------------

        return {
            "level": "Normal",
            "reason": (
                "No immediate emergency indicators "
                "were detected."
            )
        }


    # ======================================================
    # SAFETY MESSAGE
    # ======================================================

    def get_safety_message(self, safety_result):

        level = safety_result.get(
            "level",
            "Normal"
        )


        # --------------------------------------------------
        # EMERGENCY
        # --------------------------------------------------

        if level == "Emergency":

            return (
                "This question may describe a potentially "
                "urgent medical situation. If these symptoms "
                "are happening now or are severe, seek "
                "immediate medical attention or contact your "
                "local emergency medical service. "
                "MediExplain AI cannot determine the cause "
                "or severity of an emergency."
            )


        # --------------------------------------------------
        # WARNING
        # --------------------------------------------------

        if level == "Warning":

            return (
                "These symptoms may require medical "
                "evaluation, especially if they are severe, "
                "persistent, or getting worse. Consider "
                "speaking with a qualified healthcare "
                "professional."
            )


        # --------------------------------------------------
        # NORMAL
        # --------------------------------------------------

        return (
            "This information is for educational purposes "
            "and is not a substitute for professional "
            "medical advice."
        )