import re
from typing import Dict, List, Optional, Any
from rag.vector_store import MedicalVectorStore
from rag.reranker import MedicalReranker


# Medical treatments knowledge base across diverse diseases for grounded retrieval synthesis
DISEASE_TREATMENT_PROFILES = {
    "asthma": [
        {
            "treatment": "Inhaled Short-Acting Beta2-Agonists (SABA, e.g. Albuterol / Salbutamol)",
            "how_it_works": "Selectively stimulates beta-2 adrenergic receptors in bronchial smooth muscles, causing rapid relaxation and airway dilation within 3-5 minutes.",
            "main_purpose": "Immediate rescue relief of acute bronchospasm, wheezing, coughing, and shortness of breath.",
            "benefits": "Fast-acting rescue relief during asthma attacks or 15 minutes before vigorous physical exertion.",
            "side_effects": "Temporary rapid heart rate (tachycardia), finger tremors, nervousness, headache.",
            "risks_precautions": "Overreliance indicates uncontrolled underlying inflammation; consult doctor if using more than twice a week.",
            "suitability": "Suitable for most patients needing immediate rescue relief; use cautiously in severe cardiovascular arrhythmias.",
            "administration": "Inhalation via Metered-Dose Inhaler (MDI) with spacer or Dry Powder Inhaler (DPI).",
            "evidence_sources": ["MedlinePlus: Asthma Overview", "NHLBI Asthma Management Guidelines"]
        },
        {
            "treatment": "Inhaled Corticosteroids (ICS, e.g. Fluticasone, Budesonide)",
            "how_it_works": "Suppresses inflammatory cytokine production, reduces bronchial swelling, and prevents mucus hypersecretion in airways over time.",
            "main_purpose": "First-line long-term maintenance and prevention of asthma flare-ups and hospitalizations.",
            "benefits": "Significantly decreases airway hyperresponsiveness, preserves lung function, and reduces reliance on rescue inhalers.",
            "side_effects": "Oral thrush (candidiasis), dysphonia (hoarseness), mild throat irritation.",
            "risks_precautions": "Rinse mouth thoroughly with water and spit after each inhalation to prevent fungal infection.",
            "suitability": "Cornerstone maintenance therapy for persistent asthma in adults and children.",
            "administration": "Daily scheduled inhalation morning and evening as prescribed.",
            "evidence_sources": ["MedlinePlus: Inhaled Steroids", "WHO Asthma Fact Sheet"]
        },
        {
            "treatment": "Leukotriene Receptor Antagonists (LTRA, e.g. Montelukast)",
            "how_it_works": "Blocks cysteinyl leukotriene receptors on inflammatory cells, attenuating smooth muscle contraction and eosinophilic inflammation.",
            "main_purpose": "Oral controller medication for maintenance control and allergic asthma / exercise-induced bronchoconstriction.",
            "benefits": "Convenient oral tablet; particularly effective when asthma coexists with allergic rhinitis.",
            "side_effects": "Headache, mild stomach discomfort, upper respiratory infection symptoms.",
            "risks_precautions": "Monitor for neuropsychiatric changes (unusual dreams, mood alterations); notify doctor if present.",
            "suitability": "Useful as add-on therapy or alternative for patients who have difficulty with inhaler technique.",
            "administration": "Once daily oral tablet taken in the evening.",
            "evidence_sources": ["MedlinePlus Drug Information: Montelukast", "Mayo Clinic: Asthma Treatment"]
        }
    ],
    "diabetes": [
        {
            "treatment": "Metformin (Biguanide Oral Antidiabetic)",
            "how_it_works": "Decreases hepatic gluconeogenesis (glucose release from liver), reduces intestinal glucose absorption, and enhances peripheral insulin sensitivity.",
            "main_purpose": "First-line pharmacological management for Type 2 Diabetes mellitus alongside diet and physical activity.",
            "benefits": "Effective glycemic control (reduces HbA1c ~1-1.5%), low risk of hypoglycemia as monotherapy, weight neutral or slight weight loss.",
            "side_effects": "Gastrointestinal upset (nausea, diarrhea, abdominal cramping, metallic taste).",
            "risks_precautions": "Contraindicated in severe renal impairment (eGFR < 30 mL/min); rare risk of lactic acidosis. Monitor kidney function regularly.",
            "suitability": "First choice for newly diagnosed Type 2 diabetes without advanced kidney or severe liver disease.",
            "administration": "Oral tablet taken with or immediately after meals.",
            "evidence_sources": ["MedlinePlus Drug Information: Metformin", "NIDDK Diabetes Guidelines"]
        },
        {
            "treatment": "SGLT2 Inhibitors (e.g. Empagliflozin, Dapagliflozin)",
            "how_it_works": "Inhibits sodium-glucose co-transporter-2 in renal proximal tubules, promoting urinary excretion of excess blood glucose.",
            "main_purpose": "Blood sugar reduction with proven cardiovascular and renal protective benefits.",
            "benefits": "Significant reductions in heart failure hospitalizations, kidney disease progression, and modest blood pressure/weight reduction.",
            "side_effects": "Increased urination, genital fungal infections, urinary tract infections, volume depletion/dehydration.",
            "risks_precautions": "Maintain adequate hydration; rare risk of euglycemic diabetic ketoacidosis (DKA) during severe illness.",
            "suitability": "Highly recommended for patients with Type 2 diabetes and comorbid heart failure or chronic kidney disease.",
            "administration": "Once daily oral tablet taken in the morning.",
            "evidence_sources": ["MedlinePlus: Empagliflozin", "Cleveland Clinic: SGLT2 Inhibitors"]
        },
        {
            "treatment": "Insulin Therapy (Basal & Bolus Analogues)",
            "how_it_works": "Directly substitutes endogenous hormone insulin to facilitate cellular uptake of glucose from blood into skeletal muscle and adipose tissue.",
            "main_purpose": "Essential life-saving therapy in Type 1 Diabetes; indicated in Type 2 Diabetes when oral agents fail to achieve glycemic goals.",
            "benefits": "Potent and dose-adjustable; can reduce severely elevated glucose and HbA1c to any target level.",
            "side_effects": "Risk of hypoglycemia (shakiness, confusion, sweating), potential weight gain, injection site lipodystrophy.",
            "risks_precautions": "Requires frequent capillary or continuous glucose monitoring; requires precise dose matching to meals and activity.",
            "suitability": "Mandatory in Type 1 diabetes; indicated in acute illness, pregnancy with diabetes, or severe insulinopaenia.",
            "administration": "Subcutaneous injection via insulin pen, syringe, or continuous subcutaneous insulin infusion pump.",
            "evidence_sources": ["MedlinePlus: Insulin Information", "NIDDK: Insulin and Other Diabetes Medicines"]
        }
    ],
    "hypertension": [
        {
            "treatment": "ACE Inhibitors / ARBs (e.g. Lisinopril, Losartan)",
            "how_it_works": "Blocks conversion of angiotensin I to angiotensin II (or blocks AT1 receptors), preventing vasoconstriction and aldosterone-mediated sodium retention.",
            "main_purpose": "First-line blood pressure lowering and organ protection.",
            "benefits": "Protects renal function (especially in diabetic nephropathy), reduces heart failure mortality and left ventricular hypertrophy.",
            "side_effects": "Dry persistent cough (common with ACE inhibitors), dizziness, elevated serum potassium (hyperkalemia).",
            "risks_precautions": "Contraindicated in pregnancy (teratogenic); regular monitoring of serum creatinine and potassium required.",
            "suitability": "First-line for non-black patients, especially those with diabetes, proteinuria, or heart failure.",
            "administration": "Once daily oral tablet.",
            "evidence_sources": ["MedlinePlus: Blood Pressure Medicines", "American Heart Association Guidelines"]
        },
        {
            "treatment": "Calcium Channel Blockers (Dihydropyridines, e.g. Amlodipine)",
            "how_it_works": "Inhibits calcium ion influx into vascular smooth muscle cells, causing peripheral arterial dilation and systemic vascular resistance reduction.",
            "main_purpose": "Lowering systolic and diastolic blood pressure; treating stable angina.",
            "benefits": "Highly effective across broad demographic groups including elderly and African descent individuals; minimal biochemical monitoring required.",
            "side_effects": "Peripheral edema (ankle swelling), headache, facial flushing, lightheadedness.",
            "risks_precautions": "Edema is dose-dependent; does not respond to diuretics as it is capillary leak, not fluid overload.",
            "suitability": "Excellent first-line choice for isolated systolic hypertension and patients with angina.",
            "administration": "Once daily oral tablet taken with or without food.",
            "evidence_sources": ["MedlinePlus Drug Information: Amlodipine", "Mayo Clinic: High Blood Pressure"]
        },
        {
            "treatment": "Thiazide-like Diuretics (e.g. Chlorthalidone, Hydrochlorothiazide)",
            "how_it_works": "Inhibits sodium and chloride reabsorption in renal distal convoluted tubules, increasing urinary excretion of water and salt.",
            "main_purpose": "Volume and vascular tone reduction in essential hypertension.",
            "benefits": "Extensive clinical trial evidence demonstrating reductions in stroke and cardiovascular events; low cost.",
            "side_effects": "Hypokalemia (low blood potassium), mild hyponatremia, increased uric acid (may trigger gout).",
            "risks_precautions": "Periodic electrolyte panel checks needed; monitor blood pressure for postural hypotension.",
            "suitability": "Effective first-line agent, especially in elderly patients or as combination therapy.",
            "administration": "Once daily oral tablet taken in the morning to prevent nocturnal urination.",
            "evidence_sources": ["MedlinePlus: Diuretics Information", "NHS: High Blood Pressure Medicines"]
        }
    ],
    "migraine": [
        {
            "treatment": "Triptans (Serotonin 5-HT1B/1D Receptor Agonists, e.g. Sumatriptan)",
            "how_it_works": "Causes cranial blood vessel constriction and inhibits release of inflammatory neuropeptides from trigeminal nerve endings.",
            "main_purpose": "Acute abortive treatment for moderate to severe migraine attacks with or without aura.",
            "benefits": "Rapidly relieves throbbing headache, nausea, photophobia, and phonophobia within 1-2 hours when taken early.",
            "side_effects": "Chest tightness or pressure sensation, tingling, flushing, drowsiness, dizziness.",
            "risks_precautions": "Contraindicated in coronary artery disease, stroke, uncontrolled hypertension, or peripheral vascular disease.",
            "suitability": "First-line acute therapy when simple analgesics (NSAIDs) are insufficient.",
            "administration": "Oral tablet, nasal spray, or subcutaneous auto-injector taken at onset of headache phase.",
            "evidence_sources": ["MedlinePlus: Sumatriptan", "American Headache Society Guidelines"]
        },
        {
            "treatment": "CGRP Receptor Antagonists / Monoclonal Antibodies (e.g. Erenumab, Rimegepant)",
            "how_it_works": "Blocks calcitonin gene-related peptide (CGRP) or its receptor, halting neurogenic inflammation and migraine pain transmission.",
            "main_purpose": "Targeted preventive therapy for frequent episodic or chronic migraine.",
            "benefits": "Substantially reduces monthly migraine days with superior tolerability compared to older oral preventives.",
            "side_effects": "Constipation, mild injection site redness, muscle spasms.",
            "risks_precautions": "Specialty medication; high cost; monitor blood pressure with certain agents.",
            "suitability": "Patients with 4+ migraine days per month unresponsive to traditional preventives.",
            "administration": "Monthly subcutaneous injection or oral orally disintegrating tablet (gepant).",
            "evidence_sources": ["MedlinePlus: Erenumab", "Cleveland Clinic: CGRP Inhibitors"]
        },
        {
            "treatment": "Preventive Lifestyle & Trigger Management",
            "how_it_works": "Maintains circadian and metabolic homeostasis, stabilizing trigeminovascular activation threshold.",
            "main_purpose": "Non-pharmacological baseline reduction of attack frequency and severity.",
            "benefits": "Zero medical side effects; improves overall well-being, sleep quality, and stress resilience.",
            "side_effects": "None.",
            "risks_precautions": "Requires disciplined adherence; rarely sufficient alone for severe chronic migraine.",
            "suitability": "Essential foundational management for all migraine patients.",
            "administration": "Consistent sleep schedule, adequate hydration, stress management, regular meals, avoiding known dietary triggers.",
            "evidence_sources": ["MedlinePlus: Migraine Overview", "Mayo Clinic: Migraine Prevention"]
        }
    ],
    "rheumatoid arthritis": [
        {
            "treatment": "Conventional Synthetic DMARDs (e.g. Methotrexate)",
            "how_it_works": "Inhibits dihydrofolate reductase and promotes adenosine release, down-regulating inflammatory immune activity.",
            "main_purpose": "First-line anchor disease-modifying therapy to halt joint damage and induce remission.",
            "benefits": "Slows or halts radiographic joint destruction; established long-term efficacy and safety profile.",
            "side_effects": "Nausea, mouth ulcers (stomatitis), fatigue, mild hair thinning.",
            "risks_precautions": "Requires folic acid supplementation; routine liver enzymes and CBC monitoring; contraindicated in pregnancy.",
            "suitability": "First-line recommendation for active rheumatoid arthritis upon diagnosis.",
            "administration": "Taken once weekly (oral tablet or subcutaneous injection). Never taken daily.",
            "evidence_sources": ["MedlinePlus: Methotrexate", "ACR Rheumatoid Arthritis Guidelines"]
        },
        {
            "treatment": "Biologic DMARDs (TNF Inhibitors, e.g. Adalimumab, Etanercept)",
            "how_it_works": "Neutralizes tumor necrosis factor-alpha (TNF-alpha), a pivotal proinflammatory cytokine in rheumatoid joint inflammation.",
            "main_purpose": "Targeted therapy for moderate-to-severe active disease with inadequate response to conventional DMARDs.",
            "benefits": "Rapid improvement in pain, swelling, and physical function; proven reduction in joint erosion.",
            "side_effects": "Injection site reactions, increased susceptibility to serious bacterial and viral infections.",
            "risks_precautions": "Mandatory screening for latent tuberculosis and hepatitis B before initiation; withhold during active infections.",
            "suitability": "Patients with refractory disease or poor prognostic factors.",
            "administration": "Subcutaneous self-injection every 1-2 weeks.",
            "evidence_sources": ["MedlinePlus: Adalimumab", "WHO Musculoskeletal Health Guidance"]
        }
    ]
}


class TreatmentComparator:
    """
    General-purpose dynamic treatment comparison across medical conditions.
    Combines ChromaDB vector retrieval, MedlinePlus search, and established
    evidence profiles to produce structured comparisons.
    """

    def __init__(self, vector_store: Optional[MedicalVectorStore] = None, reranker: Optional[MedicalReranker] = None):
        self.vector_store = vector_store or MedicalVectorStore()
        self.reranker = reranker or MedicalReranker()

    def identify_condition(self, query: str) -> str:
        """Extracts key condition or disease name from comparison request."""
        clean = query.lower().strip()
        # Remove comparison boilerplate
        clean = re.sub(r"\b(compare|the|available|treatments|treatment|for|options|in|of|versus|vs|what|are)\b", " ", clean)
        clean = re.sub(r"[^\w\s-]", " ", clean).strip()
        words = [w for w in clean.split() if len(w) > 2]
        return " ".join(words) if words else "general condition"

    def compare(self, query_or_condition: str, language: str = "English") -> Dict[str, Any]:
        """
        Dynamically compares evidence-based treatment options for any disease.
        """
        condition_name = self.identify_condition(query_or_condition)
        cond_lower = condition_name.lower()

        # 1. Check known high-evidence profiles first
        matched_profile_key = None
        for key in DISEASE_TREATMENT_PROFILES:
            if key in cond_lower or cond_lower in key:
                matched_profile_key = key
                break

        # 2. Vector search in ChromaDB for retrieved context
        search_query = f"treatment medications therapy for {condition_name}"
        vector_results = self.vector_store.search(query=search_query, top_k=6)

        retrieved_docs = []
        if vector_results and vector_results.get("documents"):
            docs = vector_results["documents"][0]
            metas = vector_results.get("metadatas", [[]])[0]
            for idx, c in enumerate(docs):
                m = metas[idx] if idx < len(metas) else {}
                retrieved_docs.append({
                    "content": c,
                    "title": m.get("title", f"{condition_name.capitalize()} Reference"),
                    "source": m.get("source", "MedlinePlus"),
                    "category": m.get("category", "treatment"),
                    "url": m.get("url", "")
                })

        # 3. If exact profile exists, use it and enrich with retrieved sources
        if matched_profile_key:
            raw_options = DISEASE_TREATMENT_PROFILES[matched_profile_key]
            options = []
            for item in raw_options:
                opt_copy = dict(item)
                # Backward-compatibility aliases for existing clients/tests
                opt_copy["risks"] = opt_copy.get("risks_precautions", "")
                opt_copy["limitations"] = opt_copy.get("risks_precautions", "")
                opt_copy["category"] = opt_copy.get("administration", "")
                opt_copy["evidence_source"] = opt_copy["evidence_sources"][0] if opt_copy.get("evidence_sources") else ""
                options.append(opt_copy)

            if retrieved_docs:
                for opt in options:
                    for rd in retrieved_docs:
                        if rd.get("url") and rd["url"] not in opt["evidence_sources"]:
                            opt["evidence_sources"].append(f"{rd['source']}: {rd['title']}")
                            break

            display_name = matched_profile_key.title()
            overview = (
                f"Evidence-based comparative overview of primary treatment modalities evaluated for {display_name}. "
                "Individual treatment selection depends on symptom severity, patient age, comorbidities, contraindications, and clinical evaluation."
            )
        elif retrieved_docs:
            # Dynamic synthesis from retrieved ChromaDB documents
            display_name = condition_name.title()
            overview = (
                f"Dynamic treatment comparison for {display_name} synthesized from retrieved medical evidence documents. "
                "Therapy depends on individual clinical diagnosis and patient response."
            )
            # Synthesize options from top retrieved documents
            options = []
            for doc in retrieved_docs[:3]:
                title = doc.get("title", f"{display_name} Care")
                content = doc.get("content", "")
                source_name = doc.get("source", "Medical Reference")
                src_url = doc.get("url", "")

                # Extract key benefit or mechanism sentence
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if len(s.strip()) > 30]
                summary_sentence = sentences[0] if sentences else content[:180]
                how_sentence = sentences[1] if len(sentences) > 1 else "Acts on pathophysiological pathways related to this condition."

                options.append({
                    "treatment": f"Evidence-based approach: {title}",
                    "how_it_works": how_sentence,
                    "main_purpose": f"Management and symptom reduction of {display_name}.",
                    "benefits": summary_sentence,
                    "side_effects": "Potential adverse effects vary by agent; review clinical packaging and doctor instructions.",
                    "risks_precautions": "Requires individualized evaluation for contraindications, organ function, and drug interactions.",
                    "suitability": "Dependent on disease staging, patient tolerability, and physician judgment.",
                    "administration": "Administered under medical supervision as directed.",
                    "evidence_sources": [f"{source_name}: {title}" + (f" ({src_url})" if src_url else "")]
                })
        else:
            # Insufficient evidence found
            display_name = condition_name.title()
            overview = (
                f"Insufficient trusted medical documents were found in the knowledge base to generate an accurate, evidence-grounded treatment comparison for '{display_name}'."
            )
            options = []

        disclaimer = (
            "CLINICAL TREATMENT NOTICE: MediExplain AI does not rank treatments as universally superior. "
            "Treatment efficacy, tolerability, and safety vary greatly among individuals. "
            "Never start, stop, or switch prescription medications without consulting your personal physician."
        )

        return {
            "condition": display_name,
            "overview": overview,
            "options": options,
            "evidence_count": len(retrieved_docs),
            "disclaimer": disclaimer
        }
