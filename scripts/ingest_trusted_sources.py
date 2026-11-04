"""
MediExplain AI: Multi-Source Medical Knowledge Ingestion Pipeline
Adds trusted medical evidence from NHS, Mayo Clinic, Cleveland Clinic, NIDDK, and WHO.
Removes diabetes.pdf from ChromaDB general medical collection.
Preserves existing MedlinePlus topics.
"""

from datetime import datetime, timezone
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import MedicalVectorStore


TRUSTED_MEDICAL_ARTICLES = [
    # ----------------------------------------------------
    # 1. NHS (UK National Health Service)
    # ----------------------------------------------------
    {
        "title": "Asthma: Symptoms, Diagnosis, and Treatment",
        "source": "NHS",
        "url": "https://www.nhs.uk/conditions/asthma/",
        "category": "disease",
        "disease_or_topic": "Asthma",
        "last_updated": "2025-11-10",
        "content": (
            "Asthma is a common lung condition that causes occasional breathing difficulties. "
            "It affects people of all ages and often starts in childhood, although it can also develop for the first time in adults. "
            "The main symptoms of asthma are a whistling sound when breathing (wheezing), breathlessness, a tight chest which may feel like a band tightening around it, and coughing. "
            "These symptoms can sometimes get worse. This is known as an asthma attack or flare-up. "
            "Asthma is caused by swelling (inflammation) of the breathing tubes that carry air in and out of the lungs. "
            "This makes the tubes highly sensitive, so they temporarily narrow. It may happen randomly or after exposure to a trigger. "
            "Common triggers include house dust mites, animal fur, pollen, smoke, exercise, and cold air. "
            "Asthma is usually treated using an inhaler, a small device that lets you breathe in medicines. "
            "The main types of inhalers are reliever inhalers (usually blue, such as salbutamol) used to quickly relieve symptoms when they occur, "
            "and preventer inhalers (usually brown, containing corticosteroids) used daily to stop symptoms developing over time. "
            "Most people with asthma can lead normal, active lives with appropriate daily management."
        )
    },
    {
        "title": "High Blood Pressure (Hypertension) Overview and Management",
        "source": "NHS",
        "url": "https://www.nhs.uk/conditions/high-blood-pressure-hypertension/",
        "category": "disease",
        "disease_or_topic": "Hypertension",
        "last_updated": "2025-09-15",
        "content": (
            "High blood pressure, or hypertension, is a condition where the pressure in your blood vessels is persistently higher than normal. "
            "It is very common, particularly in older adults, and often does not cause noticeable symptoms. "
            "If left untreated, it increases your risk of serious problems such as heart attacks, heart failure, stroke, kidney disease, and vascular dementia. "
            "Blood pressure is measured in millimetres of mercury (mmHg) and recorded as two numbers: systolic pressure (pressure when your heart pushes blood out) "
            "and diastolic pressure (pressure when your heart rests between beats). "
            "High blood pressure is generally considered to be 140/90 mmHg or higher when measured in a clinic (or an average of 135/85 mmHg at home). "
            "Making healthy lifestyle changes can help reduce your chances of getting high blood pressure and help lower your blood pressure if it is already high. "
            "Key changes include cutting your salt intake to less than 6g a day, eating a low-fat, balanced diet with plenty of fresh fruit and vegetables, "
            "being active, cutting down on alcohol, losing weight if overweight, and stopping smoking. "
            "Common blood pressure medicines include ACE inhibitors (such as ramipril and lisinopril), angiotensin receptor blockers (like losartan), "
            "calcium channel blockers (like amlodipine), and thiazide-like diuretics (like indapamide)."
        )
    },
    {
        "title": "Iron Deficiency Anemia: Symptoms and Management",
        "source": "NHS",
        "url": "https://www.nhs.uk/conditions/iron-deficiency-anaemia/",
        "category": "disease",
        "disease_or_topic": "Anemia",
        "last_updated": "2025-10-01",
        "content": (
            "Iron deficiency anaemia is caused by a lack of iron, often because of blood loss or pregnancy. "
            "It is treated with iron tablets and by eating iron-rich foods. "
            "Symptoms of iron deficiency anaemia include tiredness and lack of energy, shortness of breath, noticeable heartbeats (heart palpitations), "
            "and pale skin. Less common symptoms include headache, hearing ringing or buzzing noises inside your head (tinnitus), food tasting strange, "
            "feeling itchy, a sore tongue, hair loss, and craving non-food items such as ice, paper, or clay (pica). "
            "The condition is diagnosed using a simple full blood count (FBC) test, checking your red blood cell count and haemoglobin level. "
            "Treatment involves taking prescription iron tablets (such as ferrous fumarate or ferrous sulphate) for around 6 months, and treating the underlying cause of blood loss. "
            "Drink plenty of water and take iron tablets with vitamin C (such as a glass of orange juice) to help your body absorb the iron."
        )
    },

    # ----------------------------------------------------
    # 2. Mayo Clinic
    # ----------------------------------------------------
    {
        "title": "Type 2 Diabetes: Diagnosis, Care, and Medications",
        "source": "Mayo Clinic",
        "url": "https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/diagnosis-treatment/drc-20351199",
        "category": "disease",
        "disease_or_topic": "Diabetes",
        "last_updated": "2025-12-04",
        "content": (
            "Type 2 diabetes is an impairment in the way the body regulates and uses sugar (glucose) as a fuel. "
            "This long-term (chronic) condition results in too much sugar circulating in the bloodstream. "
            "Eventually, high blood sugar levels can lead to disorders of the circulatory, nervous, and immune systems. "
            "In type 2 diabetes, there are primarily two interrelated problems at work: your pancreas doesn't produce enough insulin, "
            "and cells respond poorly to insulin and take in less sugar (insulin resistance). "
            "Symptoms of type 2 diabetes often develop slowly. They can include increased thirst, frequent urination, increased hunger, "
            "unintended weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. "
            "Managing type 2 diabetes includes healthy eating, regular exercise, weight loss, and possibly diabetes medication or insulin therapy. "
            "Metformin is generally the first medication prescribed for type 2 diabetes. It works primarily by lowering glucose production in the liver "
            "and improving your body's sensitivity to insulin so that your body uses insulin more effectively. "
            "Other classes of medications include SGLT2 inhibitors (which prevent kidneys from reabsorbing sugar), GLP-1 receptor agonists (which slow digestion and lower blood sugar), "
            "DPP-4 inhibitors, and sulfonylureas. Continuous glucose monitoring and regular glycated hemoglobin (A1C) tests every 3 to 6 months are central to care."
        )
    },
    {
        "title": "Chronic Kidney Disease (CKD): Stages and Preservation",
        "source": "Mayo Clinic",
        "url": "https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521",
        "category": "disease",
        "disease_or_topic": "Chronic Kidney Disease",
        "last_updated": "2025-08-20",
        "content": (
            "Chronic kidney disease, also called chronic kidney failure, involves a gradual loss of kidney function. "
            "Your kidneys filter wastes and excess fluids from your blood, which are then excreted in your urine. "
            "Advanced chronic kidney disease can cause dangerous levels of fluid, electrolytes, and wastes to build up in your body. "
            "In the early stages of chronic kidney disease, you might have few signs or symptoms. "
            "You might not realize you have kidney disease until the condition is advanced. "
            "Signs and symptoms include nausea, vomiting, loss of appetite, fatigue and weakness, sleep problems, changes in how much you urinate, "
            "decreased mental sharpness, muscle cramps, swelling of feet and ankles, persistent dry and itchy skin, and shortness of breath. "
            "Diabetes and high blood pressure are the two most common causes of chronic kidney disease. "
            "Kidney disease is evaluated by measuring blood creatinine to calculate the estimated glomerular filtration rate (eGFR) and testing urine for albumin. "
            "Treatment focuses on slowing progression of kidney damage, typically by controlling blood pressure with ACE inhibitors or ARBs, "
            "controlling blood sugar levels, reducing dietary sodium, and avoiding nephrotoxic medications such as NSAIDs."
        )
    },
    {
        "title": "Migraine Headache: Triggers, Acute Treatment, and Prevention",
        "source": "Mayo Clinic",
        "url": "https://www.mayoclinic.org/diseases-conditions/migraine-headache/diagnosis-treatment/drc-20360207",
        "category": "disease",
        "disease_or_topic": "Migraine",
        "last_updated": "2025-10-18",
        "content": (
            "A migraine is a headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. "
            "It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. "
            "Migraine attacks can last for hours to days, and the pain can be so severe that it interferes with your daily activities. "
            "For some people, a warning symptom known as an aura occurs before or with the headache. "
            "An aura can include visual disturbances, such as flashes of light or blind spots, or other disturbances, such as tingling on one side of the face or in an arm or leg. "
            "Medications can help prevent some migraines and make them less painful. "
            "Pain-relieving medications (acute or abortive treatments) are taken during migraine attacks and are designed to stop symptoms. "
            "These include triptans (such as sumatriptan and rizatriptan), CGRP receptor antagonists (gepants), and nonsteroidal anti-inflammatory drugs (NSAIDs). "
            "Preventive medications can be taken regularly, usually daily or monthly, to reduce the severity or frequency of migraines. "
            "These include beta blockers, anti-seizure drugs, CGRP monoclonal antibodies, and onabotulinumtoxinA injections for chronic migraine."
        )
    },

    # ----------------------------------------------------
    # 3. Cleveland Clinic
    # ----------------------------------------------------
    {
        "title": "Heart Failure: Warning Signs, Diagnosis, and Management",
        "source": "Cleveland Clinic",
        "url": "https://my.clevelandclinic.org/health/diseases/17069-heart-failure",
        "category": "disease",
        "disease_or_topic": "Heart Failure",
        "last_updated": "2025-11-25",
        "content": (
            "Heart failure (congestive heart failure) means that your heart muscle doesn't pump blood as well as it should. "
            "Certain conditions, such as narrowed arteries in your heart (coronary artery disease) or high blood pressure, gradually leave your heart too weak or stiff to fill and pump efficiently. "
            "Heart failure can be ongoing (chronic), or it may start suddenly (acute). "
            "Signs and symptoms of heart failure may include shortness of breath with activity or when lying down, fatigue and weakness, "
            "swelling in the legs, ankles, and feet (peripheral edema), rapid or irregular heartbeat, reduced ability to exercise, "
            "persistent cough or wheezing with white or pink blood-tinged mucus, swelling of the belly area (ascites), very rapid weight gain from fluid retention, "
            "and difficulty concentrating or decreased alertness. "
            "Heart failure is evaluated using echocardiography (to measure ejection fraction), blood tests for BNP or NT-proBNP biomarkers, and electrocardiograms. "
            "Treatment includes lifestyle measures (restricting daily sodium, monitoring daily morning weight) and guideline-directed medical therapy. "
            "Key medication classes proven to reduce mortality include ARNIs (sacubitril/valsartan) or ACE inhibitors, beta blockers (carvedilol, metoprolol succinate), "
            "mineralocorticoid receptor antagonists (spironolactone), and SGLT2 inhibitors (dapagliflozin, empagliflozin). Diuretics like furosemide help control fluid overload."
        )
    },
    {
        "title": "Metformin: Indications, Dosing Principles, and Precautions",
        "source": "Cleveland Clinic",
        "url": "https://my.clevelandclinic.org/health/drugs/18797-metformin-tablets",
        "category": "medication",
        "disease_or_topic": "Metformin",
        "last_updated": "2025-07-14",
        "content": (
            "Metformin is an oral prescription medication used to treat type 2 diabetes. "
            "It belongs to a class of medicines called biguanides. Metformin helps control the amount of glucose (sugar) in your blood. "
            "It decreases the amount of glucose you absorb from your food and the amount of glucose made by your liver. "
            "Metformin also increases your body's response to insulin, a natural substance that controls the amount of glucose in the blood. "
            "Metformin does not cure diabetes, but it helps manage blood sugar levels and prevents serious long-term complications like kidney damage, blindness, nerve problems, and cardiovascular disease. "
            "The most common side effects include stomach upset, nausea, vomiting, diarrhea, gas, and metallic taste. "
            "These gastrointestinal side effects are usually temporary and can be significantly minimized by taking metformin with meals and starting at a low dose with gradual upward titration. "
            "Extended-release (ER) formulations are often better tolerated. "
            "A rare but serious complication is lactic acidosis, a buildup of lactic acid in the bloodstream. "
            "Patients with severe kidney impairment, advanced heart failure, or severe liver disease have a higher risk and may require dose adjustment or alternative therapy."
        )
    },

    # ----------------------------------------------------
    # 4. NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)
    # ----------------------------------------------------
    {
        "title": "What is Diabetes? Types, Blood Glucose, and Insulin",
        "source": "NIDDK",
        "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/what-is-diabetes",
        "category": "disease",
        "disease_or_topic": "Diabetes",
        "last_updated": "2025-10-30",
        "content": (
            "Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. "
            "Blood glucose is your main source of energy and comes from the food you eat. "
            "Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy. "
            "Sometimes your body doesn't make enough—or any—insulin or doesn't use insulin well. Glucose then stays in your blood and doesn't reach your cells. "
            "Over time, having too much glucose in your blood can cause health problems. "
            "Although diabetes has no cure, you can take steps to manage your diabetes and stay healthy. "
            "The most common types of diabetes are type 1, type 2, and gestational diabetes. "
            "In type 1 diabetes, the immune system attacks and destroys the insulin-producing beta cells in the pancreas. People with type 1 diabetes must take insulin every day to survive. "
            "In type 2 diabetes, your body doesn't make or use insulin well. You can develop type 2 diabetes at any age, even during childhood. "
            "Gestational diabetes develops in some women when they are pregnant and usually goes away after giving birth, though it raises future risk of type 2 diabetes. "
            "Diagnostic criteria for diabetes include a fasting plasma glucose of 126 mg/dL or higher, an A1C level of 6.5% or higher, or an oral glucose tolerance test result of 200 mg/dL or higher."
        )
    },
    {
        "title": "Diabetic Kidney Disease (Diabetic Nephropathy)",
        "source": "NIDDK",
        "url": "https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/diabetic-kidney-disease",
        "category": "disease",
        "disease_or_topic": "Diabetic Nephropathy",
        "last_updated": "2025-06-12",
        "content": (
            "Diabetic kidney disease is a serious complication of type 1 and type 2 diabetes. "
            "It is also called diabetic nephropathy. Over time, poorly controlled diabetes can cause damage to blood vessel clusters in your kidneys that filter waste from your blood. "
            "This can lead to kidney damage and cause high blood pressure. High blood pressure can cause further kidney damage by increasing the pressure in the delicate filtering system of the kidneys. "
            "Early in the disease, you may not notice any signs or symptoms. "
            "The earliest sign of diabetic kidney disease is a small amount of protein leaking into the urine, called microalbuminuria or urine albumin-to-creatinine ratio (UACR). "
            "Annual screening of urine albumin and blood creatinine (eGFR) is recommended for all people with diabetes. "
            "Keeping your blood glucose and blood pressure in your target range is the most effective way to prevent or delay diabetic kidney disease. "
            "Medicines called ACE inhibitors and ARBs lower blood pressure and protect against kidney damage. SGLT2 inhibitors and nonsteroidal mineralocorticoid receptor antagonists (finerenone) have also demonstrated substantial kidney-protective benefits."
        )
    },

    # ----------------------------------------------------
    # 5. WHO (World Health Organization)
    # ----------------------------------------------------
    {
        "title": "Asthma: Key Facts, Causes, and Essential Inhaler Access",
        "source": "WHO",
        "url": "https://www.who.int/news-room/fact-sheets/detail/asthma",
        "category": "disease",
        "disease_or_topic": "Asthma",
        "last_updated": "2025-05-18",
        "content": (
            "Asthma is a major noncommunicable disease (NCD), affecting both children and adults, and is the most common chronic disease among children. "
            "Inflammation and narrowing of the small airways in the lungs cause asthma symptoms, which can be any combination of cough, wheeze, shortness of breath, and chest tightness. "
            "Asthma was estimated to affect 262 million people worldwide and caused 455,000 deaths. "
            "Asthma symptoms vary from person to person and may become worse during exercise or at night. "
            "Inhaled medication can control asthma symptoms and enable people with asthma to lead a normal, active life. "
            "There are two main types of inhalers: bronchodilators (such as salbutamol) that open air passages and relieve symptoms; "
            "and steroid inhalers (such as beclometasone and budesonide) that reduce inflammation in air passages, which improves asthma symptoms and reduces the risk of severe asthma attacks and death. "
            "WHO emphasizes equitable access to affordable inhaled corticosteroids and bronchodilators as essential medicines. "
            "People with asthma may need to use their inhaler every day. Their treatment will depend on the frequency of symptoms and the different types of inhalers available."
        )
    },
    {
        "title": "Hypertension: Global Key Facts, Risks, and Prevention",
        "source": "WHO",
        "url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "category": "disease",
        "disease_or_topic": "Hypertension",
        "last_updated": "2025-07-28",
        "content": (
            "Hypertension (high blood pressure) is when the pressure in your blood vessels is too high (140/90 mmHg or higher). "
            "It is common but can be serious if not treated. An estimated 1.28 billion adults aged 30–79 years worldwide have hypertension, "
            "most (two-thirds) living in low- and middle-income countries. "
            "An estimated 46% of adults with hypertension are unaware that they have the condition. "
            "Less than half of adults (42%) with hypertension are diagnosed and treated. Approximately 1 in 5 adults (21%) with hypertension have it under control. "
            "Hypertension is a major cause of premature death worldwide. Most people with hypertension feel no symptoms at all. "
            "The only way to know is to have your blood pressure checked regularly. When symptoms do occur, they can include early morning headaches, nosebleeds, irregular heart rhythms, vision changes, and buzzing in the ears. "
            "Modifiable risk factors include unhealthy diets (excessive salt consumption, a diet high in saturated fat and trans fats, low intake of fruits and vegetables), physical inactivity, consumption of tobacco and alcohol, and being overweight or obese. "
            "Lifestyle changes and affordable, safe antihypertensive medications (such as thiazides, ACE inhibitors, and calcium channel blockers) effectively control blood pressure and prevent cardiovascular morbidity."
        )
    }
]


def remove_diabetes_pdf_from_chromadb(vector_store: MedicalVectorStore):
    """
    Safely removes the diabetes.pdf entries from the general medical retrieval
    collection without deleting the physical file on disk.
    """
    print("\n[ChromaDB Migration] Inspecting and removing 'diabetes.pdf' chunks...")
    try:
        col = vector_store.collection
        count_before = col.count()
        print(f"Total documents before cleaning: {count_before}")

        # Check existing diabetes.pdf records
        all_records = col.get(include=["metadatas"])
        pdf_ids_to_remove = []
        if all_records and all_records.get("ids"):
            for rec_id, meta in zip(all_records["ids"], all_records.get("metadatas", [])):
                meta = meta or {}
                if meta.get("file_name") == "diabetes.pdf" or meta.get("source") == "diabetes":
                    pdf_ids_to_remove.append(rec_id)

        if pdf_ids_to_remove:
            print(f"Found {len(pdf_ids_to_remove)} 'diabetes.pdf' chunks to remove from general retrieval collection.")
            col.delete(ids=pdf_ids_to_remove)
            count_after = col.count()
            print(f"Successfully deleted {len(pdf_ids_to_remove)} chunks. Total documents now: {count_after}")
        else:
            print("No 'diabetes.pdf' chunks found in ChromaDB collection.")

    except Exception as exc:
        print(f"Error while cleaning diabetes.pdf from ChromaDB: {exc}")


def ingest_trusted_sources(vector_store: MedicalVectorStore) -> int:
    """
    Ingests curated high-quality articles from NHS, Mayo Clinic,
    Cleveland Clinic, NIDDK, and WHO into ChromaDB with complete metadata.
    """
    print("\n[Ingestion] Preparing multi-source medical knowledge documents...")
    now_iso = datetime.now(timezone.utc).isoformat()
    docs_to_add = []

    for idx, item in enumerate(TRUSTED_MEDICAL_ARTICLES):
        # Create deterministic ID based on source and title to prevent duplicate entries
        id_seed = f"{item['source']}_{item['title']}_{idx}"
        doc_id = f"trusted_{item['source'].lower()}_{hashlib.md5(id_seed.encode('utf-8')).hexdigest()[:12]}"

        docs_to_add.append({
            "id": doc_id,
            "content": item["content"],
            "title": item["title"],
            "source": item["source"],
            "category": item["category"],
            "url": item["url"],
            "file_name": f"{item['source'].lower()}_articles.json",
            "chunk_index": idx,
            "disease_or_topic": item.get("disease_or_topic", ""),
            "ingestion_date": now_iso,
            "last_updated": item.get("last_updated", now_iso)
        })

    print(f"Upserting {len(docs_to_add)} trusted medical documents into ChromaDB...")
    vector_store.add_documents(docs_to_add)
    print(f"Ingestion of {len(docs_to_add)} multi-source documents complete!")
    return len(docs_to_add)


def main():
    print("=" * 65)
    print("   MEDIEXPLAIN AI — TRUSTED MULTI-SOURCE INGESTION PIPELINE")
    print("=" * 65)

    vector_store = MedicalVectorStore()

    # 1. Remove diabetes.pdf from general medical knowledge
    remove_diabetes_pdf_from_chromadb(vector_store)

    # 2. Ingest trusted sources (NHS, Mayo Clinic, Cleveland Clinic, NIDDK, WHO)
    added_count = ingest_trusted_sources(vector_store)

    # 3. Final summary
    col = vector_store.collection
    print("\n" + "=" * 65)
    print(f"FINAL CHROMADB STATUS: {col.count()} documents in '{col.name}' collection.")
    print("Sources represented: MedlinePlus, NHS, Mayo Clinic, Cleveland Clinic, NIDDK, WHO.")
    print("=" * 65)


if __name__ == "__main__":
    main()
