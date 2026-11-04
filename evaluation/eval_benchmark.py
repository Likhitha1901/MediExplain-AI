"""
MediExplain AI: Multi-Source RAG & Medical Information Evaluation Benchmark
Compares evidence retrieval, reranking quality, multi-source diversity,
and computes Precision@5, Recall@5, and MRR (Mean Reciprocal Rank)
before and after reranking.
"""

import os
import sys
import time
from typing import Dict, List, Set

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import MedicalVectorStore
from rag.reranker import MedicalReranker
from rag.safety import MedicalSafety
from rag.citations import extract_supporting_passage


# Labelled Ground-Truth Benchmark Queries across multiple medical specialties
LABELLED_BENCHMARK = [
    {
        "id": "BM01",
        "category": "respiratory",
        "query": "What are the common symptoms and inhaler treatments for asthma?",
        "expected_safety": "Normal",
        "target_keywords": {"asthma", "inhaler", "wheezing", "salbutamol", "corticosteroids", "bronchodilator"},
        "relevant_sources": {"NHS", "WHO", "MedlinePlus", "Mayo Clinic"}
    },
    {
        "id": "BM02",
        "category": "cardiovascular",
        "query": "How is high blood pressure treated and what are healthy blood pressure levels?",
        "expected_safety": "Normal",
        "target_keywords": {"blood pressure", "hypertension", "140/90", "arteries", "amlodipine", "systolic"},
        "relevant_sources": {"NHS", "WHO", "MedlinePlus", "Mayo Clinic", "Cleveland Clinic"}
    },
    {
        "id": "BM03",
        "category": "endocrine",
        "query": "What is type 2 diabetes, what are its symptoms, and how does metformin work?",
        "expected_safety": "Normal",
        "target_keywords": {"diabetes", "metformin", "insulin", "glucose", "thirst", "liver"},
        "relevant_sources": {"NIDDK", "Mayo Clinic", "Cleveland Clinic", "MedlinePlus"}
    },
    {
        "id": "BM04",
        "category": "renal",
        "query": "What are the signs of chronic kidney disease and how is kidney function checked?",
        "expected_safety": "Normal",
        "target_keywords": {"kidney", "creatinine", "filtration", "egfr", "albumin", "renal"},
        "relevant_sources": {"Mayo Clinic", "NIDDK", "MedlinePlus"}
    },
    {
        "id": "BM05",
        "category": "cardiac",
        "query": "What are the warning signs and medications used for heart failure?",
        "expected_safety": "Normal",
        "target_keywords": {"heart failure", "edema", "breathlessness", "pump", "sacubitril", "beta blocker"},
        "relevant_sources": {"Cleveland Clinic", "MedlinePlus"}
    },
    {
        "id": "BM06",
        "category": "hematology",
        "query": "What is iron deficiency anemia and what are its symptoms?",
        "expected_safety": "Normal",
        "target_keywords": {"anemia", "anaemia", "iron", "tiredness", "fatigue", "hemoglobin", "palpitations"},
        "relevant_sources": {"NHS", "MedlinePlus"}
    },
    {
        "id": "BM07",
        "category": "neurology",
        "query": "What are the triggers, symptoms, and acute treatments for migraine?",
        "expected_safety": "Normal",
        "target_keywords": {"migraine", "headache", "triptans", "throbbing", "aura", "light"},
        "relevant_sources": {"Mayo Clinic", "MedlinePlus"}
    },
    {
        "id": "BM08",
        "category": "emergency",
        "query": "I have sudden crushing chest pain and shortness of breath",
        "expected_safety": "Emergency",
        "target_keywords": {"chest pain", "heart", "breathing"},
        "relevant_sources": {"MedlinePlus", "NHS"}
    }
]


def is_document_relevant(doc: Dict, target_keywords: Set[str]) -> bool:
    """Evaluates whether retrieved document is clinically relevant to query."""
    text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
    matches = sum(1 for kw in target_keywords if kw in text)
    return matches >= 2


def calculate_retrieval_metrics(retrieved_docs: List[Dict], target_keywords: Set[str], k: int = 5) -> Dict:
    """
    Computes Precision@K, Recall@K, and Reciprocal Rank (RR).
    """
    top_k_docs = retrieved_docs[:k]
    if not top_k_docs:
        return {"precision": 0.0, "recall": 0.0, "reciprocal_rank": 0.0, "relevant_count": 0}

    relevant_flags = [is_document_relevant(d, target_keywords) for d in top_k_docs]
    relevant_in_top_k = sum(1 for f in relevant_flags if f)

    precision = relevant_in_top_k / k

    # Find rank of first relevant doc for Reciprocal Rank (MRR)
    reciprocal_rank = 0.0
    for rank, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            reciprocal_rank = 1.0 / rank
            break

    # Proxy recall based on query keyword coverage
    total_expected_relevant = max(len(target_keywords) // 2, 1)
    recall = min(relevant_in_top_k / total_expected_relevant, 1.0)

    return {
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "relevant_count": relevant_in_top_k
    }


def run_evaluation() -> Dict:
    print("=" * 70)
    print("   MEDIEXPLAIN AI — COMPREHENSIVE MULTI-SOURCE RETRIEVAL BENCHMARK")
    print("=" * 70)

    vector_store = MedicalVectorStore()
    reranker = MedicalReranker()
    safety = MedicalSafety()

    baseline_precisions = []
    baseline_recalls = []
    baseline_mrrs = []

    reranked_precisions = []
    reranked_recalls = []
    reranked_mrrs = []

    sources_encountered = set()
    diabetes_pdf_leak_count = 0
    safety_matches = 0

    print(f"\nRunning benchmark across {len(LABELLED_BENCHMARK)} medical queries...\n")

    for item in LABELLED_BENCHMARK:
        qid = item["id"]
        query = item["query"]
        expected_safety = item["expected_safety"]
        target_keywords = item["target_keywords"]

        # 1. Safety Check
        safety_res = safety.analyze(query)
        is_safe_ok = (safety_res["level"] == expected_safety)
        if is_safe_ok:
            safety_matches += 1

        # 2. Raw ChromaDB Retrieval (Top-8)
        t0 = time.perf_counter()
        raw_res = vector_store.search(query=query, top_k=8)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        raw_docs = []
        if raw_res and raw_res.get("documents"):
            retrieved_docs = raw_res["documents"][0]
            retrieved_metas = raw_res.get("metadatas", [[]])[0]
            for idx, content in enumerate(retrieved_docs):
                meta = retrieved_metas[idx] if idx < len(retrieved_metas) else {}
                source_name = meta.get("source", "MedlinePlus")
                fn = meta.get("file_name", "")
                if fn == "diabetes.pdf" or source_name == "diabetes":
                    diabetes_pdf_leak_count += 1

                sources_encountered.add(source_name)
                raw_docs.append({
                    "content": content,
                    "title": meta.get("title", "Medical Reference"),
                    "source": source_name,
                    "category": meta.get("category", "medical_information"),
                    "url": meta.get("url", "")
                })

        # Baseline Metrics (Before Reranking)
        base_metrics = calculate_retrieval_metrics(raw_docs, target_keywords, k=5)
        baseline_precisions.append(base_metrics["precision"])
        baseline_recalls.append(base_metrics["recall"])
        baseline_mrrs.append(base_metrics["reciprocal_rank"])

        # 3. Reranked Retrieval (Top-5)
        t0 = time.perf_counter()
        reranked_results = reranker.rerank(query=query, documents=raw_docs, top_k=5)
        rerank_ms = (time.perf_counter() - t0) * 1000

        reranked_docs = [r["document"] for r in reranked_results]
        rerank_metrics = calculate_retrieval_metrics(reranked_docs, target_keywords, k=5)
        reranked_precisions.append(rerank_metrics["precision"])
        reranked_recalls.append(rerank_metrics["recall"])
        reranked_mrrs.append(rerank_metrics["reciprocal_rank"])

        top_title = reranked_docs[0]["title"] if reranked_docs else "None"
        top_src = reranked_docs[0]["source"] if reranked_docs else "None"
        top_score = reranked_results[0]["score"] if reranked_results else 0.0

        print(f"[{qid}] Query: '{query[:50]}...'")
        print(f"     Safety: {safety_res['level']} [{'PASS' if is_safe_ok else 'FAIL'}]")
        print(f"     Baseline P@5: {base_metrics['precision']:.2f} | Reranked P@5: {rerank_metrics['precision']:.2f} | Reranked MRR: {rerank_metrics['reciprocal_rank']:.2f}")
        print(f"     Top Document: '{top_title}' ({top_src}) — Relevance Score: {top_score:.4f}")

    # Summary Statistics
    avg_base_p5 = sum(baseline_precisions) / len(baseline_precisions)
    avg_base_r5 = sum(baseline_recalls) / len(baseline_recalls)
    avg_base_mrr = sum(baseline_mrrs) / len(baseline_mrrs)

    avg_rerank_p5 = sum(reranked_precisions) / len(reranked_precisions)
    avg_rerank_r5 = sum(reranked_recalls) / len(reranked_recalls)
    avg_rerank_mrr = sum(reranked_mrrs) / len(reranked_mrrs)

    safety_acc = (safety_matches / len(LABELLED_BENCHMARK)) * 100

    print("\n" + "=" * 70)
    print("   BENCHMARK EVALUATION RESULTS: BEFORE vs AFTER RERANKING")
    print("=" * 70)
    print(f"Metric                  Baseline (No Rerank)    With MedicalReranker    Improvement")
    print(f"Precision@5             {avg_base_p5:.4f}                  {avg_rerank_p5:.4f}                  +{((avg_rerank_p5 - avg_base_p5)*100):.1f}%")
    print(f"Recall@5                {avg_base_r5:.4f}                  {avg_rerank_r5:.4f}                  +{((avg_rerank_r5 - avg_base_r5)*100):.1f}%")
    print(f"Mean Recip Rank (MRR)   {avg_base_mrr:.4f}                  {avg_rerank_mrr:.4f}                  +{((avg_rerank_mrr - avg_base_mrr)*100):.1f}%")
    print("-" * 70)
    print(f"Safety Classification Accuracy : {safety_acc:.1f}%")
    print(f"Sources Successfully Retrieved : {', '.join(sorted(sources_encountered))}")
    print(f"Diabetes PDF Leaks into Search : {diabetes_pdf_leak_count} (Must be 0)")
    print("=" * 70)

    return {
        "precision_at_5_baseline": round(avg_base_p5, 4),
        "precision_at_5_reranked": round(avg_rerank_p5, 4),
        "recall_at_5_baseline": round(avg_base_r5, 4),
        "recall_at_5_reranked": round(avg_rerank_r5, 4),
        "mrr_baseline": round(avg_base_mrr, 4),
        "mrr_reranked": round(avg_rerank_mrr, 4),
        "sources_retrieved": list(sources_encountered),
        "diabetes_pdf_leaks": diabetes_pdf_leak_count,
        "safety_accuracy": safety_acc
    }


if __name__ == "__main__":
    run_evaluation()
