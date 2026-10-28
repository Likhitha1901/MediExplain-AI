"""
MediExplain AI: RAG Pipeline & Medical Information Evaluation Benchmark
Compares evidence retrieval, reranking quality, safety detection, and citation accuracy.
"""

import os
import sys
import time
from typing import Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vector_store import MedicalVectorStore
from rag.reranker import MedicalReranker
from rag.safety import MedicalSafety
from rag.citations import extract_supporting_passage, map_claims_to_sources


BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "category": "disease",
        "query": "What are the common symptoms of diabetes?",
        "expected_safety": "Normal",
        "key_terms": ["thirst", "urination", "glucose", "insulin", "diabetes"]
    },
    {
        "id": "Q2",
        "category": "medication",
        "query": "What are the side effects and precautions for metformin?",
        "expected_safety": "Normal",
        "key_terms": ["metformin", "nausea", "diarrhea", "lactic", "kidney"]
    },
    {
        "id": "Q3",
        "category": "disease",
        "query": "What is asthma and how is it treated?",
        "expected_safety": "Normal",
        "key_terms": ["asthma", "airway", "inhaler", "wheezing", "bronchodilator"]
    },
    {
        "id": "Q4",
        "category": "warning",
        "query": "I have had a high fever and severe headache for three days",
        "expected_safety": "Warning",
        "key_terms": ["fever", "headache"]
    },
    {
        "id": "Q5",
        "category": "emergency",
        "query": "I have sudden severe chest pain and left arm numbness",
        "expected_safety": "Emergency",
        "key_terms": ["chest pain", "heart"]
    }
]


def run_evaluation() -> Dict:
    print("=" * 65)
    print("   MEDIEXPLAIN AI — RAG & SAFETY BENCHMARK EVALUATION")
    print("=" * 65)

    vector_store = MedicalVectorStore()
    reranker = MedicalReranker()
    safety = MedicalSafety()

    results = []
    total_retrieval_time = 0.0
    total_rerank_time = 0.0
    safety_matches = 0

    for item in BENCHMARK_QUERIES:
        query = item["query"]
        print(f"\n[Test {item['id']}] Query: '{query}'")

        # 1. Safety Evaluation
        t0 = time.perf_counter()
        safety_res = safety.analyze(query)
        safety_time = time.perf_counter() - t0
        is_safety_correct = safety_res["level"] == item["expected_safety"]
        if is_safety_correct:
            safety_matches += 1

        print(f"  Safety: {safety_res['level']} (Expected: {item['expected_safety']}) -> {'PASS' if is_safety_correct else 'FAIL'}")

        # 2. Retrieval Evaluation
        t0 = time.perf_counter()
        raw_results = vector_store.search(query=query, top_k=8)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        total_retrieval_time += retrieval_ms

        raw_docs = []
        if raw_results and raw_results.get("documents"):
            retrieved_docs = raw_results["documents"][0]
            retrieved_metas = raw_results.get("metadatas", [[]])[0]
            for idx, content in enumerate(retrieved_docs):
                meta = retrieved_metas[idx] if idx < len(retrieved_metas) else {}
                raw_docs.append({
                    "content": content,
                    "title": meta.get("title", "Medical Reference"),
                    "source": meta.get("source", "MedlinePlus"),
                    "category": meta.get("category", "medical_information"),
                    "url": meta.get("url", "")
                })

        print(f"  Retrieval Latency: {retrieval_ms:.1f}ms | Docs Retrieved: {len(raw_docs)}")

        # 3. Reranking Evaluation
        t0 = time.perf_counter()
        reranked = reranker.rerank(query=query, documents=raw_docs, top_k=3)
        rerank_ms = (time.perf_counter() - t0) * 1000
        total_rerank_time += rerank_ms

        top_score = reranked[0]["score"] if reranked else 0.0
        top_title = reranked[0]["document"].get("title", "N/A") if reranked else "None"

        # Check evidence citation
        top_content = reranked[0]["document"].get("content", "") if reranked else ""
        excerpt = extract_supporting_passage(query, top_content)

        print(f"  Rerank Latency: {rerank_ms:.1f}ms | Top Doc: '{top_title}' (Score: {top_score:.4f})")
        print(f"  Excerpt: {excerpt[:80]}...")

        results.append({
            "id": item["id"],
            "query": query,
            "safety_result": safety_res["level"],
            "safety_correct": is_safety_correct,
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "top_doc": top_title,
            "top_score": round(top_score, 4)
        })

    avg_retrieval = total_retrieval_time / len(BENCHMARK_QUERIES)
    avg_rerank = total_rerank_time / len(BENCHMARK_QUERIES)
    safety_accuracy = (safety_matches / len(BENCHMARK_QUERIES)) * 100

    print("\n" + "=" * 65)
    print("   BENCHMARK SUMMARY RESULTS")
    print("=" * 65)
    print(f"Safety Detection Accuracy   : {safety_accuracy:.1f}% ({safety_matches}/{len(BENCHMARK_QUERIES)})")
    print(f"Average Retrieval Latency   : {avg_retrieval:.2f} ms")
    print(f"Average Reranker Latency    : {avg_rerank:.2f} ms")
    print(f"Total Pipeline Latency      : {(avg_retrieval + avg_rerank):.2f} ms (excluding external LLM)")
    print("=" * 65)

    return {
        "safety_accuracy": safety_accuracy,
        "avg_retrieval_ms": avg_retrieval,
        "avg_rerank_ms": avg_rerank,
        "results": results
    }


if __name__ == "__main__":
    run_evaluation()
