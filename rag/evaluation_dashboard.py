from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from rag.vector_store import MedicalVectorStore
    from rag.reranker import MedicalReranker
except Exception as exc:
    MedicalVectorStore = None
    MedicalReranker = None
    IMPORT_ERROR = str(exc)
else:
    IMPORT_ERROR = ""

app = FastAPI(title="MediExplain AI Evaluation Dashboard")
vector_store = None
reranker = None


class EvaluationCase(BaseModel):
    question: str = Field(min_length=1)
    relevant_titles: list[str] = []
    answer_correct: bool | None = None
    faithful: bool | None = None
    citation_correct: bool | None = None
    safety_expected: bool | None = None
    safety_passed: bool | None = None
    answer_latency_seconds: float | None = None


class EvaluationRequest(BaseModel):
    cases: list[EvaluationCase] = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=10, ge=1, le=50)


def get_components():
    global vector_store, reranker
    if MedicalVectorStore is None or MedicalReranker is None:
        raise HTTPException(status_code=500, detail=f"Could not import project retrieval components: {IMPORT_ERROR}")
    if vector_store is None:
        try:
            vector_store = MedicalVectorStore()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not open ChromaDB: {exc}")
    if reranker is None:
        try:
            reranker = MedicalReranker()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not initialize reranker: {exc}")
    return vector_store, reranker


def get_raw_docs(store, query: str, candidate_k: int):
    result = store.search(query=query, top_k=candidate_k)
    docs = (result.get("documents") or [[]])[0] if result else []
    metas = (result.get("metadatas") or [[]])[0] if result else []
    normalized = []
    for i, content in enumerate(docs or []):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        normalized.append({
            "content": content or "",
            "title": meta.get("title", ""),
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "category": meta.get("category", "")
        })
    return normalized


def title_relevant(title: str, expected_titles: list[str]) -> bool:
    title = (title or "").casefold().strip()
    return any(
        (expected.casefold().strip() in title or title in expected.casefold().strip())
        for expected in expected_titles if expected.strip()
    )


def metrics_for_ranked_lists(cases, ranked_lists, k):
    recall_hits, precisions, reciprocal_ranks = [], [], []
    for case, docs in zip(cases, ranked_lists):
        if not case.relevant_titles:
            continue
        flags = [title_relevant(d.get("title", ""), case.relevant_titles) for d in docs[:k]]
        recall_hits.append(1.0 if any(flags) else 0.0)  # Hit Rate@K
        precisions.append(sum(flags) / k)
        first = next((i + 1 for i, flag in enumerate(flags) if flag), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
    return {
        "recall_at_k": sum(recall_hits) / len(recall_hits) if recall_hits else None,
        "precision_at_k": sum(precisions) / len(precisions) if precisions else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "labelled_cases": len(recall_hits)
    }


def mean(values):
    return sum(values) / len(values) if values else None


def labelled_rate(values):
    known = [v for v in values if v is not None]
    return sum(1 for v in known if v) / len(known) if known else None


def make_rows(cases, ranked_lists, k):
    rows = []
    for case, docs in zip(cases, ranked_lists):
        flags = [title_relevant(d.get("title", ""), case.relevant_titles) for d in docs[:k]] if case.relevant_titles else []
        rows.append({
            "question": case.question,
            "retrieval_status": "labelled" if case.relevant_titles else "missing relevant_titles",
            "retrieved": [
                {
                    "rank": i + 1,
                    "title": doc.get("title") or "(untitled)",
                    "source": doc.get("source") or "Unknown source",
                    "url": doc.get("url") or "",
                    "relevant": flags[i] if i < len(flags) else None
                } for i, doc in enumerate(docs[:k])
            ]
        })
    return rows


@app.get("/", response_class=HTMLResponse)
def dashboard():
    path = Path(__file__).with_name("evaluation_dashboard.html")
    if not path.exists():
        raise HTTPException(status_code=500, detail="evaluation_dashboard.html is missing next to evaluation_dashboard.py")
    return path.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "vector_store_imported": MedicalVectorStore is not None,
        "reranker_imported": MedicalReranker is not None,
        "vector_store_initialized": vector_store is not None,
        "reranker_initialized": reranker is not None,
        "import_error": IMPORT_ERROR or None
    }


@app.get("/api/evaluation/sample")
def sample():
    path = Path(__file__).with_name("evaluation_dataset.example.json")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/evaluation/run")
def run_evaluation(request: EvaluationRequest):
    store, ranker = get_components()
    raw_lists, reranked_lists = [], []
    raw_times, rerank_times = [], []

    for case in request.cases:
        start = time.perf_counter()
        docs = get_raw_docs(store, case.question, request.candidate_k)
        raw_times.append(time.perf_counter() - start)
        raw_lists.append(docs[:request.k])

        start = time.perf_counter()
        ranked = ranker.rerank(query=case.question, documents=docs, top_k=request.k)
        rerank_times.append(time.perf_counter() - start)
        reranked_lists.append([item["document"] for item in ranked])

    raw_metrics = metrics_for_ranked_lists(request.cases, raw_lists, request.k)
    reranked_metrics = metrics_for_ranked_lists(request.cases, reranked_lists, request.k)

    answer_latencies = [c.answer_latency_seconds for c in request.cases if c.answer_latency_seconds is not None and c.answer_latency_seconds >= 0]
    results = {
        "case_count": len(request.cases),
        "k": request.k,
        "candidate_k": request.candidate_k,
        "raw_retrieval_metrics": raw_metrics,
        "reranked_metrics": reranked_metrics,
        "other_metrics": {
            "answer_correctness": labelled_rate([c.answer_correct for c in request.cases]),
            "faithfulness": labelled_rate([c.faithful for c in request.cases]),
            "citation_accuracy": labelled_rate([c.citation_correct for c in request.cases]),
            "safety_pass_rate": labelled_rate([
                c.safety_passed if c.safety_expected is True else None for c in request.cases
            ]),
            "mean_retrieval_latency_seconds": mean(raw_times),
            "mean_rerank_latency_seconds": mean(rerank_times),
            "mean_answer_latency_seconds": mean(answer_latencies)
        },
        "label_counts": {
            "retrieval_labelled_cases": sum(bool(c.relevant_titles) for c in request.cases),
            "answer_correctness_labelled": sum(c.answer_correct is not None for c in request.cases),
            "faithfulness_labelled": sum(c.faithful is not None for c in request.cases),
            "citation_labelled": sum(c.citation_correct is not None for c in request.cases),
            "safety_labelled": sum(c.safety_expected is True and c.safety_passed is not None for c in request.cases)
        },
        "raw_rows": make_rows(request.cases, raw_lists, request.k),
        "reranked_rows": make_rows(request.cases, reranked_lists, request.k),
        "notes": [
            "Recall@K here means Hit Rate@K: at least one expected relevant title appears in the top K.",
            "Precision@K divides relevant titles in the top K by K.",
            "MRR is the mean reciprocal rank of the first matching expected title.",
            "This compares raw ChromaDB results against the project's MedicalReranker output.",
            "Answer correctness, faithfulness, citation accuracy, and safety require human-reviewed labels.",
            "Latency is measured separately for vector retrieval and reranking. Full chat latency needs instrumentation in the actual /api/chat route."
        ]
    }
    return results
