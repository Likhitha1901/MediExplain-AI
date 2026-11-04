MediExplain AI Evaluation Dashboard v2

This version evaluates both:
1. Raw ChromaDB retrieval.
2. The MedicalReranker class used in the existing chat route.

Copy evaluation_dashboard.py, evaluation_dashboard.html, and evaluation_dataset.example.json into your existing `rag` folder, replacing the old dashboard files only. Do not replace routes.py or your main chat files.

Run from project root:
    uvicorn rag.evaluation_dashboard:app --reload --port 8001

Open http://127.0.0.1:8001

Dataset `relevant_titles` must be verified against your ChromaDB titles. Human quality labels remain null until you review the answer and citations. This dashboard measures retrieval/reranking only; it does not yet measure full chat latency or automatically score answer correctness.
