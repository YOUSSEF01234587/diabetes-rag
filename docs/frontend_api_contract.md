# Frontend API Contract

Backend: FastAPI on `localhost:8000`

---

## `GET /health`
```json
{
  "status": "ok",
  "index_ready": true,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "openai/gpt-oss-20b:free",
  "version": "2.0.0"
}
```

---

## `POST /api/chat`
**Request:**
```json
{
  "message": "What is the fasting glucose cutoff for diabetes?",
  "top_k": 8,
  "evidence_k": 5
}
```

**Response:**
```json
{
  "request_id": "a1b2c3d4",
  "answer": "Based on the ADA Standards of Care 2026, the fasting plasma glucose (FPG) cutoff for diagnosing diabetes is >= 126 mg/dL (>= 7.0 mmol/L). [Evidence 1][Evidence 2]",
  "confidence": "high",
  "grounded": true,
  "citations": [
    {
      "evidence_index": 1,
      "source": "ADA Standards of Care 2026",
      "page": 2,
      "section": "Screening and Diagnosis of Diabetes"
    }
  ],
  "sources": [
    {
      "source_id": "ada_soc_2026_diagnosis",
      "source_title": "2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes—2026",
      "short_title": "ADA Standards of Care 2026 - Diagnosis",
      "organization": "American Diabetes Association",
      "page": 2,
      "section": "Screening and Diagnosis of Diabetes",
      "doi": "10.2337/dc26-S002"
    }
  ],
  "evidence": [
    {
      "chunk_id": "4b75976adba0",
      "text": "Screening and Diagnosis of Diabetes...",
      "source_id": "ada_soc_2026_diagnosis",
      "source_label": "ADA Standards of Care 2026 - Diagnosis",
      "organization": "American Diabetes Association",
      "page": 2,
      "section": "Screening and Diagnosis of Diabetes"
    }
  ],
  "refused": false,
  "query_type": "threshold_question",
  "safety": {
    "requires_professional": false,
    "risk_level": "low",
    "risk_flags": []
  },
  "verification": {
    "passed": true,
    "issues": []
  },
  "timings": {
    "retrieval_ms": 102,
    "multi_query_ms": 0,
    "rerank_ms": 0,
    "llm_ms": 32000
  },
  "total_ms": 32500
}
```

### Response fields

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique ID for this request |
| `answer` | string | Grounded answer with `[Evidence N]` citations |
| `confidence` | string | `"high"` / `"medium"` / `"low"` / `"insufficient"` |
| `grounded` | boolean | Whether answer is grounded in evidence |
| `citations` | array | Citation metadata for each `[Evidence N]` in answer |
| `sources` | array | Source document metadata for each evidence chunk |
| `evidence` | array | Full evidence chunks used |
| `refused` | boolean | Whether the system refused to answer |
| `query_type` | string | `"threshold_question"` / `"comparison"` / `"classification"` / `"monitoring"` / `"standard"` / `"medication_question"` / `"emergency"` |
| `safety` | object | `risk_level`: `"low"` / `"medium"` / `"high"`. `risk_flags`: array of `"treatment_change"`, `"emergency_symptoms"`, `"dangerous_action"` |
| `verification` | object | Post-generation verification: `passed` (bool), `issues` (array of strings) |
| `timings` | object | Component latencies in ms |
| `total_ms` | number | End-to-end latency in ms |

### Refusal responses
When `refused=true`, the `answer` contains a polite refusal message. `confidence` is `"insufficient"`. The `query_type` indicates why:
- `"medication_question"` — asked for dosing/medication advice
- `"emergency"` — emergency symptoms detected (system directs to emergency services)
- `"personal_medical_advice"` — asked for personalized medical advice

---

## `POST /api/search`
**Request:**
```json
{
  "query": "prediabetes diagnosis criteria",
  "top_k": 8
}
```
**Response:**
```json
{
  "request_id": "e5f6g7h8",
  "query": "prediabetes diagnosis criteria",
  "results": [
    {
      "rank": "1",
      "chunk_id": "ac3650f3a557",
      "text": "...",
      "fusion_score": "0.0492",
      "dense_score": "0.2175",
      "bm25_score": "21.3198",
      "reranker_score": "0.0492",
      "source_id": "ada_soc_2026_diagnosis",
      "source_title": "...",
      "short_title": "ADA Standards of Care 2026 - Diagnosis",
      "organization": "American Diabetes Association",
      "page_pdf": "6",
      "page_document": "6",
      "section": "Diagnosis of Prediabetes",
      "subsection": "None",
      "doi": "10.2337/dc26-S002",
      "official_url": "https://diabetesjournals.org/care/article/49/Supplement_1/S2",
      "year": "2026",
      "authority": "high",
      "has_table": "True"
    }
  ],
  "timings": {...},
  "total_ms": 102
}
```

---

## `GET /api/sources`
**Response:**
```json
{
  "sources": [
    {
      "title": "...",
      "short_title": "...",
      "organization": "...",
      "year": "2026",
      "doi": "...",
      "official_url": "...",
      "pages": "...",
      "indexed_chunks": 50
    }
  ]
}
```

---

## `GET /api/stats`
**Response:**
```json
{
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "llm_model": "openai/gpt-oss-20b:free",
  "index_stats": {
    "total_chunks": 58,
    "dimension": 384,
    "collection": "diabetes_evidence"
  },
  "source_count": 2,
  "version": "2.0.0"
}
```

---

## Key metrics (frozen baseline)
- R@5: 1.0000
- MRR: 0.8236
- Source Accuracy: 1.0000
- Section Accuracy: 0.8739
- Retrieval latency: ~100ms
- Chunks: 58 (2 documents)
- Config hash: see `logs/pre_frontend_baseline.json`
