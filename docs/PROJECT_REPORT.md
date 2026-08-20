# Clinical Evidence Copilot — Diabetes Edition

## Technical Project Report

**Hackathon:** Clinical Evidence Copilot
**Domain:** Diabetes Medical Information Retrieval & Generation
**Date:** August 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Data Sources](#4-data-sources)
5. [Ingestion Pipeline](#5-ingestion-pipeline)
6. [PDF Parsing](#6-pdf-parsing)
7. [Section Detection & Metadata](#7-section-detection--metadata)
8. [Chunking Strategy](#8-chunking-strategy)
9. [Embedding Model](#9-embedding-model)
10. [Vector Store](#10-vector-store)
11. [Lexical Search (BM25)](#11-lexical-search-bm25)
12. [Hybrid Retrieval & Fusion](#12-hybrid-retrieval--fusion)
13. [Intent Detection](#13-intent-detection)
14. [Query Expansion](#14-query-expansion)
15. [Evidence Selection](#15-evidence-selection)
16. [Reranking (Disabled)](#16-reranking-disabled)
17. [LLM Generation](#17-llm-generation)
18. [Provider Chain & Failover](#18-provider-chain--failover)
19. [Prompt Engineering](#19-prompt-engineering)
20. [Safety Layer](#20-safety-layer)
21. [Answer Verification](#21-answer-verification)
22. [Citation Engine](#22-citation-engine)
23. [Conflict Detection](#23-conflict-detection)
24. [Grounding Score](#24-grounding-score)
25. [Backend API](#25-backend-api)
26. [Frontend Architecture](#26-frontend-architecture)
27. [UI Components](#27-ui-components)
28. [Responsive Design](#28-responsive-design)
29. [Markdown & LaTeX Rendering](#29-markdown--latex-rendering)
30. [Evaluation Results](#30-evaluation-results)
31. [Visual QA Testing](#31-visual-qa-testing)
32. [Configuration Reference](#32-configuration-reference)
33. [Limitations & Future Work](#33-limitations--future-work)
34. [Conclusion](#34-conclusion)

---

## 1. Executive Summary

Clinical Evidence Copilot — Diabetes Edition is a Retrieval-Augmented Generation (RAG) system designed to answer medical questions about diabetes using only authoritative clinical sources. Built for the Clinical Evidence Copilot hackathon, the system retrieves evidence from peer-reviewed guidelines and NIH patient education materials, then generates grounded, cited answers with built-in safety guardrails.

The architecture chains a browser-based static frontend through a FastAPI backend into an intent-aware hybrid retrieval pipeline. Queries are classified, expanded with medical terminology, and fused across dense (embedding-based) and sparse (BM25) search. Retrieved evidence passes through a safety layer, is fed to a multi-provider LLM chain, and returns a JSON response with citations, grounding scores, confidence levels, and refusal signals when appropriate.

Key results include 100% hit rate and source accuracy on a 40-question evaluation set, sub-100ms retrieval latency, and 169/169 passing visual QA checks across six viewports with zero console errors.

---

## 2. Problem Statement

Clinicians, researchers, and patients need fast, reliable answers to diabetes-related questions grounded in authoritative medical literature. General-purpose LLMs hallucinate medical facts, fabricate citations, and provide advice that may be clinically dangerous. There is a need for a system that:

- Retrieves only from verified, authoritative sources
- Grounds every claim in specific document passages
- Refuses to answer when evidence is insufficient
- Detects and blocks potentially dangerous medical advice
- Provides transparent citations for every statement
- Operates without requiring user authentication for rapid clinical consultation

Clinical Evidence Copilot addresses these requirements with a domain-specific RAG system focused on diabetes care, combining hybrid retrieval, multi-layer safety, and citation-verified generation.

---

## 3. System Architecture

The system follows a linear pipeline with branching logic at the intent and safety stages:

```
Browser → Static Frontend → FastAPI Backend → Intent Detection → Hybrid Retrieval
→ Evidence Selection → Safety Layer → LLM Generation → Citation/Verification → JSON Response
```

| Layer | Technology | Responsibility |
|-------|-----------|----------------|
| Presentation | Vanilla JS ES6, HTML5, CSS3 | User interaction, evidence display |
| API Gateway | FastAPI (Python) | Request routing, response schema |
| Intent Detection | Rule-based classifier | Source routing, query categorization |
| Retrieval | ChromaDB + BM25 + RRF | Hybrid dense/sparse search |
| Evidence | Custom validator | Relevance filtering, grounding |
| Safety | Regex + rule engine | Risk pattern detection, refusal |
| Generation | Multi-provider LLM chain | Answer synthesis with citations |
| Verification | Post-generation checker | 5-check answer validation |

The backend runs as a single process serving both the API and static files. No external message queues, caches, or databases beyond ChromaDB persistent storage are required.

---

## 4. Data Sources

The system ingests exactly two authoritative documents:

| # | Document | Organization | Pages | DOI / Status | Chunks |
|---|----------|-------------|-------|--------------|--------|
| 1 | `ADA_Standards_of_Care_2026_Diagnosis.pdf` | American Diabetes Association | 23 | 10.2337/dc26-S002 | 58 |
| 2 | `NIDDK_Diabetes_Prediabetes_Tests.pdf` | NIH / NIDDK | 1 | Public domain | 58 |

**Source 1 — ADA Standards of Care 2026**
Full title: "2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes—2026". This is a 23-page clinical guideline covering diagnostic criteria, classification, and testing recommendations for diabetes and prediabetes. It represents the current standard of care published by the American Diabetes Association.

**Source 2 — NIDDK Diabetes and Prediabetes Tests**
A concise 1-page patient education document from the National Institute of Diabetes and Digestive and Kidney Diseases (NIH). It covers blood tests used to diagnose diabetes and prediabetes, including FPG, A1C, and OGTT, written for a general audience. As a U.S. government publication, it is in the public domain.

Both documents are stored in `data/raw/` and processed through the ingestion pipeline to produce 116 total chunks with full metadata.

---

## 5. Ingestion Pipeline

The ingestion pipeline converts raw PDFs into searchable, chunked records with rich metadata. The pipeline consists of four stages:

1. **PDF Extraction** — PyMuPDF (`fitz`) extracts raw text page-by-page
2. **Section Detection** — Regex patterns identify headings and subsections
3. **Source-Aware Chunking** — Different chunk sizes per document type
4. **Metadata Enrichment** — Each chunk receives full source and positional metadata

The pipeline is idempotent: re-running ingestion on the same documents produces identical chunks. Parent/child chunk relationships are preserved to enable context expansion during retrieval.

---

## 6. PDF Parsing

PDF text extraction uses PyMuPDF (`fitz`), selected for its speed, accuracy, and ability to handle complex layouts including multi-column clinical guidelines.

| Aspect | Detail |
|--------|--------|
| Library | PyMuPDF (`fitz`) |
| Extraction method | Page-by-page text extraction |
| Page count (ADA) | 23 |
| Page count (NIDDK) | 1 |
| Output | Raw text strings per page |

PyMuPDF is preferred over alternatives such as pdfplumber or pdfminer due to its C-based implementation providing faster extraction speeds, particularly relevant when processing multi-page clinical guidelines with complex formatting.

---

## 7. Section Detection & Metadata

After text extraction, the pipeline applies regex-based heading detection to identify logical sections within each document. Detected headings are used to assign `section` and `subsection` metadata to each chunk.

**Metadata fields applied to every chunk:**

| Field | Description |
|-------|-------------|
| `source_id` | Unique identifier for the source document |
| `source_title` | Full document title |
| `short_title` | Abbreviated title for display |
| `organization` | Publishing organization (ADA, NIH/NIDDK) |
| `year` | Publication year |
| `doi` | Digital Object Identifier (where available) |
| `official_url` | Canonical URL to the source |
| `page` | Page number within the PDF |
| `section` | Top-level section heading |
| `subsection` | Nested section heading |

This metadata enables source-specific routing, section-aware boosting, and detailed citation rendering in the frontend.

---

## 8. Chunking Strategy

Chunking parameters are source-aware, with different configurations for clinical guidelines versus patient education materials:

| Parameter | ADA Guidelines | NIDDK Tests |
|-----------|---------------|-------------|
| Chunk size | 700 tokens | 300 tokens |
| Overlap | 100 tokens | 0 tokens |
| Total chunks | 58 | 58 |

**Rationale:**
- **ADA (700 tokens, 100 overlap):** Clinical guidelines contain dense, interconnected information. Larger chunks preserve context across diagnostic criteria, recommendations, and evidence grades. The 100-token overlap prevents information loss at chunk boundaries.
- **NIDDK (300 tokens, 0 overlap):** The 1-page patient education document is concise and self-contained. Smaller chunks enable precise retrieval of specific test descriptions without redundancy.

Parent/child chunk relationships are preserved in metadata, enabling the retrieval system to fetch a matching chunk and expand to its parent for additional context when needed.

---

## 9. Embedding Model

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Framework | Sentence-Transformers |
| Type | Dense bi-encoder |

The `all-MiniLM-L6-v2` model produces 384-dimensional dense vectors optimized for semantic similarity. It balances embedding quality with inference speed, enabling sub-100ms retrieval latency in production. The model is well-suited for medical text despite not being domain-specific, as clinical guideline language is structurally similar to the general English text in its training corpus.

Embeddings are computed at ingestion time and stored in ChromaDB. Query embeddings are computed at runtime and compared against the stored corpus using cosine distance.

---

## 10. Vector Store

| Property | Value |
|----------|-------|
| Database | ChromaDB |
| Storage | Persistent (disk) |
| Distance metric | Cosine similarity |
| Collection | Single collection for all documents |
| Top-K default | 8 |
| Similarity threshold | 0.35 |

ChromaDB serves as the primary vector store, providing persistent storage and efficient approximate nearest neighbor search. The cosine distance metric measures semantic similarity between query and chunk embeddings. Chunks scoring below the 0.35 similarity threshold are filtered out to prevent low-relevance results from polluting the evidence set.

The system retrieves Top-K=8 candidates from the vector store, which are then combined with BM25 results through reciprocal rank fusion.

---

## 11. Lexical Search (BM25)

| Property | Value |
|----------|-------|
| Library | `rank_bm25.BM25Okapi` |
| Tokenization | Whitespace + lowercase |
| Index scope | All chunks across both sources |

In addition to dense vector search, the system maintains a BM25Okapi lexical index over all chunks. BM25 provides exact keyword matching capability, which is critical for medical queries containing specific terminology (e.g., "HbA1c", "FPG", "OGTT") that may not be fully captured by semantic similarity alone.

The BM25 index is built in-memory at startup from the same chunked data used to populate ChromaDB. At query time, BM25 produces its own ranked list of results, which are fused with the dense retrieval results.

---

## 12. Hybrid Retrieval & Fusion

The retrieval system combines dense and sparse search results using Reciprocal Rank Fusion (RRF):

| Parameter | Value |
|-----------|-------|
| Fusion method | Reciprocal Rank Fusion (RRF) |
| Dense weight | 0.6 |
| Top-K | 8 |
| Similarity threshold | 0.35 |
| Retrieval latency | ~100ms |

**RRF Formula:**

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where `k = 60` (standard RRF constant) and `rank_i(d)` is the rank of document `d` in result list `i`.

The `DENSE_WEIGHT=0.6` parameter allows tuning the balance between semantic and lexical relevance. The default 60/40 weighting favors semantic understanding while preserving keyword precision. Results falling below the `SIMILARITY_THRESHOLD=0.35` on the dense path are excluded before fusion.

---

## 13. Intent Detection

Intent detection classifies incoming queries to route them to the appropriate source and determine the response strategy.

| Intent Category | Description |
|----------------|-------------|
| `diagnosis` | Diagnostic criteria, classification |
| `testing` | Blood tests, screening, laboratory |
| `treatment` | Medication, therapy, management |
| `prevention` | Risk reduction, lifestyle, prediabetes |
| `complications` | Comorbidities, long-term effects |
| `nutrition` | Diet, dietary guidelines |
| `emergency` | Urgent/dangerous symptoms |
| `general` | Broad or unclear queries |

The intent classifier also determines source routing:
- **NIDDK routing** — Testing-related queries, patient-education-level questions
- **ADA routing** — Clinical guidelines, diagnostic criteria, treatment protocols
- **Both** — Queries spanning multiple categories

Intent detection is rule-based (no LLM call), ensuring low latency for query classification.

---

## 14. Query Expansion

The system expands queries with medical terminology mappings to improve retrieval recall without requiring an LLM call.

| Feature | Detail |
|---------|--------|
| Method | Rule-based terminology mapping |
| Scope | Common diabetes abbreviations and synonyms |
| LLM requirement | None |
| Latency impact | Negligible |

Examples of expanded mappings:
- "A1C" ↔ "HbA1c" ↔ "hemoglobin A1c"
- "FPG" ↔ "fasting plasma glucose" ↔ "fasting blood sugar"
- "T2D" ↔ "type 2 diabetes"
- "T1D" ↔ "type 1 diabetes"

Query expansion ensures that abbreviations and synonyms present in user queries are mapped to the canonical terminology used in the source documents, improving the likelihood of relevant chunk retrieval.

---

## 15. Evidence Selection

After hybrid retrieval and fusion, the evidence selection stage filters and ranks candidate chunks for inclusion in the LLM prompt.

**Selection criteria:**
- Similarity score ≥ 0.35 (from dense retrieval path)
- Relevance to detected intent category
- Source match based on intent-based routing
- Section boosting: +0.10 score boost for chunks in sections relevant to the query intent

**Top-K output:** The final evidence set contains up to 8 chunks, ordered by fused relevance score. Each chunk carries full metadata for downstream citation and grounding.

Section boosting improves precision by preferring chunks from semantically relevant sections (e.g., diagnostic criteria sections for diagnosis queries) without excluding high-relevance chunks from other sections.

---

## 16. Reranking (Disabled)

| Property | Value |
|----------|-------|
| Model | `BAAI/bge-reranker-base` |
| Enabled | **false** |
| Rerank Top-K | 15 |

The system includes integration for `BAAI/bge-reranker-base`, a cross-encoder reranker that can re-score the top-15 candidates from the initial retrieval stage. However, the reranker is currently **disabled** (`RERANKER_ENABLED=false`).

**Reason for disabling:** The reranker adds significant inference latency (~200-500ms per query) which exceeds the performance budget for the hackathon demo. With only 2 source documents and 116 chunks, the initial hybrid retrieval achieves sufficient precision without reranking.

Enabling the reranker would be beneficial as the corpus scales to dozens or hundreds of documents, where cross-encoder re-ranking provides meaningful precision improvements over bi-encoder retrieval.

---

## 17. LLM Generation

The system generates answers using a multi-provider LLM chain with configurable model selection.

| Provider | Model | Role |
|----------|-------|------|
| Google Gemini | `gemini-2.0-flash` | Primary |
| Groq | `llama-3.3-70b-versatile` | Secondary |
| OpenRouter | `openai/gpt-oss-20b:free` | Tertiary |
| Refusal fallback | N/A | Final fallback |

Generation receives the evidence chunks, query, intent classification, and safety signals. The prompt instructs the LLM to answer only from provided evidence, cite sources, and express uncertainty when evidence is insufficient.

The generated answer is then subjected to post-generation verification before being returned to the user.

---

## 18. Provider Chain & Failover

The system implements a cascading provider chain with automatic failover:

```
Gemini 2.0 Flash → [timeout/error] → Groq Llama 3.3 70B → [timeout/error] → OpenRouter GPT-OSS 20B → [timeout/error] → Refusal fallback
```

| Parameter | Value |
|-----------|-------|
| Provider timeout | 30.0 seconds |
| Failover trigger | Timeout or API error |
| Final fallback | Structured refusal response |

Each provider is attempted in order. If a provider times out (exceeding `PROVIDER_TIMEOUT_SECONDS=30.0`) or returns an error, the system automatically falls through to the next provider. If all providers fail, the system returns a structured refusal with `refusal_reason=provider_failure`.

This chain ensures maximum availability: even if the primary provider is down, the system continues to function through backup providers.

---

## 19. Prompt Engineering

The system prompt is designed to enforce evidence-grounded generation:

**Core instructions:**
- Answer ONLY from provided evidence chunks
- Cite specific sources for every claim
- Express uncertainty when evidence is insufficient
- Never fabricate information not present in the evidence
- Do not provide medical advice or treatment recommendations
- Detect and refuse dangerous medical queries

**Prompt structure:**
1. System role definition (clinical evidence assistant)
2. Evidence chunks with source metadata
3. User query
4. Instructions for citation format
5. Safety constraints

The prompt is calibrated to balance comprehensiveness with honesty — the system should provide thorough answers when evidence supports them, but clearly state when evidence is insufficient or when the question falls outside the scope of available sources.

---

## 20. Safety Layer

The safety layer operates at two points: pre-generation (query screening) and post-generation (answer validation).

**Pre-generation risk detection:**

| Risk Pattern | Detection Method | Action |
|-------------|-----------------|--------|
| `treatment_change` | Regex pattern matching | Refuse with `medical_advice` |
| `emergency_symptoms` | Regex pattern matching | Refuse with `emergency` |
| `dangerous_action` | Regex pattern matching | Refuse with `dangerous_action` |

**Post-generation verification (5 checks):**

1. **Citation validity** — All cited sources exist in the evidence set
2. **No fabrication** — Answer does not contain claims unsupported by evidence
3. **Safety compliance** — Answer does not contain dangerous medical advice
4. **Confidence alignment** — Confidence level matches evidence strength
5. **Source consistency** — Citations match the content they claim to support

**Refusal reasons (8 categories):**

| Reason | Trigger |
|--------|---------|
| `low_relevance` | Query unrelated to available sources |
| `insufficient_evidence` | Evidence too weak to support answer |
| `provider_failure` | All LLM providers failed |
| `medical_advice` | Query requests specific treatment advice |
| `emergency` | Query describes emergency symptoms |
| `verification_failed` | Post-generation checks failed |
| `no_safe_answer` | No safe answer possible |
| `technical_error` | System error during processing |

---

## 21. Answer Verification

After LLM generation, the system runs a 5-check verification pipeline implemented in `evidence_validator.py`:

| Check | Description | Failure Action |
|-------|-------------|----------------|
| Citation validity | Cited sources exist in evidence set | Remove invalid citations |
| No fabrication | Claims are supported by evidence | Flag for review |
| Safety compliance | No dangerous medical advice | Trigger refusal |
| Confidence alignment | Confidence matches evidence strength | Adjust confidence level |
| Source consistency | Citations match their claimed content | Correct misattributed citations |

**Confidence levels:**

| Level | Criteria |
|-------|----------|
| `high` | Multiple strong evidence sources, clear match |
| `medium` | Limited evidence, moderate match |
| `low` | Weak evidence, indirect match |
| `insufficient` | Evidence too weak to support any answer |

Verification ensures that the final answer meets quality and safety standards before being returned to the user.

---

## 22. Citation Engine

The citation engine (`citation_engine.py`) manages citation generation, deduplication, and formatting.

**Citation pipeline:**
1. Extract source references from generated answer
2. Match citations against evidence chunk metadata
3. Deduplicate identical source references
4. Format citations with source title, organization, and year

**Deduplication logic:** When multiple evidence chunks originate from the same document and section, the citation engine consolidates them into a single citation entry rather than repeating the same reference. This produces cleaner, more readable answers.

Citations are returned in the response as structured objects containing source title, short title, organization, DOI, and official URL, enabling the frontend to render clickable, formatted citation references.

---

## 23. Conflict Detection

The conflict detector (`conflict_detector.py`) identifies disagreements between retrieved evidence chunks and between the LLM's generated answer and the evidence.

**Detection types:**

| Type | Description |
|------|-------------|
| Threshold differences | Different diagnostic thresholds mentioned in different contexts |
| Genuine disagreements | Contradictory claims between sources |
| Answer-evidence mismatch | LLM answer contradicts retrieved evidence |

**Display logic in the frontend:**

| Conflict Type | UI Treatment |
|--------------|-------------|
| Threshold differences | Blue "Clinical Info" notice — contextual, not contradictory |
| Genuine disagreements | Orange warning — flagged for user attention |

This distinction is critical in clinical contexts where different diagnostic thresholds (e.g., FPG vs. A1C criteria) represent complementary information rather than contradictions.

---

## 24. Grounding Score

The grounding score is calculated in `evidence_validator.py` and measures how well the generated answer is supported by the retrieved evidence.

**Calculation factors:**
- Proportion of answer claims traceable to evidence chunks
- Strength of similarity scores for cited evidence
- Consistency between citation references and cited content
- Coverage of the answer across the evidence set

**Score interpretation:**
- High grounding score → Answer is well-supported by evidence
- Low grounding score → Answer may contain unsupported claims

The grounding score is included in the response as `evidence_validation`, enabling the frontend to display a grounding indicator and helping users assess answer reliability.

---

## 25. Backend API

The backend is built with FastAPI, serving both the API endpoints and static frontend files.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Main query endpoint |
| `GET` | `/api/sources` | List available sources |
| `GET` | `/api/evidence` | Retrieve evidence for a query |
| `GET` | `/api/citations` | Get citation details |

**Chat request schema:**

The `POST /api/chat` endpoint accepts a JSON body with a `message` field (not `question`).

**Chat response schema:**

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique request identifier |
| `answer` | string | Generated answer text |
| `confidence` | string | high / medium / low / insufficient |
| `grounded` | boolean | Whether answer is grounded in evidence |
| `citations` | array | Citation objects with source details |
| `sources` | array | Source documents referenced |
| `evidence` | array | Evidence chunks used |
| `refused` | boolean | Whether the query was refused |
| `refusal_reason` | string | Reason for refusal (if applicable) |
| `query_type` | string | Detected intent category |
| `safety` | object | Safety screening results |
| `verification` | object | Post-generation verification results |
| `timings` | object | Timing breakdown |
| `total_ms` | number | Total response time in milliseconds |
| `evidence_validation` | object | Grounding score details |

---

## 26. Frontend Architecture

The frontend is built with vanilla JavaScript ES6 modules, with no build step, no framework, and no external dependencies.

| Aspect | Choice |
|--------|--------|
| Framework | None (vanilla JS ES6) |
| Build step | None |
| Module system | ES6 native modules |
| Controller | `src/app.js` |
| Persistence | localStorage (max 50 conversations) |
| CSS | `main.css` (~2600 lines) |

**File structure:**

| Directory | Contents |
|-----------|----------|
| `src/app.js` | Main SPA controller |
| `Chat/message.js` | Chat message rendering |
| `Drawer/evidence-drawer.js` | Evidence panel component |
| `Evidence/conflict.js` | Conflict display component |
| `Services/api.js` | HTTP client for backend API |
| `Services/chats.js` | localStorage persistence, conversation management |
| `Utils/markdown.js` | Markdown and LaTeX rendering |
| `Utils/dom.js` | DOM utility helpers |
| `CSS/main.css` | Clinical design system (~2600 lines) |

The SPA architecture uses `app.js` as the central controller, coordinating between chat interactions, evidence display, and API communication. Conversations persist in localStorage with a maximum of 50 stored conversations.

---

## 27. UI Components

**Chat Component (`Chat/message.js`):**
Renders user and assistant messages, including markdown-formatted answers with inline citations. Supports conversation history display and new message submission.

**Evidence Drawer (`Drawer/evidence-drawer.js`):**
Displays retrieved evidence chunks, source information, and grounding details. On desktop, renders in a grid-column layout alongside the chat. On mobile, renders as a slide-over panel.

**Conflict Display (`Evidence/conflict.js`):**
Renders conflict notices with contextual styling:
- Blue "Clinical Info" notice for threshold differences (complementary information)
- Orange warning for genuine disagreements (contradictory claims)

**Header:**
Contains the brand logo and navigation, with gradient favicon SVG for the browser tab.

---

## 28. Responsive Design

The layout adapts across three breakpoints:

| Breakpoint | Layout |
|-----------|--------|
| ≥1440px | 3-column grid (sidebar + chat + evidence) |
| ≤1024px | Sidebar collapses, 2-column layout |
| ≤768px | Mobile single-column, evidence as slide-over |

**Tested viewports:**

| Viewport | Width | Category |
|----------|-------|----------|
| Desktop XL | 1440px | Full 3-column |
| Desktop | 1280px | Full 3-column |
| Laptop | 1024px | Collapsed sidebar |
| Tablet | 768px | Mobile layout |
| Mobile L | 390px | Mobile layout |
| Mobile | 375px | Mobile layout |

The CSS design system in `main.css` (~2600 lines) implements a clinical aesthetic with clean typography, structured spacing, and accessible color contrast suitable for medical information display.

---

## 29. Markdown & LaTeX Rendering

The `Utils/markdown.js` module handles rendering of LLM-generated markdown and medical notation:

**LaTeX normalization:**

| Symbol | Input | Normalized |
|--------|-------|-----------|
| ≥ | `\geq`, `≥` | `≥` |
| ≤ | `\leq`, `≤` | `≤` |
| ± | `\pm`, `±` | `±` |
| × | `\times`, `×` | `×` |

**Markdown rendering:**
- **Bold** text (`**...**`)
- Headings (`#`, `##`, `###`)
- Unordered and ordered lists
- Multi-citation links (clickable references to source documents)

Medical notation is normalized to Unicode symbols for consistent cross-browser rendering without requiring MathJax or KaTeX. Citation references are rendered as clickable links that scroll to the corresponding entry in the evidence panel.

---

## 30. Evaluation Results

The system was evaluated on a 40-question test set covering diabetes diagnosis, testing, and clinical management.

**Retrieval metrics:**

| Metric | Value |
|--------|-------|
| Recall@1 | 0.2414 |
| Recall@3 | 0.6025 |
| Recall@5 | 0.7929 |
| Recall@10 | 1.0000 |
| MRR | 0.8316 |
| Hit Rate | 1.0000 |
| Source Accuracy | 1.0000 |
| Section Accuracy | 0.8739 |
| Retrieval latency | ~100ms |

**Interpretation:**
- **100% Hit Rate** — Every query retrieves at least one relevant chunk
- **100% Source Accuracy** — The correct source document is always included in results
- **87.4% Section Accuracy** — The correct section within the source is identified in the vast majority of cases
- **83.2% MRR** — On average, the first relevant result appears in the top 1-2 positions
- **100% Recall@10** — All relevant chunks are captured within the top 10 results

**API integration test results:**

| Scenario | Expected | Result |
|----------|----------|--------|
| FPG query | Answered | PASS |
| France-specific query | Refused | PASS |
| Metformin advice query | Refused | PASS |
| Emergency symptoms query | Refused | PASS |
| Prediabetes query | Answered | PASS |

All 5 API integration scenarios passed, confirming correct behavior for answered queries, refused medical advice, and emergency detection.

---

## 31. Visual QA Testing

Visual QA was conducted using Playwright automated browser testing across 6 viewports.

| Metric | Value |
|--------|-------|
| Total checks | 169 |
| Passed | 169 |
| Failed | 0 |
| Console errors | 0 |
| Horizontal overflow | 0 |

**Test coverage across viewports:**

| Viewport | Width | Checks |
|----------|-------|--------|
| Desktop XL | 1440px | ✓ |
| Desktop | 1280px | ✓ |
| Laptop | 1024px | ✓ |
| Tablet | 768px | ✓ |
| Mobile L | 390px | ✓ |
| Mobile | 375px | ✓ |

All 169 visual QA checks passed with zero console errors and zero horizontal overflow across all tested viewports. The frontend renders correctly and responsively at every breakpoint.

**Backend integrity:**
- All 31 JavaScript files pass `node --check` syntax validation
- Backend code untouched: `git diff -- backend/app/` produces empty output

---

## 32. Configuration Reference

All configuration values are defined in `backend/app/config.py`:

| Parameter | Value |
|-----------|-------|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_DIMENSION` | 384 |
| `CHUNK_SIZE_TOKENS` | 700 |
| `CHUNK_OVERLAP_TOKENS` | 100 |
| `TOP_K` | 8 |
| `RERANK_TOP_K` | 15 |
| `SIMILARITY_THRESHOLD` | 0.35 |
| `DENSE_WEIGHT` | 0.6 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` |
| `RERANKER_ENABLED` | false |
| `GEMINI_MODEL` | `gemini-2.0-flash` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `LLM_MODEL` | `openai/gpt-oss-20b:free` |
| `LLM_PROVIDER` | auto |
| `PROVIDER_TIMEOUT_SECONDS` | 30.0 |

---

## 33. Limitations & Future Work

**Current limitations:**

| Limitation | Impact |
|-----------|--------|
| Single domain (diabetes) | Cannot answer questions about other medical conditions |
| 2 source documents | Small corpus limits breadth of answers |
| Reranker disabled | Retrieval precision may degrade at larger scales |
| No multimodal support | PDF images, figures, and tables are not parsed |
| No user authentication | No access control or usage tracking |
| No A/B testing infrastructure | Cannot experimentally compare pipeline configurations |

**Future work:**

| Area | Description |
|------|-------------|
| Corpus expansion | Add more clinical guidelines, research papers, and patient materials |
| Reranker activation | Enable BAAI/bge-reranker-base for improved precision at scale |
| Multimodal parsing | Extract and interpret figures, tables, and charts from PDFs |
| Multi-domain support | Extend to cardiology, oncology, and other clinical domains |
| User authentication | Add login, session management, and usage analytics |
| A/B testing | Build experimentation framework for pipeline parameter tuning |
| Real-time data | Integrate with live clinical trial databases and drug interaction APIs |
| Streaming responses | Implement Server-Sent Events for progressive answer rendering |

---

## 34. Conclusion

Clinical Evidence Copilot — Diabetes Edition demonstrates that a focused RAG system can achieve high retrieval accuracy and safe, grounded generation from authoritative clinical sources. The system retrieves with 100% hit rate, verifies answers through a 5-check pipeline, and refuses unsafe queries across 8 refusal categories — all within a responsive, accessible frontend requiring no build tools or frameworks.

The hybrid retrieval architecture combining dense embeddings (MiniLM-L6-v2) with sparse lexical search (BM25) through Reciprocal Rank Fusion provides robust recall across different query types, from keyword-heavy test abbreviations to semantic clinical questions. The multi-provider LLM chain with automatic failover ensures high availability, while the safety layer provides clinical-grade guardrails appropriate for medical information retrieval.

With 116 chunks from 2 authoritative sources, the system operates at a scale appropriate for a hackathon proof of concept. The architecture is designed to scale: enabling the reranker, expanding the corpus, and adding multimodal parsing are straightforward extensions of the existing pipeline. Clinical Evidence Copilot establishes a foundation for trustworthy, citation-verified medical AI that can grow to serve broader clinical domains.

---

*Report generated for the Clinical Evidence Copilot hackathon — August 2026*
