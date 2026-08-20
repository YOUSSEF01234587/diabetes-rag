"""Diabetes RAG Backend - Main FastAPI Application."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import uuid
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import (
    TOP_K, RERANK_TOP_K, EMBEDDING_MODEL, LLM_MODEL,
    RERANKER_ENABLED, SOURCE_REGISTRY, VECTOR_DB_DIR, LOGS_DIR,
    LLM_PROVIDER, GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL,
    API_KEY, LLM_BASE_URL,
)
from .retrieval.vector_store import get_collection, get_index_stats, reset_collection
from .retrieval.hybrid_search import build_bm25_index, load_bm25_index
from .retrieval.retriever import hybrid_search
from .generation.llm import generate_answer
from .generation.prompt import classify_query
from .generation.providers import build_provider_chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BM25_PATH = str(VECTOR_DB_DIR / "bm25_index.pkl")
_index_ready = False
_provider_chain_names = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index_ready, _provider_chain_names
    t_start = time.time()
    logger.info("Starting Diabetes RAG Backend...")

    # Phase 1: Load vector store + BM25
    try:
        t0 = time.time()
        _ = get_collection()
        loaded = load_bm25_index(BM25_PATH)
        if loaded:
            _index_ready = True
            logger.info(f"BM25 index loaded ({(time.time()-t0)*1000:.0f}ms)")
        else:
            logger.warning("BM25 index not found. Run build_index.py first.")
    except Exception as e:
        logger.warning(f"Index not ready: {e}")

    # Phase 2: Preload embedding model
    from .retrieval.embeddings import get_embedding_model
    try:
        t0 = time.time()
        get_embedding_model(EMBEDDING_MODEL)
        logger.info(f"Embedding model preloaded ({(time.time()-t0)*1000:.0f}ms)")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {e}")

    # Phase 3: Build provider chain and report config
    chain = build_provider_chain(
        gemini_api_key=GEMINI_API_KEY,
        gemini_model=GEMINI_MODEL,
        groq_api_key=GROQ_API_KEY,
        groq_model=GROQ_MODEL,
        groq_base_url=GROQ_BASE_URL,
        openrouter_api_key=API_KEY,
        openrouter_model=LLM_MODEL,
        openrouter_base_url=LLM_BASE_URL,
    )
    _provider_chain_names = [p.name for p in chain]

    elapsed_startup = (time.time() - t_start) * 1000
    logger.info("=" * 60)
    logger.info("STARTUP DIAGNOSTIC")
    logger.info(f"  Embedding model:   {EMBEDDING_MODEL}")
    logger.info(f"  Reranker enabled:  {RERANKER_ENABLED}")
    logger.info(f"  Index ready:       {_index_ready}")
    logger.info(f"  LLM provider mode: {LLM_PROVIDER}")
    logger.info(f"  LLM providers:     {' → '.join(_provider_chain_names) if _provider_chain_names else 'none configured'}")
    for p in chain:
        logger.info(f"    {p.name}: {p.is_configured}")
    logger.info(f"  Cold-start time:   {elapsed_startup:.0f}ms")
    logger.info("=" * 60)

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Diabetes Evidence Assistant",
    description="Evidence-grounded medical RAG for diabetes diagnosis and classification",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=TOP_K, ge=1, le=50)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=50)
    evidence_k: int = Field(default=5, ge=1, le=10)


class IndexRebuildRequest(BaseModel):
    confirm: bool = Field(default=False)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_ready": _index_ready,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "llm_provider_mode": LLM_PROVIDER,
        "providers": _provider_chain_names,
        "version": "2.0.0",
    }


@app.post("/api/search")
async def api_search(req: SearchRequest):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    try:
        result = hybrid_search(
            req.query,
            top_k=req.top_k,
            rerank_top_k=RERANK_TOP_K,
            enable_reranker=RERANKER_ENABLED,
        )
        elapsed = time.time() - t0

        return {
            "request_id": request_id,
            "query": req.query,
            "results": result["results"],
            "timings": result["timings"],
            "total_ms": round(elapsed * 1000, 1),
        }
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    request_id = str(uuid.uuid4())[:8]
    t0 = time.time()

    try:
        search_result = hybrid_search(
            req.message,
            top_k=req.top_k,
            rerank_top_k=RERANK_TOP_K,
            enable_reranker=RERANKER_ENABLED,
        )

        generation_result = generate_answer(
            req.message,
            search_result["results"],
            evidence_k=req.evidence_k,
        )

        elapsed = time.time() - t0

        return {
            "request_id": request_id,
            "answer": generation_result["answer"],
            "confidence": generation_result["confidence"],
            "grounded": generation_result["grounded"],
            "citations": generation_result.get("citations", []),
            "sources": generation_result["sources"],
            "evidence": generation_result["evidence"],
            "refused": generation_result["refused"],
            "refusal_reason": generation_result.get("refusal_reason"),
            "query_type": generation_result["query_type"],
            "safety": generation_result.get("safety", {}),
            "verification": generation_result.get("verification", {}),
            "timings": {
                **search_result.get("timings", {}),
                **generation_result.get("timings", {}),
            },
            "total_ms": round(elapsed * 1000, 1),
            "evidence_validation": generation_result.get("evidence_validation"),
        }
    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources")
async def api_sources():
    sources = []
    stats = get_index_stats()
    for sid, meta in SOURCE_REGISTRY.items():
        sources.append({
            **meta,
            "indexed_chunks": stats.get("total_chunks", 0) if stats else 0,
        })
    return {"sources": sources}


@app.get("/api/stats")
async def api_stats():
    try:
        stats = get_index_stats()
    except Exception:
        stats = {"total_chunks": 0}
    return {
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "providers": _provider_chain_names,
        "index_stats": stats,
        "source_count": len(SOURCE_REGISTRY),
        "version": "2.0.0",
    }


@app.post("/api/index/rebuild")
async def api_index_rebuild(req: IndexRebuildRequest):
    if not req.confirm:
        return {"status": "cancelled", "message": "Set confirm=true to rebuild"}

    try:
        reset_collection()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_index",
            str(VECTOR_DB_DIR.parent.parent / "scripts" / "build_index.py"),
        )
        build_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_mod)
        build_mod.build_full_index()
        global _index_ready
        _index_ready = True
        return {"status": "success", "message": "Index rebuilt successfully"}
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/evaluation")
async def api_evaluation():
    eval_path = LOGS_DIR / "evaluation_results.json"
    if eval_path.exists():
        import json
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "no_evaluation_run", "message": "Run evaluation first"}


@app.post("/api/evaluate")
async def api_evaluate():
    try:
        from .evaluation.evaluate import run_evaluation
        result = run_evaluation()
        return result
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
