"""Diabetes RAG Configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
LOGS_DIR = BASE_DIR / "logs"

for d in [PROCESSED_DIR, VECTOR_DB_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "700"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

TOP_K = int(os.getenv("TOP_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "15"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").lower() == "true"

# LLM provider selection: "auto" = chain, or explicit "gemini" / "groq" / "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b:free")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# OpenRouter (stored as OPENAI_API_KEY for backward compat)
API_KEY = os.getenv("OPENAI_API_KEY", "")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Groq (OpenAI-compatible)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

SOURCE_REGISTRY = {
    "ada_soc_2026_diagnosis": {
        "source_id": "ada_soc_2026_diagnosis",
        "organization": "American Diabetes Association",
        "authority": "high",
        "document_type": "clinical_practice_guidelines",
        "title": "2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes-2026",
        "short_title": "ADA Standards of Care 2026 - Diagnosis",
        "journal": "Diabetes Care",
        "year": 2026,
        "volume": "49",
        "supplement": "Supplement_1",
        "pages": "S27-S49",
        "doi": "10.2337/dc26-S002",
        "official_url": "https://diabetesjournals.org/care/article/49/Supplement_1/S27/163926/2-Diagnosis-and-Classification-of-Diabetes",
        "pdf_file": "ADA_Standards_of_Care_2026_Diagnosis.pdf",
        "page_offset": 0,
        "printed_page_start": "S27",
        "copyright": "2025 by the American Diabetes Association. Educational, noncommercial use with proper citation.",
        "usage_notes": "Do not reproduce large portions. Paraphrase and cite. Link to official source.",
    },
    "niddk_diabetes_prediabetes_tests": {
        "source_id": "niddk_diabetes_prediabetes_tests",
        "organization": "National Institute of Diabetes and Digestive and Kidney Diseases",
        "parent_organization": "National Institutes of Health",
        "authority": "high",
        "document_type": "clinical_information_page",
        "title": "Diabetes & Prediabetes Tests",
        "short_title": "NIDDK Diabetes & Prediabetes Tests",
        "year": 2020,
        "doi": None,
        "official_url": "https://www.niddk.nih.gov/health-information/professionals/clinical-tools-patient-management/diabetes/diabetes-prediabetes",
        "pdf_file": "NIDDK_Diabetes_Prediabetes_Tests.pdf",
        "page_offset": 0,
        "copyright": "Public domain (NIH). Proper attribution required.",
        "usage_notes": "Official NIH source. Attribute clearly.",
    },
}

SOURCE_BY_FILE = {v["pdf_file"]: v for v in SOURCE_REGISTRY.values()}
