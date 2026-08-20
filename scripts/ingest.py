"""Ingest PDFs and produce chunk reports."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from backend.app.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, LOGS_DIR
from backend.app.ingestion.loader import load_documents
from backend.app.ingestion.metadata import enrich_metadata
from backend.app.ingestion.chunker import chunk_documents, generate_chunk_report


def main():
    print("=" * 60)
    print("DIABETES RAG - DOCUMENT INGESTION")
    print("=" * 60)

    print("\n[1/4] Loading documents...")
    pages = load_documents()
    print(f"  Total pages: {len(pages)}")

    print("\n[2/4] Enriching metadata...")
    pages = enrich_metadata(pages)

    print("\n[3/4] Chunking...")
    chunks = chunk_documents(pages, max_tokens=CHUNK_SIZE_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS)
    print(f"  Total chunks: {len(chunks)}")

    print("\n[4/4] Generating report...")
    report = generate_chunk_report(chunks, str(LOGS_DIR / "chunk_report.json"))

    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"  Total chunks: {report['total_chunks']}")
    print(f"  Chunks per document: {report['chunks_per_document']}")
    print(f"  Pages per document: {report['pages_per_document']}")
    print(f"  Avg tokens/chunk: {report['avg_token_length']:.0f}")
    print(f"  Empty chunks: {report['empty_chunks']}")
    print(f"  Table chunks: {report['table_chunks']}")
    print(f"  Sections: {len(report['sections'])}")
    print(f"{'='*60}")

    processed_dir = LOGS_DIR.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = processed_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nChunks saved to: {chunks_path}")

    return report


if __name__ == "__main__":
    main()
