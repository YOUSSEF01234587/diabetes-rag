"""V4.1 Rebuild: Clean ingestion pipeline with table-aware extraction.

Rebuilds all chunks using:
- Text cleaning (URLs, noise removal)
- Table-aware extraction
- Proper chunk sizes from config
- Correct parent/child support
- Section router with true_section metadata
"""
import sys, os, io, json, time, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chromadb
from pathlib import Path
from backend.app.config import (
    DATA_DIR, VECTOR_DB_DIR, SOURCE_REGISTRY, SOURCE_BY_FILE,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS,
)
from backend.app.ingestion.parser import parse_pdf
from backend.app.ingestion.metadata import enrich_metadata
from backend.app.ingestion.cleaner import clean_for_retrieval, extract_source_metadata
from backend.app.ingestion.table_extractor import extract_tables_from_page
from backend.app.retrieval.page_sections import ADA_PAGE_SECTIONS, NIDDK_PAGE_SECTIONS


def estimate_tokens(text):
    return max(1, len(text) // 4)


def make_chunk_id(source_id, page, idx):
    raw = f"{source_id}_p{page}_c{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def split_text_into_chunks(text, max_tokens, overlap_tokens):
    """Split text into chunks by word boundaries."""
    words = text.split()
    chunks = []
    current_words = []
    current_tokens = 0

    for word in words:
        word_tokens = max(1, len(word) // 4)
        # Use estimate_tokens for consistency with the rest of the pipeline
        if estimate_tokens(" ".join(current_words + [word])) > max_tokens and current_words:
            chunks.append(" ".join(current_words))
            # Overlap
            overlap_words = []
            overlap_count = 0
            for w in reversed(current_words):
                wt = max(1, len(w) // 4)
                if overlap_count + wt > overlap_tokens:
                    break
                overlap_words.insert(0, w)
                overlap_count += wt
            current_words = overlap_words
            current_tokens = overlap_count
        current_words.append(word)
        current_tokens += word_tokens

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def get_true_section(page_pdf, source_id):
    if "niddk" in source_id.lower():
        return NIDDK_PAGE_SECTIONS.get(page_pdf, ("Comparing Diabetes Blood Tests", ""))[0]
    return ADA_PAGE_SECTIONS.get(page_pdf, ("Unknown", ""))[0]


def rebuild_all():
    """Full rebuild: parse -> clean -> extract tables -> chunk -> index."""
    t_start = time.time()

    all_chunks = []
    stats = {"pages": 0, "chunks": 0, "table_chunks": 0, "text_chunks": 0,
             "urls_removed": 0, "dois_extracted": 0}

    # Process each PDF
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        source_meta = SOURCE_BY_FILE.get(pdf_path.name)
        if not source_meta:
            print(f"  Skipping {pdf_path.name} (no registry)")
            continue

        source_id = source_meta["source_id"]
        print(f"\nProcessing: {pdf_path.name} ({source_id})")

        pages = parse_pdf(pdf_path)
        pages = enrich_metadata(pages)

        for page in pages:
            page_pdf = page["page_pdf"]
            text = page["text"]
            structured_lines = page.get("structured_lines", [])

            # Get true section
            true_section = get_true_section(page_pdf, source_id)

            # Skip reference pages (use true_section, not metadata section)
            if true_section == "References":
                continue

            # --- CLEAN TEXT ---
            cleaned = clean_for_retrieval(text)
            retrieval_text = cleaned["retrieval_text"]
            display_text = cleaned["display_text"]
            stats["urls_removed"] += len(cleaned["removed_urls"])
            stats["dois_extracted"] += len(cleaned["removed_dois"])

            # --- EXTRACT TABLES ---
            table_chunks = extract_tables_from_page(
                text, structured_lines,
                source_id=source_id,
                page_pdf=page_pdf,
                section=true_section,
                subsection=page.get("subsection", ""),
            )

            for tc in table_chunks:
                tc["chunk_id"] = make_chunk_id(source_id, page_pdf, len(all_chunks))
                tc["page_document"] = page.get("page_document", page_pdf)
                tc["source_title"] = source_meta.get("title", "")
                tc["short_title"] = source_meta.get("short_title", "")
                tc["organization"] = source_meta.get("organization", "")
                tc["doi"] = source_meta.get("doi", "")
                tc["official_url"] = source_meta.get("official_url", "")
                tc["year"] = source_meta.get("year", 0)
                tc["authority"] = source_meta.get("authority", "high")
                tc["pdf_file"] = pdf_path.name
                tc["token_estimate"] = estimate_tokens(tc["text"])
                all_chunks.append(tc)
                stats["table_chunks"] += 1

            # --- CHUNK CLEANED TEXT ---
            # Determine chunk size (NIDDK gets smaller chunks)
            if "niddk" in source_id.lower():
                chunk_max = 300
                overlap = 0
            else:
                chunk_max = CHUNK_SIZE_TOKENS
                overlap = CHUNK_OVERLAP_TOKENS

            # Split by paragraphs
            paragraphs = retrieval_text.split('\n\n')
            # Filter very short paragraphs
            paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

            if not paragraphs:
                # Fallback: split by single newlines
                paragraphs = retrieval_text.split('\n')
                paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

            if not paragraphs and len(retrieval_text) > 50:
                paragraphs = [retrieval_text]

            # Build chunks from paragraphs
            current_text = ""
            current_tokens = 0
            chunk_idx = len(all_chunks)

            for para in paragraphs:
                para_tokens = estimate_tokens(para)

                # Use 90% of chunk_max as flush threshold to account for token estimation errors
                flush_threshold = int(chunk_max * 0.9)

                if current_tokens + para_tokens > flush_threshold and current_text:
                    # Flush current chunk
                    chunk_id = make_chunk_id(source_id, page_pdf, chunk_idx)
                    chunk = {
                        "chunk_id": chunk_id,
                        "text": current_text.strip(),
                        "retrieval_text": current_text.strip(),
                        "display_text": current_text.strip(),
                        "source_id": source_id,
                        "source_title": source_meta.get("title", ""),
                        "short_title": source_meta.get("short_title", ""),
                        "organization": source_meta.get("organization", ""),
                        "page_pdf": page_pdf,
                        "page_document": page.get("page_document", page_pdf),
                        "section": true_section,
                        "subsection": page.get("subsection", ""),
                        "doi": source_meta.get("doi", ""),
                        "official_url": source_meta.get("official_url", ""),
                        "year": source_meta.get("year", 0),
                        "authority": source_meta.get("authority", "high"),
                        "pdf_file": pdf_path.name,
                        "has_table": page.get("has_table_keywords", False),
                        "is_reference": False,
                        "token_estimate": estimate_tokens(current_text),
                        "is_parent": False,
                    }
                    all_chunks.append(chunk)
                    chunk_idx += 1
                    stats["text_chunks"] += 1

                    # Overlap
                    if overlap > 0:
                        words = current_text.split()
                        overlap_words = []
                        ocount = 0
                        for w in reversed(words):
                            wt = max(1, len(w) // 4)
                            if ocount + wt > overlap:
                                break
                            overlap_words.insert(0, w)
                            ocount += wt
                        current_text = " ".join(overlap_words)
                        current_tokens = ocount
                    else:
                        current_text = ""
                        current_tokens = 0

                # If single paragraph is too big, split it
                if para_tokens > chunk_max:
                    if current_text:
                        chunk_id = make_chunk_id(source_id, page_pdf, chunk_idx)
                        chunk = {
                            "chunk_id": chunk_id,
                            "text": current_text.strip(),
                            "retrieval_text": current_text.strip(),
                            "display_text": current_text.strip(),
                            "source_id": source_id,
                            "source_title": source_meta.get("title", ""),
                            "short_title": source_meta.get("short_title", ""),
                            "organization": source_meta.get("organization", ""),
                            "page_pdf": page_pdf,
                            "page_document": page.get("page_document", page_pdf),
                            "section": true_section,
                            "subsection": page.get("subsection", ""),
                            "doi": source_meta.get("doi", ""),
                            "official_url": source_meta.get("official_url", ""),
                            "year": source_meta.get("year", 0),
                            "authority": source_meta.get("authority", "high"),
                            "pdf_file": pdf_path.name,
                            "has_table": page.get("has_table_keywords", False),
                            "is_reference": False,
                            "token_estimate": estimate_tokens(current_text),
                            "is_parent": False,
                        }
                        all_chunks.append(chunk)
                        chunk_idx += 1
                        stats["text_chunks"] += 1
                        current_text = ""
                        current_tokens = 0

                    # Split the big paragraph
                    sub_chunks = split_text_into_chunks(para, chunk_max, overlap)
                    for sc in sub_chunks:
                        chunk_id = make_chunk_id(source_id, page_pdf, chunk_idx)
                        chunk = {
                            "chunk_id": chunk_id,
                            "text": sc.strip(),
                            "retrieval_text": sc.strip(),
                            "display_text": sc.strip(),
                            "source_id": source_id,
                            "source_title": source_meta.get("title", ""),
                            "short_title": source_meta.get("short_title", ""),
                            "organization": source_meta.get("organization", ""),
                            "page_pdf": page_pdf,
                            "page_document": page.get("page_document", page_pdf),
                            "section": true_section,
                            "subsection": page.get("subsection", ""),
                            "doi": source_meta.get("doi", ""),
                            "official_url": source_meta.get("official_url", ""),
                            "year": source_meta.get("year", 0),
                            "authority": source_meta.get("authority", "high"),
                            "pdf_file": pdf_path.name,
                            "has_table": page.get("has_table_keywords", False),
                            "is_reference": False,
                            "token_estimate": estimate_tokens(sc),
                            "is_parent": False,
                        }
                        all_chunks.append(chunk)
                        chunk_idx += 1
                        stats["text_chunks"] += 1
                else:
                    if current_text:
                        current_text += "\n\n" + para
                    else:
                        current_text = para
                    current_tokens = estimate_tokens(current_text)

            # Flush remaining
            if current_text.strip():
                chunk_id = make_chunk_id(source_id, page_pdf, chunk_idx)
                chunk = {
                    "chunk_id": chunk_id,
                    "text": current_text.strip(),
                    "retrieval_text": current_text.strip(),
                    "display_text": current_text.strip(),
                    "source_id": source_id,
                    "source_title": source_meta.get("title", ""),
                    "short_title": source_meta.get("short_title", ""),
                    "organization": source_meta.get("organization", ""),
                    "page_pdf": page_pdf,
                    "page_document": page.get("page_document", page_pdf),
                    "section": true_section,
                    "subsection": page.get("subsection", ""),
                    "doi": source_meta.get("doi", ""),
                    "official_url": source_meta.get("official_url", ""),
                    "year": source_meta.get("year", 0),
                    "authority": source_meta.get("authority", "high"),
                    "pdf_file": pdf_path.name,
                    "has_table": page.get("has_table_keywords", False),
                    "is_reference": False,
                    "token_estimate": estimate_tokens(current_text),
                    "is_parent": False,
                }
                all_chunks.append(chunk)
                chunk_idx += 1
                stats["text_chunks"] += 1

            stats["pages"] += 1
            if (stats["pages"] % 5 == 0):
                print(f"  Pages processed: {stats['pages']}")

    stats["chunks"] = len(all_chunks)

    # --- POST-PROCESSING: clean, filter, dedup ---

    cleaned_chunks = []
    noise_filtered = 0
    for c in all_chunks:
        # Clean the text
        cleaned = clean_for_retrieval(c["text"])
        clean_text = cleaned["retrieval_text"].strip()

        # Filter: minimum 50 tokens after cleaning
        if estimate_tokens(clean_text) < 50:
            noise_filtered += 1
            continue

        # Update text fields
        c["text"] = clean_text
        c["retrieval_text"] = clean_text
        c["display_text"] = clean_text
        c["token_estimate"] = estimate_tokens(clean_text)
        cleaned_chunks.append(c)

    # Dedup by text hash
    seen_texts = set()
    deduped = []
    dedup_count = 0
    for c in cleaned_chunks:
        text_hash = hashlib.md5(c["text"].encode()).hexdigest()[:16]
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            deduped.append(c)
        else:
            dedup_count += 1

    all_chunks = deduped
    stats["chunks"] = len(all_chunks)
    stats["noise_filtered"] = noise_filtered
    stats["deduped"] = dedup_count

    print(f"\nPost-processing: {noise_filtered} noise chunks filtered, {dedup_count} duplicates removed")
    print(f"Final chunks: {stats['chunks']} ({stats['text_chunks']} text + {stats['table_chunks']} table)")

    # --- INDEX ---
    print(f"\nTotal chunks: {stats['chunks']} ({stats['text_chunks']} text + {stats['table_chunks']} table)")
    print(f"URLs removed: {stats['urls_removed']}")
    print(f"DOIs extracted: {stats['dois_extracted']}")

    # Save chunk data
    chunk_data_path = Path("data/processed/chunks_v41.json")
    chunk_data_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert for JSON serialization
    json_chunks = []
    for c in all_chunks:
        jc = {k: v for k, v in c.items()}
        json_chunks.append(jc)

    with open(chunk_data_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": json_chunks, "stats": stats}, f, indent=2, ensure_ascii=False)
    print(f"Chunk data saved to {chunk_data_path}")

    # Index in ChromaDB
    print("\nIndexing in ChromaDB...")

    # Pre-compute embeddings to avoid ChromaDB's ONNX OOM
    print("  Pre-computing embeddings...")
    from backend.app.retrieval.embeddings import embed_texts
    all_texts = [c["text"] for c in all_chunks]
    all_embeddings = embed_texts(all_texts, batch_size=32)
    print(f"  Total embeddings computed: {len(all_embeddings)}")

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    # Delete old collection
    try:
        client.delete_collection("diabetes_rag")
        print("  Deleted old collection")
    except Exception:
        pass

    collection = client.create_collection(
        "diabetes_rag",
        metadata={"hnsw:space": "cosine"}
    )

    # Batch insert
    batch_size = 20
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = []
        for c in batch:
            meta = {}
            for k, v in c.items():
                if k == "text":
                    continue
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif isinstance(v, list):
                    meta[k] = json.dumps(v)
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=all_embeddings[i:i+batch_size],
            metadatas=metadatas,
        )
        print(f"  Indexed batch {i//batch_size + 1}: {len(batch)} chunks")

    print(f"\nTotal indexed: {collection.count()} chunks")

    # Build BM25 index
    print("\nBuilding BM25 index...")
    from backend.app.retrieval.hybrid_search import build_bm25_index
    bm25_path = str(VECTOR_DB_DIR / "bm25_index.pkl")
    build_bm25_index(all_chunks, bm25_path)
    print(f"BM25 index saved to {bm25_path}")

    # Generate chunk report
    from collections import Counter
    sections = Counter(c["section"] for c in all_chunks)
    sources = Counter(c["source_id"] for c in all_chunks)
    token_stats = [c["token_estimate"] for c in all_chunks]

    report = {
        "total_chunks": len(all_chunks),
        "text_chunks": stats["text_chunks"],
        "table_chunks": stats["table_chunks"],
        "urls_removed": stats["urls_removed"],
        "dois_extracted": stats["dois_extracted"],
        "sections": dict(sections.most_common()),
        "sources": dict(sources.most_common()),
        "token_stats": {
            "min": min(token_stats) if token_stats else 0,
            "max": max(token_stats) if token_stats else 0,
            "avg": round(sum(token_stats) / len(token_stats), 1) if token_stats else 0,
        },
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open("logs/v41_chunk_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nRebuild complete in {elapsed:.1f}s")
    print(f"Chunks: {len(all_chunks)} ({stats['text_chunks']} text + {stats['table_chunks']} table)")
    print(f"Sources: {dict(sources)}")
    print(f"Sections: {dict(sections.most_common())}")

    return all_chunks, stats


if __name__ == "__main__":
    rebuild_all()
