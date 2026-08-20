"""Document loader - loads PDFs and applies source metadata."""
import logging
from pathlib import Path
from ..config import DATA_DIR, SOURCE_BY_FILE
from .parser import parse_pdf

logger = logging.getLogger(__name__)


def load_documents() -> list[dict]:
    """Load all PDFs and return enriched page data."""
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    all_documents = []

    for pdf_path in pdf_files:
        source_meta = SOURCE_BY_FILE.get(pdf_path.name)
        if not source_meta:
            logger.warning("No source registry for %s, skipping.", pdf_path.name)
            continue

        logger.info("Loading: %s", pdf_path.name)
        pages = parse_pdf(pdf_path)

        for page in pages:
            page["source_id"] = source_meta["source_id"]
            page["source_title"] = source_meta["title"]
            page["short_title"] = source_meta["short_title"]
            page["organization"] = source_meta["organization"]
            page["document_type"] = source_meta["document_type"]
            page["doi"] = source_meta.get("doi")
            page["official_url"] = source_meta.get("official_url")
            page["year"] = source_meta.get("year")
            page["authority"] = source_meta.get("authority", "high")
            page["pdf_file"] = pdf_path.name

            pdf_page = page["page_pdf"]
            offset = source_meta.get("page_offset", 0)
            page["page_document"] = pdf_page + offset

        all_documents.extend(pages)
        logger.info("  Loaded %d pages", len(pages))

    return all_documents
