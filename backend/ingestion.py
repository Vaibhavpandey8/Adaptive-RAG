"""
LangChain Document Ingestion & Chunking Pipeline.
Uses pypdf + LangChain Document + RecursiveCharacterTextSplitter.
Loads ONLY user-uploaded files from data/uploads/.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, UPLOADS_DIR


def extract_text_from_file(file_path: Path) -> Tuple[str, str]:
    """Extracts text and title from PDF, Markdown, or TXT files."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[Page {idx}]\n{text}")
        full_text = "\n\n".join(pages)
        title = file_path.stem.replace("_", " ").replace("-", " ").title()
        return full_text, title

    full_text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
    title_match = re.search(r"^#\s+(.+)$", full_text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else file_path.stem.replace("_", " ").title()
    return full_text, title


def split_text_to_documents(
    text: str,
    source_name: str,
    title: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits raw text into LangChain Document chunks using RecursiveCharacterTextSplitter,
    preserving source, title, section, and chunk_id metadata.
    """
    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    raw_doc = Document(
        page_content=text,
        metadata={"source": source_name, "title": title},
    )
    split_docs = splitter.split_documents([raw_doc])

    current_section = "Overview"
    enriched: List[Document] = []
    for idx, doc in enumerate(split_docs, start=1):
        heading = re.search(r"^#{1,3}\s+(.+)$", doc.page_content, flags=re.MULTILINE)
        if heading:
            current_section = heading.group(1).strip()

        doc.metadata = {
            **doc.metadata,
            "chunk_id": f"{source_name}::c{idx}",
            "source": source_name,
            "title": title,
            "section": current_section,
        }
        enriched.append(doc)

    return enriched


def load_all_documents() -> Tuple[List[Document], List[Dict], str]:
    """
    Loads ONLY user-uploaded documents from data/uploads/ into LangChain Documents.
    Returns (documents, manifest, corpus_hash).
    """
    supported_ext = {".pdf", ".md", ".txt"}
    all_docs: List[Document] = []
    manifest: List[Dict] = []
    hasher = hashlib.sha256()

    if UPLOADS_DIR.exists():
        for fp in sorted(UPLOADS_DIR.iterdir()):
            if not (fp.is_file() and fp.suffix.lower() in supported_ext):
                continue

            text, title = extract_text_from_file(fp)
            doc_chunks = split_text_to_documents(text, source_name=fp.name, title=title)
            sections = list(dict.fromkeys(d.metadata["section"] for d in doc_chunks))

            hasher.update(fp.name.encode("utf-8"))
            hasher.update(text.encode("utf-8", errors="ignore"))

            all_docs.extend(doc_chunks)
            manifest.append({
                "filename": fp.name,
                "title": title,
                "is_sample": False,
                "chunk_count": len(doc_chunks),
                "char_count": len(text),
                "sections": sections[:6],
            })

    return all_docs, manifest, hasher.hexdigest()
