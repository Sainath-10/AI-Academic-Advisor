"""Document chunking.

``RecursiveCharacterTextSplitter`` splits each PDF page on paragraph, line and
sentence boundaries, so a rule is rarely cut in the middle. Splitting happens per
page, which keeps the ``page`` metadata exact, and every chunk keeps the source
file name, the page number and the detected section title.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try paragraph breaks first, then lines, then sentences, then words.
SEPARATORS = ["\n\n", "\n", ". ", " "]


def split_page(
    text: str,
    source: str,
    page: int,
    section: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split one page of text into overlapping, metadata-rich chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
    )

    chunks: list[dict[str, Any]] = []
    for piece in splitter.split_text(text):
        cleaned = piece.strip()
        if not cleaned:
            continue
        chunks.append(
            {
                "text": cleaned,
                "metadata": {
                    "source": source,
                    "page": int(page),
                    "section": section,
                    "chunk_index": len(chunks),
                    "doc_type": "regulation",
                },
            }
        )
    return chunks
