"""Split extracted PDF pages into overlapping text chunks."""

from typing import TypedDict

from src.pdf_loader import PDFPage


DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150


class TextChunk(TypedDict):
    """A text chunk with its source page metadata."""

    chunk_id: int
    text: str
    page_number: int


def chunk_pages(
    pages: list[PDFPage],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split each page into chunks without crossing page boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1.")

    chunks: list[TextChunk] = []
    next_chunk_id = 1

    for page in pages:
        text = " ".join(page["text"].split())
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer ending at whitespace so words are not split unnecessarily.
            if end < len(text):
                whitespace = text.rfind(" ", start, end)
                if whitespace > start:
                    end = whitespace

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": next_chunk_id,
                        "text": chunk_text,
                        "page_number": page["page_number"],
                    }
                )
                next_chunk_id += 1

            if end >= len(text):
                break

            start = max(end - chunk_overlap, start + 1)

    return chunks
