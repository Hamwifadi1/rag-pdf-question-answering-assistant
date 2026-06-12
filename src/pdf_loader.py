"""Extract text from PDF files while preserving page numbers."""

from typing import BinaryIO, TypedDict

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFPage(TypedDict):
    """Text and metadata extracted from one PDF page."""

    page_number: int
    text: str


class PDFExtractionError(Exception):
    """Raised when an uploaded PDF cannot be read."""


def extract_pdf_pages(pdf_file: BinaryIO) -> list[PDFPage]:
    """Return extracted page text with one-based page numbers."""
    try:
        pdf_file.seek(0)
        reader = PdfReader(pdf_file)

        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception as error:
                raise PDFExtractionError(
                    "The PDF is encrypted and cannot be opened."
                ) from error
            if not unlocked:
                raise PDFExtractionError(
                    "The PDF is password-protected and cannot be opened."
                )

        pages: list[PDFPage] = []
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(
                {
                    "page_number": page_number,
                    "text": page.extract_text() or "",
                }
            )

        if not pages:
            raise PDFExtractionError("The uploaded PDF contains no pages.")

        return pages
    except PDFExtractionError:
        raise
    except (PdfReadError, OSError, ValueError) as error:
        raise PDFExtractionError(
            "Could not read the uploaded PDF. Please try a valid PDF file."
        ) from error
