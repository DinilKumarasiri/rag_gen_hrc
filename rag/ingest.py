"""Turns an uploaded file's raw bytes into plain text, regardless of format."""
import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


class UnsupportedFileTypeError(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


def extract_text(filename: str, file_bytes: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext or '(none)'}'. Supported types: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    if ext in (".txt", ".md"):
        text = file_bytes.decode("utf-8", errors="ignore")
    elif ext == ".pdf":
        text = _extract_pdf(filename, file_bytes)
    else:  # .docx
        text = _extract_docx(filename, file_bytes)

    text = text.strip()
    if not text:
        raise EmptyDocumentError(
            f"'{filename}' has no extractable text — it may be empty, "
            "image-only, or corrupted."
        )
    return text


def _extract_pdf(filename: str, file_bytes: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfReadError as exc:
        raise EmptyDocumentError(f"'{filename}' could not be parsed as a PDF.") from exc


def _extract_docx(filename: str, file_bytes: bytes) -> str:
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except PackageNotFoundError as exc:
        raise EmptyDocumentError(
            f"'{filename}' could not be parsed as a DOCX file."
        ) from exc
