from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
from docx import Document

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT_CHARS = 60_000
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}


def extract_file(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError("Supported formats: PDF, DOCX, TXT, MD, CSV, and JSON.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File is larger than 10 MB.")
    if suffix == ".pdf":
        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
    elif suffix == ".docx":
        document = Document(BytesIO(data))
        text = "\n".join(p.text for p in document.paragraphs)
    else:
        text = data.decode("utf-8", errors="replace")
    text = text.strip()
    if not text:
        raise ValueError("No readable text was found in this file.")
    return text[:MAX_TEXT_CHARS]
