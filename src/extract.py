"""Bounded, in-memory extraction; no OCR, external calls, or resume logging."""

from io import BytesIO
import logging
from pathlib import PurePath
import zipfile
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader, apply_configuration

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARACTERS = 50_000
MAX_EXPANDED_BYTES = 20 * 1024 * 1024
NO_TEXT = "No readable text found. Scanned/image PDFs are not supported."

# Parser diagnostics can contain fragments of malformed uploaded documents.
# Prevent third-party PDF logging from exposing those fragments.
pdf_logger = logging.getLogger("pypdf")
pdf_logger.handlers = [logging.NullHandler()]
pdf_logger.propagate = False


class FileValidationError(ValueError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_extension(filename: str) -> str:
    extension = PurePath(filename).suffix.lower()
    if extension not in {".pdf", ".docx", ".txt"}:
        raise FileValidationError("Only PDF, DOCX, and TXT files are supported.", 415)
    return extension


def _bounded_text(parts) -> str:
    output: list[str] = []
    length = 0
    for part in parts:
        length += len(part) + (1 if output else 0)
        if length > MAX_TEXT_CHARACTERS:
            raise FileValidationError(
                "Extracted text exceeds 50,000 characters. Please upload a shorter CV."
            )
        output.append(part)
    return "\n".join(output).strip()


def _docx_parts(document):
    # Preserve body paragraph/table order, including text in nested tables.
    for block in document.iter_inner_content():
        if isinstance(block, Paragraph):
            yield block.text
        elif isinstance(block, Table):
            for row in block.rows:
                seen = set()
                for cell in row.cells:
                    if cell._tc not in seen:  # Merged cells should not repeat text.
                        seen.add(cell._tc)
                        yield from _docx_parts(cell)


def extract_text(filename: str, content: bytes) -> str:
    extension = validate_extension(filename)
    if len(content) > MAX_FILE_BYTES:
        raise FileValidationError("File exceeds the 5 MB (5,242,880 bytes) limit.", 413)
    if not content:
        raise FileValidationError(NO_TEXT)
    try:
        if extension == ".pdf":
            if not content.startswith(b"%PDF"):
                raise FileValidationError(
                    "The file does not have a valid PDF signature.", 415
                )
            with apply_configuration(zlib_maximum_output_length=MAX_EXPANDED_BYTES):
                reader = PdfReader(BytesIO(content), strict=True)
                if reader.is_encrypted:
                    raise FileValidationError(
                        "Password-protected PDFs are not supported. Upload an unlocked copy."
                    )
                if len(reader.pages) > 100:
                    raise FileValidationError("PDFs must contain at most 100 pages.")
                text = _bounded_text(page.extract_text() or "" for page in reader.pages)
        elif extension == ".docx":
            if not content.startswith(b"PK\x03\x04") or not zipfile.is_zipfile(
                BytesIO(content)
            ):
                raise FileValidationError(
                    "The file does not have a valid DOCX ZIP signature.", 415
                )
            with zipfile.ZipFile(BytesIO(content)) as archive:
                entries = archive.infolist()
                if "word/document.xml" not in archive.namelist():
                    raise FileValidationError(
                        "The DOCX archive is missing word/document.xml.", 415
                    )
                if (
                    len(entries) > 500
                    or sum(e.file_size for e in entries) > MAX_EXPANDED_BYTES
                ):
                    raise FileValidationError(
                        "The DOCX archive expands beyond the supported size limit.", 413
                    )
            text = _bounded_text(_docx_parts(Document(BytesIO(content))))
        else:
            # TXT has no universal magic signature. Require decodable plain text,
            # reject known binary formats and control bytes (no replacement decode).
            if content.startswith(
                (
                    b"%PDF",
                    b"PK\x03\x04",
                    b"\x89PNG",
                    b"\xff\xd8",
                    b"MZ",
                    b"\xd0\xcf\x11\xe0",
                )
            ):
                raise FileValidationError(
                    "The TXT file contains a different file format.", 415
                )
            if content.startswith((b"\xff\xfe", b"\xfe\xff")):
                text = content.decode("utf-16")
            else:
                try:
                    text = content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = content.decode("cp1252")
            if any(
                (ord(c) < 32 and c not in "\r\n\t") or 127 <= ord(c) < 160 for c in text
            ):
                raise FileValidationError(
                    "The TXT file must contain plain readable text, not binary data.",
                    415,
                )
            text = _bounded_text([text])
    except FileValidationError:
        raise
    except Exception:
        # Do not reflect parser exceptions or document fragments to logs/users.
        raise FileValidationError(
            "The file could not be read. It may be damaged or unsupported."
        ) from None
    if len(text.split()) < 3 or sum(c.isalpha() for c in text) < 10:
        raise FileValidationError(NO_TEXT)
    return text
