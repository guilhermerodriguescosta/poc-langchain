"""Leitores locais: convertem bytes em texto antes de qualquer chamada à API."""

from io import BytesIO
from pathlib import Path

from docx import Document
from docx.table import Table
from pypdf import PdfReader

MAX_FILE_BYTES = 10_000_000
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


class DocumentError(ValueError):
    """Problema no documento que pode ser apresentado ao usuário."""


def extract_text(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentError("Formato não suportado. Anexe um arquivo TXT, PDF ou DOCX.")
    if not content:
        raise DocumentError("O arquivo está vazio.")
    if len(content) > MAX_FILE_BYTES:
        raise DocumentError("O arquivo excede o limite de 10 MB.")

    try:
        if extension == ".txt":
            text = content.decode("utf-8-sig")
        elif extension == ".pdf":
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise DocumentError("PDF protegido por senha não é suportado.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            document = Document(BytesIO(content))
            blocks = []
            for block in document.iter_inner_content():
                if isinstance(block, Table):
                    blocks.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                else:
                    blocks.append(block.text)
            text = "\n".join(blocks)
    except DocumentError:
        raise
    except UnicodeDecodeError:
        raise DocumentError("Não foi possível ler o TXT. Salve o arquivo em UTF-8.") from None
    except Exception:
        raise DocumentError("Não foi possível ler o documento. Verifique se está íntegro e no formato correto.") from None

    text = text.strip()
    if not text:
        raise DocumentError("O documento não contém texto extraível. PDFs digitalizados precisam de OCR, fora desta POC.")
    return text
