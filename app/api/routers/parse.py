import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.routers.internal import check_secret, get_container
from app.container import Container

router = APIRouter(prefix="/internal/parse-pdf", tags=["internal"])


@router.post("", dependencies=[Depends(check_secret)])
async def parse_pdf(file: UploadFile = File(...), container: Container = Depends(get_container)) -> dict:
    """Read a PDF and return its pages and tables as JSON, without queueing or saving. For manual checks."""
    max_bytes = container.settings.max_pdf_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"PDF is bigger than {container.settings.max_pdf_mb} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a PDF")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        path = Path(folder) / "upload.pdf"
        path.write_bytes(data)
        try:
            document = container.ingestion.parse(str(path))
        except Exception as error:
            raise HTTPException(422, f"Could not read this PDF: {error}") from error

    return {
        "filename": file.filename,
        "page_count": len(document.pages),
        "pages": [
            {"page": p.page_number, "text": p.text, "tables": p.tables_markdown, "used_ocr": p.used_ocr}
            for p in document.pages
        ],
    }
