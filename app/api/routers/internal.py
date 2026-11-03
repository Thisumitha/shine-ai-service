from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile

from app.container import Container

router = APIRouter(prefix="/internal/index", tags=["internal"])


def get_container(request: Request) -> Container:
    return request.app.state.container


def check_secret(x_internal_secret: str = Header(default=""), container: Container = Depends(get_container)) -> None:
    secret = container.settings.internal_secret
    if not secret or x_internal_secret != secret:
        raise HTTPException(401, "Wrong or missing X-Internal-Secret")


@router.post("", status_code=202, dependencies=[Depends(check_secret)])
async def add_pdf(
    file: UploadFile = File(...),
    item_id: int = Form(..., alias="itemId"),
    course_id: int = Form(..., alias="courseId"),
    topic_id: int | None = Form(None, alias="topicId"),
    title: str = Form(""),
    container: Container = Depends(get_container),
) -> dict:
    """Spring sends the PDF and its details. Save the file, add it to the queue, reply at once."""
    max_bytes = container.settings.max_pdf_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"PDF is bigger than {container.settings.max_pdf_mb} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a PDF")

    pdf_path = Path(container.settings.pdf_dir) / f"{item_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(data)

    return container.job_store.add(item_id, course_id, topic_id, title or file.filename, str(pdf_path))


@router.get("/{item_id}", dependencies=[Depends(check_secret)])
def job_status(item_id: int, container: Container = Depends(get_container)) -> dict:
    job = container.job_store.latest(item_id)
    if job is None:
        raise HTTPException(404, "No job for this item")
    return job


@router.delete("/{item_id}", status_code=204, dependencies=[Depends(check_secret)])
def delete_pdf(item_id: int, container: Container = Depends(get_container)) -> None:
    for path in container.job_store.remove(item_id):
        Path(path).unlink(missing_ok=True)
