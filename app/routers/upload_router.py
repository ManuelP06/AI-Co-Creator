from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from app import schemas
from app.deps import get_db
from app.services import upload

router = APIRouter(
    prefix="/upload",
    tags=["videos"]
)

@router.post("/", response_model=schemas.VideoUploadResponse)
async def upload_video(
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    return await upload.handle_upload(username, file, db)


@router.get("/{username}", response_model=list[schemas.VideoUploadResponse])
def list_user_videos(username: str, db: Session = Depends(get_db)):
    return upload.list_videos(username, db)
