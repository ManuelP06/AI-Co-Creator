import os, uuid, shutil
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from app import models, schemas
from app.config import settings

UPLOAD_DIR = settings.upload_directory

async def handle_upload(username: str, file: UploadFile, db: Session):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        user = models.User(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)

    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    file_size = os.path.getsize(file_path)

    video = models.Video(
        user_id=user.id,
        file_path=file_path,
        original_filename=file.filename,
        file_size=file_size
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    return schemas.VideoUploadResponse(
        video_id=video.id,
        filename=video.original_filename,
        file_path=video.file_path,
        file_size=video.file_size
    )


def list_videos(username: str, db: Session):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    videos = db.query(models.Video).filter(models.Video.user_id == user.id).all()
    return [
        schemas.VideoUploadResponse(
            video_id=v.id,
            filename=v.original_filename,
            file_path=v.file_path,
            file_size=v.file_size
        )
        for v in videos
    ]
