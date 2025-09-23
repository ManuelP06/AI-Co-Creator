import os
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app import schemas
from app.deps import get_db
from app.models import Video
from app.services import upload

router = APIRouter(prefix="/upload", tags=["videos"])


@router.post("/", response_model=schemas.VideoUploadResponse)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await upload.handle_upload(file, db)


@router.get("/", response_model=list[schemas.VideoUploadResponse])
def list_videos(db: Session = Depends(get_db)):
    return upload.list_videos(db)


@router.delete("/{video_id}")
def delete_video(video_id: int, db: Session = Depends(get_db)):
    """Delete a video and all associated data"""
    # Get video from database
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        # Delete physical file if it exists
        if video.file_path and os.path.exists(video.file_path):
            os.remove(video.file_path)

        # Delete video from database (this will cascade delete related data)
        db.delete(video)
        db.commit()

        return {"message": f"Video {video_id} deleted successfully", "video_id": video_id}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")
