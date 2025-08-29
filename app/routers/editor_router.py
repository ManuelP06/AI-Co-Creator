from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import EditorAgentResponse
from app.services.editor import run_editor_agent
from app import models

router = APIRouter(prefix="/editor", tags=["editor"])

@router.post("/{video_id}/plan", response_model=EditorAgentResponse)
def plan_timeline(video_id: int, user_brief: str | None = None, db: Session = Depends(get_db)):
    # Ensure we have shots
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")
    if not video.shots:
        raise HTTPException(400, "No shots found for video. Run shot detection first.")
    return run_editor_agent(db, video_id, user_brief=user_brief)

@router.get("/{video_id}/timeline")
def get_timeline(video_id: int, db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(404, "Video not found")
    if not video.timeline_json:
        raise HTTPException(404, "No timeline generated yet.")
    return video.timeline_json
