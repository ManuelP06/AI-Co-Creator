from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.services.editor import PLATFORM_LIMITS

from app.database import get_db
from app.schemas import EditorAgentResponse
from app.services.editor import (
    VideoEditor, 
    ContentType, 
    run_editor_agent,
    create_multi_video_project,
    get_project_analytics,
    analyze_content_potential
)
from app import models

router = APIRouter(prefix="/editor", tags=["editor"])


class EditingProjectRequest(BaseModel):
    """Request model for creating editing projects"""
    name: str = Field(..., description="Project name")
    content_type: str = Field(..., description="Type of content to create")
    target_platform: str = Field(default="youtube_shorts", description="Target platform")
    brief: Optional[str] = Field(None, description="Creative brief or instructions")


class MultiVideoProjectRequest(BaseModel):
    """Request model for multi-video projects"""
    video_ids: List[int] = Field(..., min_items=2, description="List of video IDs to combine")
    project_name: str = Field(..., description="Project name")
    content_type: str = Field(..., description="Type of content to create")
    target_platform: str = Field(default="youtube_shorts", description="Target platform")
    brief: Optional[str] = Field(None, description="Creative brief or instructions")


class ProjectResponse(BaseModel):
    """Response model for project operations"""
    project_id: int
    status: str
    source_videos: int
    total_clips: int
    selected_clips: int
    final_duration: float
    content_type: str
    platform: str


@router.post("/{video_id}/plan", response_model=EditorAgentResponse)
def create_edit_plan(
    video_id: int,
    user_brief: Optional[str] = Body(None, description="Creative brief or editing instructions"),
    db: Session = Depends(get_db)
):
    """
    Create an editing plan for a single video.
    
    This endpoint analyzes the video content and generates an optimized timeline
    with the most engaging clips for short-form content creation.
    """
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    shots_count = db.query(models.Shot).filter(models.Shot.video_id == video_id).count()
    if shots_count == 0:
        raise HTTPException(
            status_code=400, 
            detail="No shots found for video. Please run shot detection first."
        )
    
    try:
        return run_editor_agent(db, video_id, user_brief=user_brief)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create edit plan: {str(e)}")


@router.get("/{video_id}/timeline")
def get_video_timeline(video_id: int, db: Session = Depends(get_db)):
    """
    Retrieve the generated timeline for a video.
    
    Returns the complete timeline JSON with clip details, timing, and metadata.
    """
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not video.timeline_json:
        raise HTTPException(
            status_code=404, 
            detail="No timeline generated yet. Please create an edit plan first."
        )
    
    return {
        "video_id": video_id,
        "timeline": video.timeline_json,
        "storyboard": video.storyboard_json,
        "generated_at": video.updated_at
    }

@router.post("/projects/multi-video", response_model=ProjectResponse)
def create_multi_video_edit(
    request: MultiVideoProjectRequest,
    db: Session = Depends(get_db)
):
    """
    Create a highlight reel from multiple videos.
    
    Combines content from multiple source videos into a single engaging compilation.
    Perfect for creating podcast highlights, interview compilations, or event summaries.
    """
    for video_id in request.video_ids:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        shots_count = db.query(models.Shot).filter(models.Shot.video_id == video_id).count()
        if shots_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Video {video_id} has no shots. Please run shot detection first."
            )
    
    try:
        result = create_multi_video_project(
            db=db,
            video_ids=request.video_ids,
            content_type=request.content_type,
            target_platform=request.target_platform,
            brief=request.brief
        )
        
        return ProjectResponse(
            project_id=result["project_id"],
            status="completed",
            source_videos=result["source_videos"],
            total_clips=result["total_clips"],
            selected_clips=result["selected_clips"],
            final_duration=result["final_duration"],
            content_type=result["content_type"],
            platform=result["platform"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-video project failed: {str(e)}")

@router.post("/{video_id}/regenerate")
def regenerate_timeline(
    video_id: int,
    content_type: Optional[str] = Body(None),
    platform: Optional[str] = Body(None),
    brief: Optional[str] = Body(None),
    db: Session = Depends(get_db)
):
    """
    Regenerate timeline with different parameters.
    
    Allows users to try different content types or platforms for the same video.
    """
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    editor = VideoEditor(db)
    
    try:
        content_enum = ContentType(content_type) if content_type else ContentType.PODCAST_HIGHLIGHTS
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {content_type}")
    
    project = editor.create_project(
        name=f"Video_{video_id}_regenerated",
        content_type=content_enum,
        target_platform=platform or "youtube_shorts",
        brief=brief
    )
    
    try:
        editor.add_video_to_project(project, video_id)
        result = editor.generate_edit(project)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {str(e)}")
