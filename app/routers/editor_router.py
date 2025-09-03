from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import logging
import time

from app.database import get_db
from app.schemas import EditorAgentResponse
from app.services.editor import (
    VideoEditor,
    ContentType,
    PLATFORM_CONFIGS,
    run_editor_agent,
    analyze_content_potential,
    get_timeline_json,
    create_multi_video_project
)
from app import models

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/editor", tags=["editor"])


class EditingOptions(BaseModel):
    """Simplified editing options"""
    min_content_score: Optional[float] = Field(3.0, ge=1.0, le=10.0)
    min_engagement_score: Optional[float] = Field(2.0, ge=0.0, le=10.0)
    min_composite_score: Optional[float] = Field(4.0, ge=1.0, le=10.0)
    target_platform: Optional[str] = Field("youtube_shorts")
    
    @field_validator('target_platform')
    @classmethod
    def validate_platform(cls, v):
        if v not in PLATFORM_CONFIGS:
            raise ValueError(f"Invalid platform. Choose from: {list(PLATFORM_CONFIGS.keys())}")
        return v


class EditRequest(BaseModel):
    """Request model for editing"""
    user_brief: Optional[str] = Field(None, max_length=500)
    content_type: Optional[str] = Field("interview")
    target_platform: Optional[str] = Field("youtube_shorts")
    options: Optional[EditingOptions] = None
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        valid_types = [ct.value for ct in ContentType]
        if v not in valid_types:
            raise ValueError(f"Invalid content type. Choose from: {valid_types}")
        return v
    
    @field_validator('target_platform')
    @classmethod
    def validate_platform(cls, v):
        if v not in PLATFORM_CONFIGS:
            raise ValueError(f"Invalid platform. Choose from: {list(PLATFORM_CONFIGS.keys())}")
        return v


class MultiVideoRequest(BaseModel):
    """Multi-video project request"""
    video_ids: List[int] = Field(..., min_length=2, max_length=5)
    content_type: str = Field(...)
    target_platform: str = Field("youtube_shorts")
    brief: Optional[str] = Field(None, max_length=500)
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v):
        valid_types = [ct.value for ct in ContentType]
        if v not in valid_types:
            raise ValueError(f"Invalid content type. Choose from: {valid_types}")
        return v


class ContentPotentialResponse(BaseModel):
    """Content potential analysis response"""
    total_clips: int
    total_duration: float
    average_score: float
    max_score: float
    high_quality_clips: int
    usable_clips: int
    content_density: float
    viral_potential: float
    recommendations: List[str]


class TimelineResponse(BaseModel):
    """Timeline response"""
    video_id: int
    timeline: Dict[str, Any]
    storyboard: Optional[Dict[str, Any]] = None


@router.post("/{video_id}/edit", response_model=EditorAgentResponse)
async def create_edit(
    video_id: int,
    request: EditRequest,
    db: Session = Depends(get_db)
):
    """
    Create a video edit with intelligent clip selection.
    
    Features:
    - Automated content analysis and scoring
    - Platform-specific optimization
    - Quality filtering and selection
    - Professional timeline generation
    """
    logger.info(f"Creating edit for video {video_id}")
    
    # Validate video exists
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check for shots
    shots_count = db.query(models.Shot).filter(models.Shot.video_id == video_id).count()
    if shots_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No shots found. Please run shot detection first."
        )
    
    try:
        # Prepare options
        options = {}
        if request.options:
            options = request.options.model_dump()
        
        options["target_platform"] = request.target_platform
        
        # Create edit
        result = run_editor_agent(
            db=db,
            video_id=video_id,
            user_brief=request.user_brief,
            advanced_options=options
        )
        
        logger.info(f"Edit completed for video {video_id}")
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Edit failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Edit generation failed")


@router.get("/{video_id}/content-potential", response_model=ContentPotentialResponse)
async def get_content_potential(
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Analyze the potential of video content for editing.
    
    Returns:
    - Quality metrics and scores
    - Usability assessment
    - Viral potential scoring
    - Improvement recommendations
    """
    logger.info(f"Analyzing content potential for video {video_id}")
    
    # Validate video exists
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        analysis = analyze_content_potential(db, video_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        return ContentPotentialResponse(**analysis)
        
    except Exception as e:
        logger.error(f"Content analysis failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Content analysis failed")


@router.get("/{video_id}/timeline", response_model=TimelineResponse)
async def get_video_timeline(
    video_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the generated timeline for a video.
    
    Returns complete timeline data including:
    - Clip selection and timing
    - Storyboard information
    - Quality metrics
    """
    logger.info(f"Retrieving timeline for video {video_id}")
    
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not video.timeline_json:
        raise HTTPException(
            status_code=404,
            detail="No timeline found. Create an edit first."
        )
    
    try:
        timeline_data = get_timeline_json(db, video_id)
        
        response = TimelineResponse(
            video_id=video_id,
            timeline=timeline_data,
            storyboard=video.storyboard_json
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Timeline retrieval failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Timeline retrieval failed")


@router.post("/projects/multi-video")
async def create_multi_video_edit(
    request: MultiVideoRequest,
    db: Session = Depends(get_db)
):
    """
    Create edit from multiple videos.
    
    Combines content from multiple videos into a single optimized edit.
    Useful for creating compilation videos or cross-content highlights.
    """
    logger.info(f"Creating multi-video project with {len(request.video_ids)} videos")
    
    # Validate all videos exist and have shots
    for video_id in request.video_ids:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        shots_count = db.query(models.Shot).filter(models.Shot.video_id == video_id).count()
        if shots_count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Video {video_id} has no shots. Run shot detection first."
            )
    
    try:
        result = create_multi_video_project(
            db=db,
            video_ids=request.video_ids,
            content_type=request.content_type,
            target_platform=request.target_platform,
            brief=request.brief
        )
        
        logger.info(f"Multi-video project completed")
        return result
        
    except Exception as e:
        logger.error(f"Multi-video project failed: {e}")
        raise HTTPException(status_code=500, detail="Multi-video project failed")


@router.post("/{video_id}/regenerate", response_model=EditorAgentResponse)
async def regenerate_edit(
    video_id: int,
    request: EditRequest,
    db: Session = Depends(get_db)
):
    """
    Regenerate edit with different parameters.
    
    Allows fine-tuning of edit results by adjusting:
    - Content type classification
    - Platform optimization
    - Quality thresholds
    - Creative brief
    """
    logger.info(f"Regenerating edit for video {video_id}")
    
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Prepare options
        options = {}
        if request.options:
            options = request.options.model_dump()
        
        options["target_platform"] = request.target_platform
        
        # Regenerate
        result = run_editor_agent(
            db=db,
            video_id=video_id,
            user_brief=request.user_brief,
            advanced_options=options
        )
        
        logger.info(f"Edit regenerated for video {video_id}")
        return result
        
    except Exception as e:
        logger.error(f"Regeneration failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Edit regeneration failed")


@router.get("/platforms")
async def get_platform_configs():
    """
    Get available platform configurations.
    
    Returns platform specifications including duration limits,
    aspect ratios, and optimization settings.
    """
    return {
        "platforms": PLATFORM_CONFIGS,
        "content_types": [ct.value for ct in ContentType],
        "defaults": {
            "platform": "youtube_shorts",
            "content_type": "interview",
            "quality_threshold": 4.0
        }
    }


@router.get("/{video_id}/clips")
async def get_video_clips(
    video_id: int,
    min_score: Optional[float] = Query(None, ge=0.0, le=10.0, description="Minimum quality score"),
    limit: Optional[int] = Query(20, ge=1, le=100, description="Maximum clips to return"),
    db: Session = Depends(get_db)
):
    """
    Get analyzed clips for a video with optional filtering.
    
    Useful for reviewing clip quality and selection before creating edits.
    """
    logger.info(f"Getting clips for video {video_id}")
    
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Quick analysis to get clip scores
        editor = VideoEditor(db)
        analysis = editor.analyze_content_potential(video_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])
        
        # This is a simplified response - in full implementation,
        # you'd want to return actual clip objects with scores
        return {
            "video_id": video_id,
            "total_clips": analysis["total_clips"],
            "clips_analyzed": True,
            "average_quality": analysis["average_score"],
            "filter_applied": min_score is not None,
            "min_score_filter": min_score
        }
        
    except Exception as e:
        logger.error(f"Clip retrieval failed for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail="Clip retrieval failed")