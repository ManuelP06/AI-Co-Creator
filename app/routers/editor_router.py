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


# Request/Response Models
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


# Single Video Editing Endpoints
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
    # Validate video exists
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Check for processed shots
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


@router.get("/{video_id}/analytics")
def get_content_analytics(video_id: int, db: Session = Depends(get_db)):
    """
    Analyze content potential and quality metrics for a video.
    
    Provides insights into clip quality, content density, and editing potential.
    """
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        analytics = analyze_content_potential(db, video_id)
        if "error" in analytics:
            raise HTTPException(status_code=400, detail=analytics["error"])
        
        return {
            "video_id": video_id,
            "analytics": analytics,
            "recommendations": _generate_content_recommendations(analytics)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


# Multi-Video Project Endpoints
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
    # Validate all videos exist and have shots
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


@router.get("/projects/analytics")
def get_multi_video_analytics(
    video_ids: List[int] = Query(..., description="List of video IDs to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get analytics for potential multi-video project.
    
    Analyzes multiple videos to provide insights about content quality,
    estimated output duration, and editing recommendations.
    """
    if len(video_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 videos required for analysis")
    
    # Validate videos exist
    for video_id in video_ids:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    try:
        analytics = get_project_analytics(db, video_ids)
        return {
            "analytics": analytics,
            "recommendations": _generate_multi_video_recommendations(analytics),
            "content_types": [ct.value for ct in ContentType],
            "platforms": list(PLATFORM_LIMITS.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


# Content Type and Platform Management
@router.get("/content-types")
def get_content_types():
    """Get available content types for editing projects"""
    return {
        "content_types": [
            {
                "value": ct.value,
                "name": ct.value.replace("_", " ").title(),
                "description": _get_content_type_description(ct)
            }
            for ct in ContentType
        ]
    }


@router.get("/platforms")
def get_supported_platforms():
    """Get supported platforms with their specifications"""
    from app.services.editor import PLATFORM_LIMITS
    
    return {
        "platforms": [
            {
                "platform": platform,
                "max_duration": limits["max_duration"],
                "ideal_clips": limits["ideal_clips"],
                "max_clip_duration": limits["max_clip_duration"],
                "recommended_for": _get_platform_recommendation(platform)
            }
            for platform, limits in PLATFORM_LIMITS.items()
        ]
    }


# Advanced Editing Features
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
    
    # Determine content type
    try:
        content_enum = ContentType(content_type) if content_type else ContentType.PODCAST_HIGHLIGHTS
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {content_type}")
    
    # Create new project with updated parameters
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


# Helper Functions
def _generate_content_recommendations(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Generate content recommendations based on analytics"""
    recommendations = {
        "suitable_for_editing": analytics.get("usable_clips", 0) > 3,
        "estimated_quality": "high" if analytics.get("average_score", 0) > 5.0 else "medium",
        "recommended_duration": min(analytics.get("total_duration", 0) * 0.2, 60.0),
        "best_content_type": "podcast_highlights",  # Default
        "editing_tips": []
    }
    
    # Dynamic recommendations based on content
    if analytics.get("content_density", 0) > 0.6:
        recommendations["editing_tips"].append("High content density - consider multiple short clips")
    
    if analytics.get("high_quality_clips", 0) > 5:
        recommendations["editing_tips"].append("Multiple high-quality moments available")
    
    return recommendations


def _generate_multi_video_recommendations(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Generate recommendations for multi-video projects"""
    return {
        "feasible": analytics.get("high_quality_clips", 0) > 5,
        "estimated_quality": "high" if analytics.get("average_quality", 0) > 4.0 else "medium",
        "recommended_clips": analytics.get("recommended_highlights", 6),
        "content_strategy": "chronological" if analytics.get("source_videos", 0) > 3 else "best_moments",
        "platform_recommendations": [
            "youtube_shorts" if analytics.get("estimated_output_duration", 0) > 30 else "tiktok"
        ]
    }


def _get_content_type_description(content_type: ContentType) -> str:
    """Get user-friendly description for content types"""
    descriptions = {
        ContentType.PODCAST_HIGHLIGHTS: "Extract key insights and memorable moments from podcasts",
        ContentType.INTERVIEW_CLIPS: "Create engaging clips from interviews and conversations",
        ContentType.EDUCATIONAL_SUMMARY: "Transform educational content into digestible segments",
        ContentType.ENTERTAINMENT_COMPILATION: "Compile entertaining moments for viral content",
        ContentType.PRODUCT_DEMO: "Showcase product features and demonstrations",
        ContentType.TESTIMONIAL_REEL: "Compile customer testimonials and reviews",
        ContentType.EVENT_HIGHLIGHTS: "Create highlight reels from events and presentations"
    }
    return descriptions.get(content_type, "Professional content editing")


def _get_platform_recommendation(platform: str) -> str:
    """Get platform-specific recommendations"""
    recommendations = {
        "youtube_shorts": "Best for educational and interview content",
        "tiktok": "Optimized for viral, entertainment-focused content",
        "instagram_reels": "Great for lifestyle, product demos, and testimonials",
        "custom": "Flexible format for any use case"
    }
    return recommendations.get(platform, "General purpose platform")