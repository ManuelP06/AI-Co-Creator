from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import os
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from app.database import get_db
from app.services.renderer import (
    Timeline,
    VideoClip,
    ViralVideoRenderer,
    VideoFormat,
    VIRAL_CAPTION_STYLES,
    check_ffmpeg_capabilities
)
from app.services.editor import get_timeline_json
from app import models

router = APIRouter(prefix="/renderer", tags=["renderer"])

# Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Background job tracking (in production, use Redis/Celery)
render_jobs: Dict[str, Dict[str, Any]] = {}


# Request/Response Models
class RenderRequest(BaseModel):
    """Request model for video rendering"""
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    platform: str = Field(default="youtube_shorts", description="Target platform")
    quality: str = Field(default="high", description="Render quality (high/medium/low)")
    use_gpu: bool = Field(default=False, description="Use GPU acceleration if available")
    auto_captions: bool = Field(default=True, description="Add automatic captions")
    caption_style: Optional[str] = Field(default=None, description="Caption style preset")


class BatchRenderRequest(BaseModel):
    """Request model for batch rendering multiple videos"""
    video_ids: List[int] = Field(..., description="List of video IDs to render")
    platform: str = Field(default="youtube_shorts", description="Target platform")
    quality: str = Field(default="high", description="Render quality")
    use_gpu: bool = Field(default=False, description="Use GPU acceleration")
    auto_captions: bool = Field(default=True, description="Add automatic captions")


class RenderResponse(BaseModel):
    """Response model for render operations"""
    job_id: str
    status: str
    video_id: int
    output_path: Optional[str] = None
    filename: Optional[str] = None
    duration: Optional[float] = None
    platform: str
    created_at: datetime


# Single Video Rendering
@router.post("/{video_id}/render")
def render_video(
    video_id: int,
    request: RenderRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Render a video timeline to final output.
    
    Creates a professionally edited short-form video optimized for the target platform.
    Supports GPU acceleration and automatic captions for maximum engagement.
    """
    # Validate video and timeline exist
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if not video.timeline_json:
        raise HTTPException(
            status_code=400,
            detail="No timeline found. Please create an edit plan first."
        )
    
    # Generate job ID and output path
    job_id = f"render_{video_id}_{int(datetime.now().timestamp())}"
    output_filename = request.output_filename or f"video_{video_id}_{request.platform}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Create job entry
    render_jobs[job_id] = {
        "status": "queued",
        "video_id": video_id,
        "output_path": output_path,
        "filename": output_filename,
        "platform": request.platform,
        "created_at": datetime.now(),
        "progress": 0
    }
    
    # Queue background rendering
    background_tasks.add_task(
        _render_video_background,
        job_id=job_id,
        video_id=video_id,
        request=request,
        db=db
    )
    
    return RenderResponse(
        job_id=job_id,
        status="queued",
        video_id=video_id,
        platform=request.platform,
        created_at=datetime.now()
    )


@router.get("/{video_id}/render/{job_id}/status")
def get_render_status(video_id: int, job_id: str):
    """
    Get rendering job status.
    
    Returns current status, progress, and completion details for a render job.
    """
    if job_id not in render_jobs:
        raise HTTPException(status_code=404, detail="Render job not found")
    
    job = render_jobs[job_id]
    if job["video_id"] != video_id:
        raise HTTPException(status_code=400, detail="Job ID doesn't match video ID")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "output_path": job.get("output_path"),
        "filename": job.get("filename"),
        "error": job.get("error"),
        "completed_at": job.get("completed_at"),
        "duration": job.get("duration")
    }


# Batch Rendering
@router.post("/batch/render")
def batch_render_videos(
    request: BatchRenderRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Render multiple videos in batch.
    
    Efficiently processes multiple videos with the same settings.
    Returns batch job ID for tracking progress.
    """
    # Validate all videos exist and have timelines
    for video_id in request.video_ids:
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        if not video.timeline_json:
            raise HTTPException(
                status_code=400,
                detail=f"Video {video_id} has no timeline. Create edit plan first."
            )
    
    # Create batch job
    batch_job_id = f"batch_{int(datetime.now().timestamp())}"
    
    # Queue individual render jobs
    job_ids = []
    for video_id in request.video_ids:
        job_id = f"render_{video_id}_{int(datetime.now().timestamp())}"
        
        render_request = RenderRequest(
            platform=request.platform,
            quality=request.quality,
            use_gpu=request.use_gpu,
            auto_captions=request.auto_captions
        )
        
        background_tasks.add_task(
            _render_video_background,
            job_id=job_id,
            video_id=video_id,
            request=render_request,
            db=db
        )
        
        job_ids.append(job_id)
    
    return {
        "batch_job_id": batch_job_id,
        "individual_jobs": job_ids,
        "total_videos": len(request.video_ids),
        "status": "queued"
    }


# File Management
@router.get("/download/{filename}")
def download_rendered_video(filename: str):
    """
    Download rendered video file.
    
    Serves the final rendered video for download.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not filename.endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )


@router.get("/outputs")
def list_rendered_videos(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0)
):
    """
    List all rendered video files.
    
    Returns paginated list of available rendered videos.
    """
    try:
        files = []
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(('.mp4', '.mov', '.avi')):
                file_path = os.path.join(OUTPUT_DIR, filename)
                stat = os.stat(file_path)
                
                files.append({
                    "filename": filename,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_ctime),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime)
                })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        total = len(files)
        files = files[offset:offset + limit]
        
        return {
            "files": files,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(files) < total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.delete("/outputs/{filename}")
def delete_rendered_video(filename: str):
    """
    Delete a rendered video file.
    
    Removes the specified file from the output directory.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    if not filename.endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        os.remove(file_path)
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


# System and Capabilities
@router.get("/capabilities")
def get_rendering_capabilities():
    """
    Get system rendering capabilities.
    
    Returns information about available codecs, GPU support, and system specs.
    """
    try:
        capabilities = check_ffmpeg_capabilities()
        
        return {
            "ffmpeg": capabilities,
            "supported_formats": [
                {
                    "name": fmt.name,
                    "value": fmt.value,
                    "resolution": f"{fmt.value['width']}x{fmt.value['height']}",
                    "aspect_ratio": fmt.value["ratio"],
                    "platform": fmt.value["platform"]
                }
                for fmt in VideoFormat
            ],
            "caption_styles": list(VIRAL_CAPTION_STYLES.keys()),
            "quality_options": ["high", "medium", "low"],
            "output_directory": OUTPUT_DIR
        }
    except Exception as e:
        return {
            "error": f"Failed to check capabilities: {str(e)}",
            "ffmpeg": {"available": False}
        }



# Background Job Functions
async def _render_video_background(
    job_id: str,
    video_id: int,
    request: RenderRequest,
    db: Session
):
    """Background task for video rendering"""
    try:
        # Update job status
        render_jobs[job_id]["status"] = "processing"
        render_jobs[job_id]["progress"] = 10
        
        # Get timeline data
        timeline_data = get_timeline_json(db, video_id)
        
        # Create timeline object
        timeline = Timeline()
        
        # Convert timeline items to video clips
        for item in timeline_data.get("items", []):
            # Get source video path from database
            shot = db.query(models.Shot).filter(models.Shot.id == item["clip_id"]).first()
            if not shot:
                continue
            
            video_clip = VideoClip(
                source_video=shot.video.file_path,
                start_frame=0,  # Calculate if needed
                end_frame=0,    # Calculate if needed
                start_time=item["start_time"],
                end_time=item["end_time"],
                duration=item["end_time"] - item["start_time"],
                transcript=shot.transcript or "",
                metadata={"highlight_reason": item.get("highlight_reason", "")}
            )
            timeline.add_clip(video_clip)
        
        render_jobs[job_id]["progress"] = 30
        
        # Initialize renderer
        renderer = ViralVideoRenderer(timeline)
        
        # Render video
        output_path = render_jobs[job_id]["output_path"]
        
        render_jobs[job_id]["progress"] = 50
        
        if request.platform in ["tiktok", "instagram", "youtube_shorts"]:
            renderer.render_viral_clip(
                output_path=output_path,
                platform=request.platform,
                quality=request.quality,
                use_gpu=request.use_gpu
            )
        else:
            # Custom rendering
            video_format = VideoFormat.PORTRAIT  # Default
            caption_style = VIRAL_CAPTION_STYLES.get(request.caption_style, VIRAL_CAPTION_STYLES["professional"])
            
            renderer.render_video(
                output_path=output_path,
                video_format=video_format,
                quality=request.quality,
                use_gpu=request.use_gpu,
                auto_captions=request.auto_captions,
                caption_style=caption_style
            )
        
        # Update job completion
        render_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.now(),
            "duration": timeline.total_duration
        })
        
    except Exception as e:
        render_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now()
        })
        logger.error(f"Render job {job_id} failed: {e}")


# Job Management
@router.get("/jobs")
def list_render_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(default=20, le=100)
):
    """
    List rendering jobs with optional status filtering.
    
    Shows current and completed render jobs for monitoring.
    """
    jobs = list(render_jobs.values())
    
    if status:
        jobs = [job for job in jobs if job.get("status") == status]
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    
    # Apply limit
    jobs = jobs[:limit]
    
    return {
        "jobs": jobs,
        "total": len(jobs),
        "statuses": list(set(job.get("status", "unknown") for job in render_jobs.values()))
    }


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    Get detailed status for a specific render job.
    
    Returns comprehensive information about job progress and results.
    """
    if job_id not in render_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = render_jobs[job_id]
    
    # Add file info if completed
    if job["status"] == "completed" and job.get("output_path"):
        if os.path.exists(job["output_path"]):
            stat = os.stat(job["output_path"])
            job["file_size_mb"] = round(stat.st_size / (1024 * 1024), 2)
        else:
            job["status"] = "failed"
            job["error"] = "Output file not found"
    
    return job
