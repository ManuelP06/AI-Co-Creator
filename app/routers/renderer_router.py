import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app import models
from app.database import get_db
from app.services.renderer import (Timeline, VideoClip, VideoFormat,
                                   VideoRenderer, check_ffmpeg_available,
                                   get_quality_settings)

router = APIRouter(prefix="/renderer", tags=["renderer"])

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

render_jobs: Dict[str, Dict[str, Any]] = {}


class RenderRequest(BaseModel):
    """Simple render request model"""

    output_filename: Optional[str] = Field(None, description="Custom output filename")
    video_format: str = Field(
        default="portrait", description="Video format: portrait, landscape, square"
    )
    quality: str = Field(
        default="high", description="Render quality: ultra, high, medium, low"
    )
    use_gpu: bool = Field(default=False, description="Use GPU acceleration")
    max_duration: Optional[float] = Field(
        default=59.0, description="Maximum video duration in seconds"
    )


@router.post("/{video_id}/render")
def render_video(
    video_id: int,
    request: RenderRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Render a video timeline to final output with maximum quality"""

    # Validate video exists
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not video.timeline_json:
        raise HTTPException(
            status_code=400,
            detail="No timeline found. Please create an edit plan first.",
        )

    # Validate FFmpeg
    if not check_ffmpeg_available():
        raise HTTPException(
            status_code=500, detail="FFmpeg not available. Please install FFmpeg."
        )

    # Create job
    job_id = f"render_{video_id}_{int(datetime.now().timestamp())}"
    # Validate and sanitize output filename
    if request.output_filename and request.output_filename.strip() and request.output_filename != "string":
        # Ensure it has proper extension
        output_filename = request.output_filename.strip()
        if not output_filename.endswith(('.mp4', '.mov', '.avi')):
            output_filename = f"{output_filename}.mp4"
    else:
        output_filename = f"video_{video_id}_{request.video_format}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    render_jobs[job_id] = {
        "status": "queued",
        "video_id": video_id,
        "output_path": output_path,
        "filename": output_filename,
        "created_at": datetime.now(),
        "progress": 0,
        "settings": {
            "video_format": request.video_format,
            "quality": request.quality,
            "use_gpu": request.use_gpu,
            "max_duration": request.max_duration,
        },
    }

    # Start background rendering
    background_tasks.add_task(
        _render_video_background,
        job_id=job_id,
        video_id=video_id,
        request=request,
        db=db,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "video_id": video_id,
        "output_filename": output_filename,
        "created_at": datetime.now(),
    }


@router.get("/jobs/{job_id}/status")
def get_render_status(job_id: str):
    """Get rendering job status"""
    if job_id not in render_jobs:
        raise HTTPException(status_code=404, detail="Render job not found")

    job = render_jobs[job_id]

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "output_path": job.get("output_path"),
        "filename": job.get("filename"),
        "error": job.get("error"),
        "completed_at": job.get("completed_at"),
        "duration": job.get("duration"),
        "settings": job.get("settings", {}),
    }


@router.get("/outputs/{filename}/download")
def download_video(filename: str):
    """Download rendered video file"""
    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    if not filename.endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    return FileResponse(path=file_path, filename=filename, media_type="video/mp4")


@router.get("/outputs")
def list_outputs():
    """List all rendered video files"""
    try:
        files = []
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith((".mp4", ".mov", ".avi")):
                file_path = os.path.join(OUTPUT_DIR, filename)
                stat = os.stat(file_path)

                files.append(
                    {
                        "filename": filename,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "created_at": datetime.fromtimestamp(stat.st_ctime),
                    }
                )

        files.sort(key=lambda x: x["created_at"], reverse=True)
        return {"files": files}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@router.delete("/outputs/{filename}")
def delete_output(filename: str):
    """Delete a rendered video file"""
    file_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        os.remove(file_path)
        return {"status": "deleted", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@router.get("/system")
def get_system_info():
    """Get system capabilities and settings"""
    return {
        "ffmpeg_available": check_ffmpeg_available(),
        "video_formats": [
            {
                "name": fmt.name.lower(),
                "resolution": f"{fmt.value['width']}x{fmt.value['height']}",
                "aspect_ratio": fmt.value["ratio"],
            }
            for fmt in VideoFormat
        ],
        "quality_settings": get_quality_settings(),
        "output_directory": OUTPUT_DIR,
    }


@router.get("/jobs")
def list_jobs():
    """List all render jobs"""
    jobs = list(render_jobs.values())
    jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    return {"jobs": jobs, "total": len(jobs)}


@router.get("/jobs/{video_id}")
def list_video_jobs(video_id: int):
    """List render jobs for a specific video"""
    video_jobs = [job for job in render_jobs.values() if job.get("video_id") == video_id]
    video_jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)

    return {"jobs": video_jobs, "total": len(video_jobs)}


class QueueRequest(BaseModel):
    """Request to add job to render queue"""
    video_id: int
    platform: str = "youtube_shorts"
    quality: str = "high"


@router.post("/queue")
def add_to_render_queue(request: QueueRequest, db: Session = Depends(get_db)):
    """Add a video to the render queue"""
    # Validate video exists
    video = db.query(models.Video).filter(models.Video.id == request.video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not video.timeline_json:
        raise HTTPException(
            status_code=400,
            detail="No timeline found. Please create an edit plan first.",
        )

    # Create job
    job_id = f"queue_{request.video_id}_{int(datetime.now().timestamp())}"
    output_filename = f"video_{request.video_id}_{request.platform}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    render_jobs[job_id] = {
        "status": "queued",
        "video_id": request.video_id,
        "output_path": output_path,
        "filename": output_filename,
        "created_at": datetime.now(),
        "progress": 0,
        "settings": {
            "video_format": request.platform,
            "quality": request.quality,
            "use_gpu": False,
            "max_duration": 59.0,
        },
    }

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Added to render queue successfully"
    }


async def _render_video_background(
    job_id: str, video_id: int, request: RenderRequest, db: Session
):
    """Background video rendering task"""
    try:
        render_jobs[job_id]["status"] = "processing"
        render_jobs[job_id]["progress"] = 10

        # Get video and timeline data
        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        timeline_data = video.timeline_json
        render_jobs[job_id]["progress"] = 20

        # Build timeline
        timeline = Timeline()

        for item in timeline_data.get("items", []):
            shot = (
                db.query(models.Shot).filter(models.Shot.id == item["clip_id"]).first()
            )
            if not shot:
                continue

            video_clip = VideoClip(
                source_video=shot.video.file_path,
                start_time=item["start_time"],
                end_time=item["end_time"],
                duration=item["end_time"] - item["start_time"],
                metadata={"reason": item.get("highlight_reason", "")},
            )
            timeline.add_clip(video_clip)

        render_jobs[job_id]["progress"] = 40

        # Optimize timeline duration
        if request.max_duration:
            timeline.optimize_for_duration(request.max_duration)

        # Map format
        format_map = {
            "portrait": VideoFormat.PORTRAIT,
            "landscape": VideoFormat.LANDSCAPE,
            "square": VideoFormat.SQUARE,
        }
        video_format = format_map.get(request.video_format, VideoFormat.PORTRAIT)

        render_jobs[job_id]["progress"] = 50

        # Render video
        renderer = VideoRenderer(timeline)
        output_path = render_jobs[job_id]["output_path"]

        renderer.render_video(
            output_path=output_path,
            video_format=video_format,
            quality=request.quality,
            use_gpu=request.use_gpu,
        )

        # Success
        render_jobs[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "completed_at": datetime.now(),
                "duration": timeline.total_duration,
            }
        )

        logger.info(f"Render job {job_id} completed successfully")

    except Exception as e:
        render_jobs[job_id].update(
            {"status": "failed", "error": str(e), "completed_at": datetime.now()}
        )
        logger.error(f"Render job {job_id} failed: {e}")
