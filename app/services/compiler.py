from __future__ import annotations

import os
import re
import logging
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from app import models
from app.services.editor import get_timeline_json
from app.services.timeline import Timeline, VideoClip, FFmpegRenderer, VideoFormat

OUTPUT_DIR = "renders"
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str, default: str) -> str:
    """Create safe filename by removing invalid characters"""
    if not name:
        return default
    base = _SAFE_NAME_RE.sub("_", os.path.basename(name))
    return base or default


def _build_timeline_from_json(db: Session, video_id: int) -> Timeline:
    """Build timeline from JSON data - pure hard cuts, no effects"""
    tj = get_timeline_json(db, video_id)
    items: List[Dict[str, Any]] = sorted(tj.get("items", []), key=lambda x: int(x.get("order", 0)))

    timeline = Timeline(target_fps=30.0)

    for it in items:
        clip_id = it.get("clip_id")
        if clip_id is None:
            continue

        shot: models.Shot | None = (
            db.query(models.Shot)
            .filter(models.Shot.id == int(clip_id), models.Shot.video_id == video_id)
            .first()
        )
        if not shot or not shot.video or not shot.video.file_path:
            logger.warning(f"Shot {clip_id} not found or missing video file")
            continue

        st_default = float(shot.start_time or 0.0)
        en_default = float(shot.end_time or 0.0)
        st = float(it.get("start_time", st_default))
        en = float(it.get("end_time", en_default))
        
        # Ensure valid duration
        if en <= st:
            en = max(st + 0.25, en_default)

        # Create video clip with no effects (standard speed, hard cuts only)
        vc = VideoClip(
            source_video=shot.video.file_path,
            start_frame=int(st * timeline.target_fps),
            end_frame=int(en * timeline.target_fps),
            start_time=st,
            end_time=en,
            duration=en - st,
            transcript=shot.transcript or "",
            metadata={"highlight_reason": it.get("highlight_reason", "")},
        )
        timeline.add_clip(vc)

    return timeline


def render_timeline_to_file(db: Session,
                            video_id: int,
                            output_filename: Optional[str] = None,
                            quality: str = "high",
                            use_gpu: bool = False,
                            video_format: VideoFormat = VideoFormat.PORTRAIT,
                            auto_captions: bool = False,
                            caption_style: Optional[Dict[str, str]] = None) -> str:
    """
    Build Timeline → compile with FFmpeg → return output path.
    High-performance rendering with hard cuts only.
    
    Args:
        db: Database session
        video_id: ID of video to render
        output_filename: Optional custom filename
        quality: Render quality (high, medium, low)
        use_gpu: Use GPU acceleration if available
        video_format: Output aspect ratio (VideoFormat enum)
        auto_captions: Whether to add auto-generated captions
        caption_style: Custom caption styling options
        
    Returns:
        Path to rendered video file
    """
    # Build timeline from database
    timeline = _build_timeline_from_json(db, video_id)
    
    if not timeline.clips:
        raise ValueError("No clips found in timeline")

    # Generate output filename
    if not output_filename:
        format_name = video_format.value["name"]
        output_filename = f"render_{video_id}_{format_name}.mp4"
    
    safe_name = _safe_filename(output_filename, f"render_{video_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, safe_name)

    # Initialize FFmpeg renderer
    renderer = FFmpegRenderer(timeline)
    
    # Set default caption style if not provided
    if caption_style is None:
        caption_style = {
            "font": "Arial-Bold",
            "size": "60",
            "color": "white",
            "outline": "black",
            "outline_width": "2"
        }
    
    # Render video
    try:
        renderer.render_video(
            output_path=output_path,
            video_format=video_format,
            quality=quality,
            use_gpu=use_gpu,
            auto_captions=auto_captions,
            caption_style=caption_style
        )
        
        logger.info(f"Video rendered successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        raise


def render_with_custom_format(db: Session,
                              video_id: int,
                              aspect_ratio: str = "9:16",
                              captions_enabled: bool = True,
                              output_filename: Optional[str] = None,
                              quality: str = "high",
                              use_gpu: bool = False) -> str:
    """
    Convenience function for rendering with common format presets.
    Optimized for viral short-form content.
    
    Args:
        db: Database session
        video_id: ID of video to render
        aspect_ratio: Aspect ratio string ("9:16", "16:9", "1:1")
        captions_enabled: Whether to add captions
        output_filename: Optional custom filename
        quality: Render quality
        use_gpu: Use GPU acceleration
        
    Returns:
        Path to rendered video file
    """
    # Map aspect ratio string to VideoFormat enum
    format_map = {
        "9:16": VideoFormat.PORTRAIT,
        "16:9": VideoFormat.LANDSCAPE, 
        "1:1": VideoFormat.SQUARE
    }
    
    video_format = format_map.get(aspect_ratio, VideoFormat.PORTRAIT)
    
    # Adjust caption size based on format
    caption_style = {
        "font": "Arial-Bold",
        "size": "60" if aspect_ratio == "9:16" else "48",
        "color": "white",
        "outline": "black", 
        "outline_width": "2"
    }
    
    return render_timeline_to_file(
        db=db,
        video_id=video_id,
        output_filename=output_filename,
        quality=quality,
        use_gpu=use_gpu,
        video_format=video_format,
        auto_captions=captions_enabled,
        caption_style=caption_style
    )


def batch_render_multiple_formats(db: Session,
                                 video_id: int,
                                 formats: List[str] = None,
                                 base_filename: Optional[str] = None,
                                 quality: str = "high",
                                 use_gpu: bool = False,
                                 captions: bool = True) -> Dict[str, str]:
    """
    Render the same video in multiple formats for different platforms.
    Perfect for viral content distribution.
    
    Args:
        db: Database session
        video_id: ID of video to render
        formats: List of aspect ratios to render (default: all formats)
        base_filename: Base filename (format will be appended)
        quality: Render quality
        use_gpu: Use GPU acceleration
        captions: Add captions to all formats
        
    Returns:
        Dict mapping format to output path
    """
    if formats is None:
        formats = ["9:16", "16:9", "1:1"]  # All formats
    
    results = {}
    
    for aspect_ratio in formats:
        try:
            # Generate format-specific filename
            if base_filename:
                format_name = aspect_ratio.replace(":", "x")
                filename = f"{base_filename}_{format_name}.mp4"
            else:
                filename = None
            
            # Render this format
            output_path = render_with_custom_format(
                db=db,
                video_id=video_id,
                aspect_ratio=aspect_ratio,
                captions_enabled=captions,
                output_filename=filename,
                quality=quality,
                use_gpu=use_gpu
            )
            
            results[aspect_ratio] = output_path
            logger.info(f"Successfully rendered {aspect_ratio} format: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to render {aspect_ratio} format: {e}")
            results[aspect_ratio] = None
    
    return results


def get_render_progress(video_id: int) -> Dict[str, Any]:
    """
    Get rendering progress information (placeholder for future implementation).
    Could be extended to track FFmpeg progress via callback.
    
    Args:
        video_id: Video ID being rendered
        
    Returns:
        Progress information dict
    """
    # This could be implemented with FFmpeg progress monitoring
    # For now, return basic info
    return {
        "video_id": video_id,
        "status": "unknown",
        "progress_percent": 0,
        "estimated_time_remaining": None
    }


# Utility functions for FFmpeg management
def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available in system PATH"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_ffmpeg_info() -> Dict[str, Any]:
    """Get FFmpeg version and capability information"""
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            version_line = lines[0] if lines else "Unknown"
            
            # Check for GPU support
            has_nvenc = "h264_nvenc" in result.stdout
            has_vaapi = "vaapi" in result.stdout
            
            return {
                "available": True,
                "version": version_line,
                "gpu_support": {
                    "nvenc": has_nvenc,
                    "vaapi": has_vaapi
                }
            }
    except Exception as e:
        logger.warning(f"Could not get FFmpeg info: {e}")
    
    return {"available": False, "error": "FFmpeg not found or not working"}