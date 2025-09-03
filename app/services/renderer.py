from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Iterable, Any

logger = logging.getLogger(__name__)


class VideoFormat(Enum):
    """Supported video output formats"""
    PORTRAIT = {"width": 1080, "height": 1920, "name": "9x16", "ratio": "9:16"}
    LANDSCAPE = {"width": 1920, "height": 1080, "name": "16x9", "ratio": "16:9"}
    SQUARE = {"width": 1080, "height": 1080, "name": "1x1", "ratio": "1:1"}


@dataclass
class VideoClip:
    """Simple video clip with timing information"""
    source_video: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict = field(default_factory=dict)


class Timeline:
    def __init__(self, target_fps: float = 30.0):
        self.clips: List[VideoClip] = []
        self.target_fps = float(target_fps)
        self.total_duration = 0.0

    def add_clip(self, clip: VideoClip, position: Optional[int] = None) -> None:
        if position is None:
            self.clips.append(clip)
        else:
            position = max(0, min(len(self.clips), int(position)))
            self.clips.insert(position, clip)
        self._update_timeline()

    def _update_timeline(self) -> None:
        self.total_duration = sum(float(c.duration) for c in self.clips)

    def optimize_for_duration(self, max_duration: float = 59.0) -> None:
        """Trim timeline to fit within maximum duration"""
        current_duration = 0.0
        optimized_clips = []
        
        for clip in self.clips:
            if current_duration + clip.duration <= max_duration:
                optimized_clips.append(clip)
                current_duration += clip.duration
            elif max_duration - current_duration > 0.5:
                # Trim the last clip to fit
                remaining_time = max_duration - current_duration
                clip.end_time = clip.start_time + remaining_time
                clip.duration = remaining_time
                optimized_clips.append(clip)
                break
            else:
                break
        
        self.clips = optimized_clips
        self._update_timeline()
        logger.info(f"Optimized timeline: {len(self.clips)} clips, {self.total_duration:.1f}s")

    def to_dict(self) -> Dict:
        return {
            "timeline": {
                "fps": self.target_fps,
                "duration": self.total_duration,
                "clips": [asdict(c) for c in self.clips],
            }
        }

    def export_json(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def import_json(self, filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        t = data["timeline"]
        self.target_fps = float(t.get("fps", 30.0))
        self.clips = [VideoClip(**c) for c in t.get("clips", [])]
        self._update_timeline()


class VideoRenderer:
    """High-quality FFmpeg video renderer focused on stability"""
    
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.temp_dir = None

    def render_video(self,
                    output_path: str,
                    video_format: VideoFormat = VideoFormat.PORTRAIT,
                    quality: str = "high",
                    use_gpu: bool = False) -> None:
        """Render video with maximum quality and stability"""
        if not self.timeline.clips:
            raise RuntimeError("Timeline has no clips to render.")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            
            try:
                logger.info(f"Rendering {len(self.timeline.clips)} clips to {video_format.value['name']} format")
                self._render_final_video(output_path, video_format, quality, use_gpu)
                logger.info(f"Video rendered successfully: {output_path}")
                
            except Exception as e:
                logger.error(f"Rendering failed: {e}")
                raise

    def _render_final_video(self, output_path: str, video_format: VideoFormat, quality: str, use_gpu: bool) -> None:
        """Render final video using FFmpeg concat demuxer for maximum stability"""
        
        # Prepare segments
        segment_files = self._prepare_segments()
        if not segment_files:
            raise RuntimeError("No valid segments to render")
        
        # Create concat file
        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        self._create_concat_file(segment_files, concat_file)
        
        # Build and execute FFmpeg command
        cmd = self._build_ffmpeg_command(
            concat_file=concat_file,
            output_path=output_path,
            video_format=video_format,
            quality=quality,
            use_gpu=use_gpu
        )
        
        self._execute_ffmpeg(cmd)

    def _prepare_segments(self) -> List[str]:
        """Prepare video segments with precise timing"""
        segment_files = []
        
        for i, clip in enumerate(self.timeline.clips):
            if not os.path.exists(clip.source_video):
                logger.warning(f"Source video not found: {clip.source_video}")
                continue
            
            segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
            
            # Extract segment with high precision
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{clip.start_time:.3f}",
                "-i", clip.source_video,
                "-t", f"{clip.duration:.3f}",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",
                segment_file
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                segment_files.append(segment_file)
                logger.debug(f"Created segment {i}: {clip.duration:.2f}s")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create segment {i}: {e}")
                continue
        
        return segment_files

    def _create_concat_file(self, segment_files: List[str], concat_file: str) -> None:
        """Create FFmpeg concat file"""
        with open(concat_file, 'w', encoding='utf-8') as f:
            for segment in segment_files:
                abs_path = os.path.abspath(segment).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

    def _build_ffmpeg_command(self,
                             concat_file: str,
                             output_path: str,
                             video_format: VideoFormat,
                             quality: str,
                             use_gpu: bool) -> List[str]:
        """Build optimized FFmpeg command for high quality output"""
        
        cmd = ["ffmpeg", "-y", "-loglevel", "info"]
        
        # Input
        cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_file])
        
        # Video processing
        format_info = video_format.value
        width = format_info["width"]
        height = format_info["height"]
        
        # Scaling and cropping filters
        filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}",
            "unsharp=5:5:1.0:5:5:0.0"  # Light sharpening
        ]
        cmd.extend(["-vf", ",".join(filters)])
        
        # Video encoding
        if use_gpu:
            cmd.extend(["-c:v", "h264_nvenc"])
            if quality == "high":
                cmd.extend(["-preset", "p4", "-rc", "vbr", "-cq", "16"])
            elif quality == "medium":
                cmd.extend(["-preset", "p5", "-rc", "vbr", "-cq", "20"])
            else:
                cmd.extend(["-preset", "p6", "-rc", "vbr", "-cq", "24"])
        else:
            cmd.extend(["-c:v", "libx264"])
            if quality == "high":
                cmd.extend(["-preset", "slow", "-crf", "16"])
            elif quality == "medium":
                cmd.extend(["-preset", "medium", "-crf", "20"])
            else:
                cmd.extend(["-preset", "fast", "-crf", "24"])
        
        # Audio encoding
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000"])
        
        # Output options for maximum compatibility
        cmd.extend([
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            "-max_muxing_queue_size", "9999",
        ])
        
        cmd.append(output_path)
        return cmd

    def _execute_ffmpeg(self, cmd: List[str]) -> None:
        """Execute FFmpeg command with error handling"""
        logger.info("Executing FFmpeg command...")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=3600
            )
            
            if result.stderr:
                logger.debug(f"FFmpeg output: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg rendering timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"FFmpeg rendering failed: {e.stderr}")


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available and working"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def get_quality_settings() -> Dict[str, Dict[str, Any]]:
    """Get available quality settings"""
    return {
        "high": {
            "description": "Highest quality, slower render",
            "crf_cpu": 16,
            "crf_gpu": 16,
            "preset_cpu": "slow"
        },
        "medium": {
            "description": "Balanced quality and speed", 
            "crf_cpu": 20,
            "crf_gpu": 20,
            "preset_cpu": "medium"
        },
        "low": {
            "description": "Faster render, lower quality",
            "crf_cpu": 24,
            "crf_gpu": 24, 
            "preset_cpu": "fast"
        }
    }