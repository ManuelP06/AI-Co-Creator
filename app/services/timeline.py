from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Iterable, Tuple

logger = logging.getLogger(__name__)


class VideoFormat(Enum):
    """Supported video output formats with dimensions"""
    PORTRAIT = {"width": 1080, "height": 1920, "name": "9x16", "ratio": "9:16"}
    LANDSCAPE = {"width": 1920, "height": 1080, "name": "16x9", "ratio": "16:9"}
    SQUARE = {"width": 1080, "height": 1080, "name": "1x1", "ratio": "1:1"}


@dataclass
class VideoClip:
    source_video: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    metadata: Dict = field(default_factory=dict)   
    transcript: str = ""
    relevance_score: float = 0.0


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

    def reorder_clips(self, new_order: Iterable[int]) -> None:
        order = [i for i in new_order if 0 <= i < len(self.clips)]
        self.clips = [self.clips[i] for i in order]
        self._update_timeline()

    def _update_timeline(self) -> None:
        self.total_duration = sum(float(c.duration) for c in self.clips)

    def to_edl(self) -> Dict:
        return {
            "timeline": {
                "fps": self.target_fps,
                "duration": self.total_duration,
                "clips": [asdict(c) for c in self.clips],
            }
        }

    def export_edl(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_edl(), f, indent=2)

    def import_edl(self, filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as f:
            edl_data = json.load(f)
        t = edl_data["timeline"]
        self.target_fps = float(t.get("fps", 30.0))
        self.clips = [VideoClip(**c) for c in t.get("clips", [])]
        self._update_timeline()

    def apply_viral_hook_trims(self, keywords: Optional[List[str]] = None,
                               pre_pad: float = 0.2,
                               post_pad: float = 4.0) -> None:
        """Apply viral hook trims based on keywords in transcripts"""
        if not keywords:
            return
            
        lowered = [k.lower() for k in keywords]
        for c in self.clips:
            t = (c.transcript or "").lower()
            if not t:
                continue
            hit = next((k for k in lowered if k in t), None)
            if not hit:
                continue
            new_start = max(c.start_time, c.start_time + 0.0 - pre_pad)
            new_end = min(c.end_time, new_start + post_pad)
            if new_end - new_start >= 0.25:
                c.start_time = new_start
                c.end_time = new_end
                c.duration = new_end - new_start
        self._update_timeline()


class FFmpegRenderer:
    """
    High-performance FFmpeg-based video renderer.
    Supports hard cuts, format conversion, and auto-captions.
    """
    
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.temp_dir = None

    def render_video(self,
                    output_path: str,
                    video_format: VideoFormat = VideoFormat.PORTRAIT,
                    quality: str = "high",
                    use_gpu: bool = False,
                    auto_captions: bool = False,
                    caption_style: Optional[Dict[str, str]] = None) -> None:
        """
        Render timeline to video using FFmpeg for maximum performance.
        
        Args:
            output_path: Output file path
            video_format: Target video format (VideoFormat enum)
            quality: Render quality (high, medium, low)
            use_gpu: Use GPU acceleration if available
            auto_captions: Whether to burn-in captions from transcripts
            caption_style: Custom caption styling options
        """
        if not self.timeline.clips:
            raise RuntimeError("Timeline has no clips to render.")

        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            
            try:
                # Step 1: Prepare video segments (with stream copy for speed)
                segment_files = self._prepare_segments()
                
                if not segment_files:
                    raise RuntimeError("No valid segments to render")
                
                # Step 2: Create concat file for FFmpeg
                concat_file = os.path.join(temp_dir, "concat_list.txt")
                self._create_concat_file(segment_files, concat_file)
                
                # Step 3: Build and execute FFmpeg command
                cmd = self._build_ffmpeg_command(
                    concat_file=concat_file,
                    output_path=output_path,
                    video_format=video_format,
                    quality=quality,
                    use_gpu=use_gpu,
                    auto_captions=auto_captions,
                    caption_style=caption_style or {}
                )
                
                # Execute FFmpeg
                self._execute_ffmpeg(cmd)
                
            except Exception as e:
                logger.error(f"FFmpeg rendering failed: {e}")
                raise

    def _prepare_segments(self) -> List[str]:
        """
        Prepare individual video segments using stream copy for maximum speed.
        Only re-encodes when absolutely necessary.
        """
        segment_files = []
        
        for i, clip in enumerate(self.timeline.clips):
            if not os.path.exists(clip.source_video):
                logger.warning(f"Source video not found: {clip.source_video}")
                continue
            
            segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
            
            # Use FFmpeg with stream copy for maximum speed
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(clip.start_time),
                "-i", clip.source_video,
                "-t", str(clip.duration),
                "-c", "copy",  # Stream copy - no re-encoding!
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",  # Generate timestamps
                segment_file
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                segment_files.append(segment_file)
                logger.debug(f"Created segment {i}: {segment_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create segment {i}: {e}")
                continue
        
        return segment_files

    def _create_concat_file(self, segment_files: List[str], concat_file: str) -> None:
        """Create FFmpeg concat file for seamless joining"""
        with open(concat_file, 'w') as f:
            for segment in segment_files:
                # Use absolute paths and escape for FFmpeg
                abs_path = os.path.abspath(segment)
                f.write(f"file '{abs_path}'\n")

    def _build_ffmpeg_command(self,
                             concat_file: str,
                             output_path: str,
                             video_format: VideoFormat,
                             quality: str,
                             use_gpu: bool,
                             auto_captions: bool,
                             caption_style: Dict[str, str]) -> List[str]:
        """Build optimized FFmpeg command for final render"""
        
        cmd = ["ffmpeg", "-y", "-loglevel", "info"]
        
        # Input: concatenated segments
        cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_file])
        
        # Video processing filters
        filters = []
        
        # Get target dimensions
        format_info = video_format.value
        width = format_info["width"]
        height = format_info["height"]
        
        # Scale and crop to target format (smart cropping)
        filters.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
        filters.append(f"crop={width}:{height}")
        
        # Add captions if requested
        if auto_captions:
            subtitle_filter = self._create_caption_filter(caption_style)
            if subtitle_filter:
                filters.append(subtitle_filter)
        
        # Apply video filters
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        
        # Codec selection
        if use_gpu:
            # NVIDIA GPU acceleration
            cmd.extend(["-c:v", "h264_nvenc"])
            cmd.extend(["-preset", "fast"])
            cmd.extend(["-rc", "vbr"])
            cmd.extend(["-cq", "22"])
        else:
            # CPU encoding
            cmd.extend(["-c:v", "libx264"])
            cmd.extend(["-preset", "fast"])
            cmd.extend(["-crf", "22"])
        
        # Audio settings
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        
        # Quality/bitrate settings
        bitrate_map = {
            "high": "8000k",
            "medium": "4000k", 
            "low": "2000k"
        }
        cmd.extend(["-b:v", bitrate_map.get(quality, "4000k")])
        
        # Optimization flags
        cmd.extend([
            "-movflags", "+faststart",  # Fast web playback
            "-pix_fmt", "yuv420p",      # Broad compatibility
            "-max_muxing_queue_size", "9999"  # Handle large files
        ])
        
        cmd.append(output_path)
        
        return cmd

    def _create_caption_filter(self, style: Dict[str, str]) -> Optional[str]:
        """Create FFmpeg subtitle filter from timeline transcripts"""
        if not any(clip.transcript for clip in self.timeline.clips):
            return None
        
        # Create SRT subtitle file
        srt_file = os.path.join(self.temp_dir, "captions.srt")
        self._create_srt_file(srt_file)
        
        if not os.path.exists(srt_file):
            return None
        
        # Build subtitle filter with styling
        font_name = style.get("font", "Arial-Bold")
        font_size = style.get("size", "60")
        font_color = style.get("color", "white")
        outline_color = style.get("outline", "black")
        outline_width = style.get("outline_width", "2")
        
        # Escape file path for FFmpeg
        escaped_path = srt_file.replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")
        
        subtitle_filter = (
            f"subtitles='{escaped_path}'"
            f":force_style='FontName={font_name},FontSize={font_size},"
            f"PrimaryColour=&H{self._color_to_hex(font_color)},"
            f"OutlineColour=&H{self._color_to_hex(outline_color)},"
            f"Outline={outline_width},Alignment=2'"
        )
        
        return subtitle_filter

    def _create_srt_file(self, srt_path: str) -> None:
        """Create SRT subtitle file from timeline clips"""
        subtitles = []
        current_time = 0.0
        subtitle_index = 1
        
        for clip in self.timeline.clips:
            if not clip.transcript or not clip.transcript.strip():
                current_time += clip.duration
                continue
            
            start_time = current_time
            end_time = current_time + clip.duration
            
            # Format for SRT
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            subtitles.append(f"{subtitle_index}")
            subtitles.append(f"{start_srt} --> {end_srt}")
            subtitles.append(clip.transcript.strip())
            subtitles.append("")
            
            current_time += clip.duration
            subtitle_index += 1
        
        # Write SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(subtitles))

    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _color_to_hex(self, color: str) -> str:
        """Convert color name to hex for FFmpeg"""
        color_map = {
            "white": "ffffff",
            "black": "000000",
            "red": "ff0000",
            "blue": "0000ff",
            "green": "00ff00",
            "yellow": "ffff00"
        }
        return color_map.get(color.lower(), "ffffff")

    def _execute_ffmpeg(self, cmd: List[str]) -> None:
        """Execute FFmpeg command with proper error handling"""
        logger.info(f"Executing FFmpeg: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("FFmpeg rendering completed successfully")
            else:
                logger.error(f"FFmpeg failed with return code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise RuntimeError(f"FFmpeg rendering failed")
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg rendering timed out")
            raise RuntimeError("FFmpeg rendering timed out after 1 hour")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"FFmpeg rendering failed: {e.stderr}")


# Legacy compatibility class
class VideoCompiler(FFmpegRenderer):
    """Compatibility wrapper for old VideoCompiler interface"""
    
    def render_with_moviepy(self, *args, **kwargs):
        """Legacy method - redirects to FFmpeg renderer"""
        logger.warning("render_with_moviepy is deprecated, using FFmpeg renderer instead")
        
        # Map old parameters to new ones
        output_path = kwargs.get('output_path')
        quality = kwargs.get('quality', 'high')
        use_gpu = kwargs.get('use_gpu', False)
        video_format = kwargs.get('video_format', VideoFormat.PORTRAIT)
        auto_captions = kwargs.get('auto_captions', False)
        
        return self.render_video(
            output_path=output_path,
            quality=quality,
            use_gpu=use_gpu,
            video_format=video_format,
            auto_captions=auto_captions
        )