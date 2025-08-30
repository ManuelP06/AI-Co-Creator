from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Iterable, Tuple, Any

logger = logging.getLogger(__name__)


class VideoFormat(Enum):
    """Supported video output formats optimized for viral platforms"""
    PORTRAIT = {"width": 1080, "height": 1920, "name": "9x16", "ratio": "9:16", "platform": "TikTok/Reels"}
    LANDSCAPE = {"width": 1920, "height": 1080, "name": "16x9", "ratio": "16:9", "platform": "YouTube"}
    SQUARE = {"width": 1080, "height": 1080, "name": "1x1", "ratio": "1:1", "platform": "Instagram"}


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
    viral_score: float = 0.0


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

    def optimize_for_viral(self, max_duration: float = 59.0) -> None:
        """Optimize timeline for viral content"""
        self.clips.sort(key=lambda c: c.viral_score, reverse=True)
        
        current_duration = 0.0
        optimized_clips = []
        
        for clip in self.clips:
            if current_duration + clip.duration <= max_duration:
                optimized_clips.append(clip)
                current_duration += clip.duration
            elif max_duration - current_duration > 0.5:  
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


class FFmpegRenderer:
    """
    High-performance FFmpeg-based video renderer with FIXED auto-captions.
    Optimized for viral short-form content creation.
    """
    
    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.temp_dir = None

    def render_video(self,
                    output_path: str,
                    video_format: VideoFormat = VideoFormat.PORTRAIT,
                    quality: str = "high",
                    use_gpu: bool = False,
                    auto_captions: bool = True,  
                    caption_style: Optional[Dict[str, str]] = None) -> None:
        """
        Render timeline to video using FFmpeg with WORKING auto-captions.
        """
        if not self.timeline.clips:
            raise RuntimeError("Timeline has no clips to render.")

        self.timeline.optimize_for_viral()

        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            
            try:
                logger.info(f"Rendering {len(self.timeline.clips)} clips to {video_format.value['name']} format")
                
                intermediate_path = os.path.join(temp_dir, "video_no_subs.mp4")
                self._render_base_video(intermediate_path, video_format, quality, use_gpu)
                
                if auto_captions:
                    self._add_captions_to_video(intermediate_path, output_path, caption_style or {})
                else:
                    import shutil
                    shutil.copy2(intermediate_path, output_path)
                
                logger.info(f"Video rendered successfully: {output_path}")
                
            except Exception as e:
                logger.error(f"FFmpeg rendering failed: {e}")
                raise

    def _render_base_video(self, output_path: str, video_format: VideoFormat, quality: str, use_gpu: bool) -> None:
        """Render base video without captions using concatenation"""
        
        segment_files = self._prepare_segments()
        
        if not segment_files:
            raise RuntimeError("No valid segments to render")
        
        concat_file = os.path.join(self.temp_dir, "concat_list.txt")
        self._create_concat_file(segment_files, concat_file)
        
        cmd = self._build_base_ffmpeg_command(
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
                logger.error(f"Failed to create segment {i}: {e.stderr if e.stderr else e}")
                continue
        
        return segment_files

    def _create_concat_file(self, segment_files: List[str], concat_file: str) -> None:
        """Create FFmpeg concat file for seamless joining"""
        with open(concat_file, 'w', encoding='utf-8') as f:
            for segment in segment_files:
                abs_path = os.path.abspath(segment).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

    def _build_base_ffmpeg_command(self,
                                  concat_file: str,
                                  output_path: str,
                                  video_format: VideoFormat,
                                  quality: str,
                                  use_gpu: bool) -> List[str]:
        """Build FFmpeg command for base video without captions"""
        
        cmd = ["ffmpeg", "-y", "-loglevel", "info"]
        
        cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_file])
        
        format_info = video_format.value
        width = format_info["width"]
        height = format_info["height"]
        
        filters = []
        
        filters.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
        filters.append(f"crop={width}:{height}")
        
        filters.append("unsharp=5:5:1.0:5:5:0.0")
        
        cmd.extend(["-vf", ",".join(filters)])
        
        if use_gpu:
            cmd.extend(["-c:v", "h264_nvenc"])
            cmd.extend(["-preset", "p4"])  
            cmd.extend(["-rc", "vbr"])
            cmd.extend(["-cq", "20"])  
            cmd.extend(["-b:v", "0"]) 
        else:
            cmd.extend(["-c:v", "libx264"])
            cmd.extend(["-preset", "medium"]) 
            cmd.extend(["-crf", "20"])  
        
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "44100"])
        

        cmd.extend([
            "-movflags", "+faststart",  
            "-pix_fmt", "yuv420p",      
            "-max_muxing_queue_size", "9999",
            "-maxrate", "8000k",        
            "-bufsize", "16000k"        
        ])
        
        cmd.append(output_path)
        return cmd

    def _add_captions_to_video(self, input_path: str, output_path: str, caption_style: Dict[str, str]) -> None:
        """Add captions to video using ASS subtitles for better control"""
        
        ass_file = os.path.join(self.temp_dir, "captions.ass")
        self._create_ass_file(ass_file, caption_style)
        
        if not os.path.exists(ass_file):
            logger.warning("No captions file created, copying video without captions")
            import shutil
            shutil.copy2(input_path, output_path)
            return
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "info",
            "-i", input_path,
            "-vf", f"ass='{ass_file.replace(chr(92), chr(47))}'", 
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "copy", 
            "-movflags", "+faststart",
            output_path
        ]
        
        try:
            self._execute_ffmpeg(cmd)
            logger.info("Captions added successfully")
        except Exception as e:
            logger.error(f"Failed to add captions: {e}")
            import shutil
            shutil.copy2(input_path, output_path)

    def _create_ass_file(self, ass_path: str, style: Dict[str, str]) -> None:
        """Create ASS subtitle file with viral-optimized styling"""
        
        if not any(clip.transcript.strip() for clip in self.timeline.clips):
            logger.warning("No transcripts found for captions")
            return
        
        font_name = style.get("font", "Arial Bold")
        font_size = int(style.get("size", "32"))  
        font_color = self._color_to_ass_hex(style.get("color", "white"))
        outline_color = self._color_to_ass_hex(style.get("outline", "black"))
        outline_width = int(style.get("outline_width", "3"))
        
        ass_header = f"""[Script Info]
Title: Auto-Generated Captions
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{font_color},&H00000000,{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,{outline_width},0,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

        events = []
        current_time = 0.0
        
        for clip in self.timeline.clips:
            if not clip.transcript or not clip.transcript.strip():
                current_time += clip.duration
                continue
            
            start_time = current_time
            end_time = current_time + clip.duration
            
            start_ass = self._seconds_to_ass_time(start_time)
            end_ass = self._seconds_to_ass_time(end_time)
            
            text = self._optimize_caption_text(clip.transcript.strip())
            
            events.append(f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{text}")
            current_time += clip.duration

        with open(ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_header)
            f.write('\n'.join(events))
        
        logger.info(f"Created ASS subtitle file with {len(events)} captions")

    def _optimize_caption_text(self, text: str) -> str:
        """Optimize caption text for viral content readability"""
        filler_words = r'\b(um|uh|like|you know|actually|basically|literally|so)\b'
        text = re.sub(filler_words, '', text, flags=re.IGNORECASE)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        viral_words = ['amazing', 'incredible', 'shocking', 'secret', 'revealed']
        for word in viral_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            replacement = f"{{\\b1}}{word}{{\\b0}}"  
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        if len(text) > 100:
            text = text[:97] + "..."
        
        return text

    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.CC)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def _color_to_ass_hex(self, color: str) -> str:
        """Convert color name to ASS hex format (&Hbbggrr)"""
        color_map = {
            "white": "&H00FFFFFF",
            "black": "&H00000000", 
            "red": "&H000000FF",
            "blue": "&H00FF0000",
            "green": "&H0000FF00",
            "yellow": "&H0000FFFF",
            "orange": "&H0000A5FF",
            "purple": "&H00FF00FF"
        }
        return color_map.get(color.lower(), "&H00FFFFFF")

    def _execute_ffmpeg(self, cmd: List[str]) -> None:
        """Execute FFmpeg command with enhanced error handling"""
        logger.info(f"Executing FFmpeg command...")
        logger.debug(f"Full command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=1800  
            )
            
            if result.stderr:
                logger.debug(f"FFmpeg warnings: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg rendering timed out")
            raise RuntimeError("FFmpeg rendering timed out after 30 minutes")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error (code {e.returncode}): {e.stderr}")
            raise RuntimeError(f"FFmpeg rendering failed: {e.stderr}")


VIRAL_CAPTION_STYLES = {
    "tiktok": {
        "font": "Arial Bold",
        "size": "36",
        "color": "white", 
        "outline": "black",
        "outline_width": "3"
    },
    "instagram": {
        "font": "Helvetica Bold",
        "size": "32",
        "color": "white",
        "outline": "black", 
        "outline_width": "2"
    },
    "youtube": {
        "font": "Roboto Bold",
        "size": "28",
        "color": "yellow",
        "outline": "black",
        "outline_width": "2"
    },
    "professional": {
        "font": "Arial",
        "size": "24",
        "color": "white",
        "outline": "black",
        "outline_width": "1"
    }
}


class ViralVideoRenderer(FFmpegRenderer):
    """Specialized renderer for viral short-form content"""
    
    def __init__(self, timeline: Timeline):
        super().__init__(timeline)
        
    def render_viral_clip(self,
                         output_path: str,
                         platform: str = "tiktok",
                         quality: str = "high",
                         use_gpu: bool = False) -> None:
        """Render optimized for specific viral platform"""
        
        platform_settings = {
            "tiktok": {
                "format": VideoFormat.PORTRAIT,
                "caption_style": VIRAL_CAPTION_STYLES["tiktok"],
                "max_duration": 60.0
            },
            "instagram": {
                "format": VideoFormat.PORTRAIT,
                "caption_style": VIRAL_CAPTION_STYLES["instagram"], 
                "max_duration": 60.0
            },
            "youtube": {
                "format": VideoFormat.PORTRAIT,
                "caption_style": VIRAL_CAPTION_STYLES["youtube"],
                "max_duration": 60.0
            },
        }
        
        settings = platform_settings.get(platform, platform_settings["tiktok"])
        
        self.timeline.optimize_for_viral(settings["max_duration"])
        
        self.render_video(
            output_path=output_path,
            video_format=settings["format"],
            quality=quality,
            use_gpu=use_gpu,
            auto_captions=True,
            caption_style=settings["caption_style"]
        )


class VideoCompiler(ViralVideoRenderer):
    """Enhanced compatibility wrapper with viral optimization"""
    
    def render_with_moviepy(self, *args, **kwargs):
        """Legacy method - redirects to optimized viral renderer"""
        logger.warning("render_with_moviepy is deprecated, using ViralVideoRenderer instead")
        
        output_path = kwargs.get('output_path')
        quality = kwargs.get('quality', 'high')
        use_gpu = kwargs.get('use_gpu', False)
        video_format = kwargs.get('video_format', VideoFormat.PORTRAIT)
        auto_captions = kwargs.get('auto_captions', True)
        caption_style = kwargs.get('caption_style', VIRAL_CAPTION_STYLES["tiktok"])
        
        return self.render_video(
            output_path=output_path,
            quality=quality,
            use_gpu=use_gpu,
            video_format=video_format,
            auto_captions=auto_captions,
            caption_style=caption_style
        )


def check_ffmpeg_capabilities() -> Dict[str, Any]:
    """Enhanced FFmpeg capability detection"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode != 0:
            return {"available": False, "error": "FFmpeg not working"}
        
        output = result.stdout.lower()
        
        capabilities = {
            "available": True,
            "version": result.stdout.split('\n')[0],
            "codecs": {
                "h264": "libx264" in output,
                "h264_nvenc": "h264_nvenc" in output,
                "aac": "aac" in output
            },
            "filters": {
                "scale": "scale" in output,
                "crop": "crop" in output, 
                "ass": "ass" in output or "subtitles" in output,
                "unsharp": "unsharp" in output
            },
            "formats": {
                "mp4": "mp4" in output,
                "concat": True  
            }
        }
        
        gpu_check = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if gpu_check.returncode == 0:
            encoders = gpu_check.stdout.lower()
            capabilities["gpu"] = {
                "nvenc_available": "h264_nvenc" in encoders,
                "vaapi_available": "h264_vaapi" in encoders,
                "qsv_available": "h264_qsv" in encoders
            }
        
        return capabilities
        
    except Exception as e:
        logger.error(f"FFmpeg capability check failed: {e}")
        return {"available": False, "error": str(e)}


def optimize_ffmpeg_for_viral() -> Dict[str, str]:
    """Return optimized FFmpeg flags for viral content"""
    return {
        "video_codec": "libx264",
        "preset": "medium", 
        "crf": "20",  
        "pixel_format": "yuv420p",
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "max_bitrate": "8000k",
        "buffer_size": "16000k",
        "movflags": "+faststart",
        "keyframe_interval": "2",  
    }