from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

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
        logger.info(
            f"Optimized timeline: {len(self.clips)} clips, {self.total_duration:.1f}s"
        )

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
    """Professional-grade FFmpeg video renderer with advanced optimization"""

    def __init__(self, timeline: Timeline):
        self.timeline = timeline
        self.temp_dir = None
        self.gpu_available = self._check_gpu_acceleration()

    def _check_gpu_acceleration(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return "h264_nvenc" in result.stdout or "h264_amf" in result.stdout
        except:
            return False

    def render_video(
        self,
        output_path: str,
        video_format: VideoFormat = VideoFormat.PORTRAIT,
        quality: str = "high",
        use_gpu: Optional[bool] = None,
        custom_filters: Optional[List[str]] = None,
        audio_enhance: bool = True,
        color_correction: bool = True,
    ) -> None:
        """Render video with professional quality and advanced features"""
        if not self.timeline.clips:
            raise RuntimeError("Timeline has no clips to render.")

        # Auto-detect GPU usage if not specified
        if use_gpu is None:
            use_gpu = self.gpu_available

        logger.info(f"Starting professional render: {len(self.timeline.clips)} clips")
        logger.info(
            f"Target format: {video_format.value['name']} ({video_format.value['ratio']})"
        )
        logger.info(
            f"Quality: {quality}, GPU: {use_gpu}, Audio enhance: {audio_enhance}"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir

            try:
                self._render_final_video(
                    output_path,
                    video_format,
                    quality,
                    use_gpu,
                    custom_filters,
                    audio_enhance,
                    color_correction,
                )

                # Verify output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Professional render completed: {output_path}")
                    self._log_render_stats(output_path)
                else:
                    raise RuntimeError("Render produced invalid output")

            except Exception as e:
                logger.error(f"Professional rendering failed: {e}")
                raise

    def _log_render_stats(self, output_path: str) -> None:
        """Log rendering statistics"""
        try:
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"Output file size: {file_size:.1f} MB")
            logger.info(f"Timeline duration: {self.timeline.total_duration:.2f}s")
            logger.info(
                f"Bitrate estimate: {(file_size * 8) / self.timeline.total_duration:.1f} Mbps"
            )
        except Exception as e:
            logger.debug(f"Could not calculate render stats: {e}")

    def _render_final_video(
        self,
        output_path: str,
        video_format: VideoFormat,
        quality: str,
        use_gpu: bool,
        custom_filters: Optional[List[str]] = None,
        audio_enhance: bool = True,
        color_correction: bool = True,
    ) -> None:
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
            use_gpu=use_gpu,
            custom_filters=custom_filters,
            audio_enhance=audio_enhance,
            color_correction=color_correction,
        )

        self._execute_ffmpeg(cmd)

    def _prepare_segments(self) -> List[str]:
        """Prepare video segments with professional precision and preprocessing"""
        segment_files = []

        for i, clip in enumerate(self.timeline.clips):
            if not os.path.exists(clip.source_video):
                logger.warning(f"Source video not found: {clip.source_video}")
                continue

            segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")

            # Professional segment extraction with stabilization
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                f"{clip.start_time:.6f}",  # Microsecond precision
                "-i",
                clip.source_video,
                "-t",
                f"{clip.duration:.6f}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",  # Better quality than ultrafast
                "-crf",
                "15",  # Higher quality constant rate factor
                "-c:a",
                "aac",
                "-b:a",
                "192k",  # High quality audio bitrate
                "-ar",
                "48000",  # Professional sample rate
                "-avoid_negative_ts",
                "make_zero",
                "-fflags",
                "+genpts+igndts",  # Better timestamp handling
                "-max_muxing_queue_size",
                "9999",
                "-vsync",
                "2",  # Variable frame rate handling
                segment_file,
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)

                # Verify segment was created successfully
                if (
                    os.path.exists(segment_file)
                    and os.path.getsize(segment_file) > 1024
                ):
                    segment_files.append(segment_file)
                    logger.debug(
                        f"Created professional segment {i}: {clip.duration:.3f}s"
                    )
                else:
                    logger.error(f"Segment {i} created but appears invalid")

            except subprocess.CalledProcessError as e:
                logger.error(
                    f"Failed to create segment {i}: {e.stderr if e.stderr else str(e)}"
                )
                continue

        logger.info(
            f"Prepared {len(segment_files)}/{len(self.timeline.clips)} segments successfully"
        )
        return segment_files

    def _create_concat_file(self, segment_files: List[str], concat_file: str) -> None:
        """Create FFmpeg concat file"""
        with open(concat_file, "w", encoding="utf-8") as f:
            for segment in segment_files:
                abs_path = os.path.abspath(segment).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

    def _build_ffmpeg_command(
        self,
        concat_file: str,
        output_path: str,
        video_format: VideoFormat,
        quality: str,
        use_gpu: bool,
        custom_filters: Optional[List[str]] = None,
        audio_enhance: bool = True,
        color_correction: bool = True,
    ) -> List[str]:
        """Build professional FFmpeg command with advanced optimization"""

        cmd = ["ffmpeg", "-y", "-loglevel", "info"]

        # Input with optimized buffering
        cmd.extend(["-f", "concat", "-safe", "0", "-i", concat_file])

        # Video processing
        format_info = video_format.value
        width = format_info["width"]
        height = format_info["height"]

        # Professional video filter chain
        filters = []

        # Base scaling and cropping with high-quality algorithms
        filters.extend(
            [
                f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos",
                f"crop={width}:{height}",
            ]
        )

        # Color correction and enhancement
        if color_correction:
            filters.extend(
                [
                    "eq=contrast=1.05:brightness=0.02:saturation=1.1",  # Subtle color enhancement
                    "hue=s=1.05",  # Slight saturation boost
                    "curves=all='0/0 0.5/0.52 1/1'",  # Gentle S-curve for contrast
                ]
            )

        # Professional sharpening
        filters.append(
            "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.8:chroma_msize_x=3:chroma_msize_y=3:chroma_amount=0.4"
        )

        # Noise reduction for cleaner output
        filters.append("hqdn3d=4:3:6:4.5")

        # Custom filters if provided
        if custom_filters:
            filters.extend(custom_filters)

        # Frame rate stabilization
        filters.append("fps=fps=30:round=near")

        cmd.extend(["-vf", ",".join(filters)])

        # Professional video encoding
        if use_gpu:
            # NVIDIA NVENC encoding
            cmd.extend(["-c:v", "h264_nvenc"])
            cmd.extend(["-gpu", "0"])  # Use first GPU

            if quality == "ultra":
                cmd.extend(
                    [
                        "-preset",
                        "p3",  # Ultra high quality preset
                        "-rc",
                        "constqp",  # Constant quality mode
                        "-qp",
                        "14",  # Ultra high quality
                        "-b:v",
                        "15M",  # Target bitrate
                        "-maxrate",
                        "20M",  # Maximum bitrate
                        "-bufsize",
                        "25M",  # Buffer size
                        "-profile:v",
                        "high",  # H.264 High profile
                        "-level",
                        "4.1",  # H.264 level
                        "-rc-lookahead",
                        "40",  # More lookahead frames
                    ]
                )
            elif quality == "high":
                cmd.extend(
                    [
                        "-preset",
                        "p4",  # High quality preset
                        "-rc",
                        "constqp",  # Constant quality mode
                        "-qp",
                        "16",  # High quality
                        "-b:v",
                        "8M",  # Target bitrate
                        "-maxrate",
                        "12M",  # Maximum bitrate
                        "-bufsize",
                        "16M",  # Buffer size
                        "-profile:v",
                        "high",  # H.264 High profile
                        "-level",
                        "4.1",  # H.264 level
                        "-rc-lookahead",
                        "32",  # Lookahead frames
                    ]
                )
            elif quality == "medium":
                cmd.extend(
                    [
                        "-preset",
                        "p5",
                        "-rc",
                        "vbr",
                        "-cq",
                        "20",
                        "-b:v",
                        "5M",
                        "-maxrate",
                        "7M",
                        "-bufsize",
                        "10M",
                    ]
                )
            else:
                cmd.extend(
                    [
                        "-preset",
                        "p6",
                        "-rc",
                        "vbr",
                        "-cq",
                        "24",
                        "-b:v",
                        "3M",
                        "-maxrate",
                        "4M",
                    ]
                )
        else:
            # CPU encoding with x264
            cmd.extend(["-c:v", "libx264"])

            if quality == "ultra":
                cmd.extend(
                    [
                        "-preset",
                        "veryslow",  # Ultimate quality preset
                        "-crf",
                        "10",  # Ultra high quality
                        "-profile:v",
                        "high",
                        "-level",
                        "4.1",
                        "-x264-params",
                        "aq-mode=3:aq-strength=0.8:deblock=1,1:ref=16:bframes=16:b-adapt=2:direct=auto:me=umh:subme=11:merange=64:trellis=2:psy-rd=1.0,0.2:no-fast-pskip=1:no-dct-decimate=1:deadzone-inter=21:deadzone-intra=11",
                    ]
                )
            elif quality == "high":
                cmd.extend(
                    [
                        "-preset",
                        "veryslow",  # Highest quality preset
                        "-crf",
                        "12",  # Very high quality
                        "-profile:v",
                        "high",
                        "-level",
                        "4.1",
                        "-x264-params",
                        "aq-mode=3:aq-strength=0.8:deblock=1,1:ref=8:bframes=8:b-adapt=2:direct=auto:me=umh:subme=11:merange=32:trellis=2:psy-rd=1.0,0.15:no-fast-pskip=1:no-dct-decimate=1",
                    ]
                )
            elif quality == "medium":
                cmd.extend(
                    [
                        "-preset",
                        "slow",
                        "-crf",
                        "16",
                        "-profile:v",
                        "high",
                        "-x264-params",
                        "aq-mode=2:ref=5:bframes=5:b-adapt=2:me=umh:subme=8:trellis=1:psy-rd=1.0,0.1",
                    ]
                )
            else:  # low quality
                cmd.extend(["-preset", "medium", "-crf", "20", "-profile:v", "high", "-x264-params", "ref=3:bframes=3"])

        # Professional audio processing
        audio_filters = []

        if audio_enhance:
            # Audio enhancement pipeline
            audio_filters.extend(
                [
                    "highpass=f=80",  # Remove low-frequency noise
                    "lowpass=f=15000",  # Remove high-frequency noise
                    "loudnorm=I=-16:TP=-1.5:LRA=11",  # Professional loudness normalization
                    "acompressor=threshold=0.089:ratio=9:attack=0.3:release=0.8",  # Gentle compression
                    "adeclick",  # Remove clicks
                    "adeclip",  # Remove clipping
                ]
            )

        if audio_filters:
            cmd.extend(["-af", ",".join(audio_filters)])

        # Professional audio encoding with quality-based bitrate
        audio_bitrate = {
            "ultra": "320k",
            "high": "256k",
            "medium": "192k",
            "low": "128k"
        }.get(quality, "256k")

        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,  # Quality-based audio bitrate
                "-ar",
                "48000",  # Professional sample rate
                "-ac",
                "2",  # Stereo
                "-aac_coder",
                "twoloop",  # Best AAC encoder
                "-profile:a",
                "aac_low",  # AAC-LC profile
            ]
        )

        # Professional output options
        cmd.extend(
            [
                "-movflags",
                "+faststart+use_metadata_tags",  # Web optimization
                "-pix_fmt",
                "yuv420p",  # Universal compatibility
                "-colorspace",
                "bt709",  # HD color space
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-color_range",
                "tv",  # Broadcast safe
                "-max_muxing_queue_size",
                "9999",
                "-vsync",
                "cfr",  # Constant frame rate
                "-async",
                "1",  # Audio sync
                "-fflags",
                "+genpts+igndts",  # Better timestamp handling
                "-metadata",
                "comment=Generated by AI Co-Creator Pro",
                "-metadata",
                "encoder=AI Co-Creator Professional Renderer v2.0",
            ]
        )

        cmd.append(output_path)

        logger.debug(f"FFmpeg command built: {len(cmd)} parameters")
        return cmd

    def _execute_ffmpeg(self, cmd: List[str]) -> None:
        """Execute FFmpeg command with error handling"""
        logger.info("Executing FFmpeg command...")

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=3600
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
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


class ProfessionalRenderer:
    """Professional rendering pipeline with advanced features"""

    def __init__(self):
        self.gpu_available = self._check_gpu_acceleration()

    def _check_gpu_acceleration(self) -> bool:
        """Check available GPU acceleration"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            encoders = result.stdout
            return any(
                enc in encoders
                for enc in ["h264_nvenc", "h264_amf", "h264_videotoolbox"]
            )
        except:
            return False

    def create_professional_timeline(
        self, clips_data: List[Dict[str, Any]]
    ) -> Timeline:
        """Create optimized timeline from clips data"""
        timeline = Timeline(target_fps=30.0)

        for clip_data in clips_data:
            clip = VideoClip(
                source_video=clip_data["source_video"],
                start_time=float(clip_data["start_time"]),
                end_time=float(clip_data["end_time"]),
                duration=float(clip_data["end_time"]) - float(clip_data["start_time"]),
                metadata=clip_data.get("metadata", {}),
            )
            timeline.add_clip(clip)

        return timeline

    def render_with_presets(
        self,
        timeline: Timeline,
        output_dir: str,
        base_name: str,
        presets: List[str] = None,
    ) -> Dict[str, str]:
        """Render video with multiple professional presets"""

        if presets is None:
            presets = ["youtube_shorts", "tiktok", "instagram_reels"]

        results = {}
        renderer = VideoRenderer(timeline)

        preset_configs = {
            "youtube_shorts": {
                "format": VideoFormat.PORTRAIT,
                "quality": "high",
                "max_duration": 60.0,
                "color_correction": True,
                "audio_enhance": True,
            },
            "tiktok": {
                "format": VideoFormat.PORTRAIT,
                "quality": "high",
                "max_duration": 60.0,
                "color_correction": True,
                "audio_enhance": True,
                "custom_filters": [
                    "eq=saturation=1.2:contrast=1.1"
                ],  # More vivid for TikTok
            },
            "instagram_reels": {
                "format": VideoFormat.PORTRAIT,
                "quality": "high",
                "max_duration": 90.0,
                "color_correction": True,
                "audio_enhance": True,
                "custom_filters": [
                    "eq=brightness=0.05:saturation=1.15"
                ],  # Instagram aesthetic
            },
            "linkedin": {
                "format": VideoFormat.LANDSCAPE,
                "quality": "high",
                "max_duration": 300.0,
                "color_correction": False,  # Professional, natural colors
                "audio_enhance": True,
            },
            "twitter": {
                "format": VideoFormat.LANDSCAPE,
                "quality": "medium",
                "max_duration": 140.0,
                "color_correction": True,
                "audio_enhance": True,
            },
        }

        os.makedirs(output_dir, exist_ok=True)

        for preset in presets:
            if preset not in preset_configs:
                logger.warning(f"Unknown preset: {preset}, skipping")
                continue

            config = preset_configs[preset]

            # Optimize timeline for preset duration
            preset_timeline = Timeline(timeline.target_fps)
            preset_timeline.clips = timeline.clips.copy()
            preset_timeline._update_timeline()

            if preset_timeline.total_duration > config["max_duration"]:
                preset_timeline.optimize_for_duration(config["max_duration"])

            # Generate output path
            format_suffix = config["format"].value["name"]
            output_path = os.path.join(
                output_dir, f"{base_name}_{preset}_{format_suffix}.mp4"
            )

            try:
                logger.info(f"Rendering {preset} preset...")

                preset_renderer = VideoRenderer(preset_timeline)
                preset_renderer.render_video(
                    output_path=output_path,
                    video_format=config["format"],
                    quality=config["quality"],
                    custom_filters=config.get("custom_filters"),
                    audio_enhance=config["audio_enhance"],
                    color_correction=config["color_correction"],
                )

                results[preset] = output_path
                logger.info(f"✅ {preset} preset completed: {output_path}")

            except Exception as e:
                logger.error(f"❌ {preset} preset failed: {e}")
                results[preset] = f"Error: {str(e)}"

        return results

    def create_comparison_video(
        self, original_video: str, rendered_video: str, output_path: str
    ) -> str:
        """Create side-by-side comparison video"""

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            original_video,
            "-i",
            rendered_video,
            "-filter_complex",
            "[0:v]scale=960:540,pad=1920:540[left];"
            "[1:v]scale=960:540[right];"
            "[left][right]hstack=inputs=2[v];"
            "[0:a][1:a]amix=inputs=2[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            output_path,
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Comparison video created: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Comparison video creation failed: {e}")
            raise


def get_quality_settings() -> Dict[str, Dict[str, Any]]:
    """Get available quality settings with professional options"""
    return {
        "ultra": {
            "description": "Ultra high quality (film/broadcast)",
            "crf_cpu": 12,
            "crf_gpu": 14,
            "preset_cpu": "veryslow",
            "bitrate": "15M",
            "notes": "Best possible quality, very slow render",
        },
        "high": {
            "description": "High quality (professional)",
            "crf_cpu": 12,
            "crf_gpu": 16,
            "preset_cpu": "veryslow",
            "bitrate": "10M",
            "notes": "Professional quality, slower render",
        },
        "medium": {
            "description": "Balanced quality and speed",
            "crf_cpu": 16,
            "crf_gpu": 20,
            "preset_cpu": "slow",
            "bitrate": "6M",
            "notes": "Good quality, reasonable render time",
        },
        "fast": {
            "description": "Faster render, good quality",
            "crf_cpu": 22,
            "crf_gpu": 24,
            "preset_cpu": "fast",
            "bitrate": "3M",
            "notes": "Quick render for previews",
        },
        "preview": {
            "description": "Very fast preview quality",
            "crf_cpu": 28,
            "crf_gpu": 30,
            "preset_cpu": "ultrafast",
            "bitrate": "1M",
            "notes": "Draft quality for quick previews",
        },
    }


def get_platform_presets() -> Dict[str, Dict[str, Any]]:
    """Get platform-specific rendering presets"""
    return {
        "youtube_shorts": {
            "name": "YouTube Shorts",
            "format": "portrait",
            "resolution": "1080x1920",
            "max_duration": 60,
            "recommended_quality": "high",
            "features": ["color_correction", "audio_enhance", "stabilization"],
        },
        "tiktok": {
            "name": "TikTok",
            "format": "portrait",
            "resolution": "1080x1920",
            "max_duration": 60,
            "recommended_quality": "high",
            "features": ["vivid_colors", "audio_enhance", "trending_effects"],
        },
        "instagram_reels": {
            "name": "Instagram Reels",
            "format": "portrait",
            "resolution": "1080x1920",
            "max_duration": 90,
            "recommended_quality": "high",
            "features": ["aesthetic_colors", "audio_enhance", "smooth_transitions"],
        },
        "instagram_story": {
            "name": "Instagram Story",
            "format": "portrait",
            "resolution": "1080x1920",
            "max_duration": 15,
            "recommended_quality": "medium",
            "features": ["quick_render", "mobile_optimized"],
        },
        "linkedin": {
            "name": "LinkedIn Video",
            "format": "landscape",
            "resolution": "1920x1080",
            "max_duration": 300,
            "recommended_quality": "high",
            "features": ["professional_colors", "clear_audio", "corporate_safe"],
        },
        "twitter": {
            "name": "Twitter Video",
            "format": "landscape",
            "resolution": "1280x720",
            "max_duration": 140,
            "recommended_quality": "medium",
            "features": ["balanced_quality", "fast_render"],
        },
        "facebook": {
            "name": "Facebook Video",
            "format": "landscape",
            "resolution": "1920x1080",
            "max_duration": 240,
            "recommended_quality": "high",
            "features": ["social_optimized", "auto_captions_ready"],
        },
    }
