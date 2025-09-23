import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.services.transcription import (TranscriptionSegment, detect_language,
                                        extract_audio_from_video,
                                        transcribe_with_timestamps)


@dataclass
class CaptionStyle:
    font_family: str = "Arial"
    font_size: int = 24
    font_color: str = "white"
    background_color: Optional[str] = "black"
    background_opacity: float = 0.7
    position: str = "bottom"  # bottom, top, center
    max_width_percent: int = 80
    line_height: float = 1.2
    border_width: int = 0
    border_color: str = "black"
    shadow: bool = True
    shadow_color: str = "black"
    shadow_offset: Tuple[int, int] = (2, 2)


@dataclass
class CaptionSegment:
    start_time: float
    end_time: float
    text: str
    position_y: Optional[float] = None
    style_override: Optional[CaptionStyle] = None


class AutoCaptionsGenerator:
    """Auto-captions generator for professional video content."""

    def __init__(self):
        self.default_style = CaptionStyle()

    def create_smart_captions(
        self,
        video_path: str,
        language: Optional[str] = None,
        max_chars_per_line: int = 40,
        max_lines: int = 2,
        min_duration: float = 0.5,
        style: Optional[CaptionStyle] = None,
    ) -> List[CaptionSegment]:
        """
        Create intelligent captions with optimal timing and formatting.
        """
        print(f"Generating smart captions for video: {video_path}")

        # Extract and transcribe audio
        audio_path = extract_audio_from_video(video_path)

        try:
            # Auto-detect language if not specified
            if language is None:
                language = detect_language(audio_path)
                print(f"Detected language: {language}")

            # Get detailed transcription with timestamps
            segments = transcribe_with_timestamps(audio_path, language)

            if not segments:
                print("No transcription segments found")
                return []

            # Process and optimize segments
            caption_segments = self._optimize_caption_segments(
                segments, max_chars_per_line, max_lines, min_duration
            )

            # Apply intelligent text formatting
            for segment in caption_segments:
                segment.text = self._format_caption_text(segment.text, language)

            print(f"Generated {len(caption_segments)} caption segments")
            return caption_segments

        finally:
            # Clean up temporary audio file
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    def _optimize_caption_segments(
        self,
        segments: List[TranscriptionSegment],
        max_chars_per_line: int,
        max_lines: int,
        min_duration: float,
    ) -> List[CaptionSegment]:
        """Optimize caption timing and text formatting."""

        caption_segments = []

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            duration = segment.end - segment.start

            # Split long text into multiple captions
            if len(text) > max_chars_per_line * max_lines:
                sub_segments = self._split_long_text(
                    text, segment.start, segment.end, max_chars_per_line, max_lines
                )
                caption_segments.extend(sub_segments)
            else:
                # Format text for optimal readability
                formatted_text = self._format_for_display(text, max_chars_per_line)

                # Ensure minimum duration
                if duration < min_duration:
                    # Extend end time slightly
                    end_time = segment.start + min_duration
                else:
                    end_time = segment.end

                caption_segments.append(
                    CaptionSegment(
                        start_time=segment.start, end_time=end_time, text=formatted_text
                    )
                )

        # Post-process to avoid overlaps and optimize timing
        return self._resolve_timing_conflicts(caption_segments)

    def _split_long_text(
        self,
        text: str,
        start_time: float,
        end_time: float,
        max_chars_per_line: int,
        max_lines: int,
    ) -> List[CaptionSegment]:
        """Split long text into multiple timed segments."""

        words = text.split()
        segments = []

        # Calculate approximate reading speed (words per second)
        duration = end_time - start_time
        words_per_second = len(words) / duration if duration > 0 else 2

        chunk_size = max_chars_per_line * max_lines
        chunks = []
        current_chunk = ""

        for word in words:
            # Check if adding this word would exceed chunk size
            test_chunk = current_chunk + (" " if current_chunk else "") + word
            if len(test_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = word
            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Distribute timing across chunks
        total_chars = sum(len(chunk) for chunk in chunks)
        current_time = start_time

        for chunk in chunks:
            # Calculate duration based on character count
            char_ratio = (
                len(chunk) / total_chars if total_chars > 0 else 1.0 / len(chunks)
            )
            chunk_duration = duration * char_ratio

            # Ensure minimum duration
            chunk_duration = max(chunk_duration, 1.0)

            formatted_chunk = self._format_for_display(chunk, max_chars_per_line)

            segments.append(
                CaptionSegment(
                    start_time=current_time,
                    end_time=current_time + chunk_duration,
                    text=formatted_chunk,
                )
            )

            current_time += chunk_duration

        return segments

    def _format_for_display(self, text: str, max_chars_per_line: int) -> str:
        """Format text for optimal display with line breaks."""

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word

            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return "\n".join(lines)

    def _format_caption_text(self, text: str, language: str) -> str:
        """Apply language-specific formatting and improvements."""

        # Remove redundant whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Language-specific improvements
        if language == "en":
            # English-specific formatting
            text = self._format_english_text(text)
        elif language == "de":
            # German-specific formatting
            text = self._format_german_text(text)
        elif language == "es":
            # Spanish-specific formatting
            text = self._format_spanish_text(text)

        # General improvements
        text = self._apply_general_formatting(text)

        return text

    def _format_english_text(self, text: str) -> str:
        """English-specific text formatting."""
        # Fix common transcription errors
        corrections = {
            r"\buh\b": "",
            r"\bum\b": "",
            r"\byou know\b": "",
            r"\blike\b(?=\s+like)": "",  # Remove repeated "like"
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def _format_german_text(self, text: str) -> str:
        """German-specific text formatting."""
        # Fix common German transcription issues
        corrections = {
            r"\bähm\b": "",
            r"\böhm\b": "",
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def _format_spanish_text(self, text: str) -> str:
        """Spanish-specific text formatting."""
        # Fix common Spanish transcription issues
        corrections = {
            r"\beeh\b": "",
            r"\beste\b(?=\s+este)": "",  # Remove repeated "este"
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text.strip()

    def _apply_general_formatting(self, text: str) -> str:
        """Apply general text formatting improvements."""

        # Remove excessive punctuation
        text = re.sub(r"[.]{2,}", "...", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)

        # Clean up spacing around punctuation
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        text = re.sub(r"([,.!?])\s+", r"\1 ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        # Capitalize first letter of sentences
        sentences = re.split(r"([.!?]+)", text)
        formatted_sentences = []

        for i, sentence in enumerate(sentences):
            if i % 2 == 0 and sentence.strip():  # Text parts (not punctuation)
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                formatted_sentences.append(sentence)
            else:
                formatted_sentences.append(sentence)

        return "".join(formatted_sentences).strip()

    def _resolve_timing_conflicts(
        self, segments: List[CaptionSegment]
    ) -> List[CaptionSegment]:
        """Resolve overlapping segments and optimize timing."""

        if not segments:
            return segments

        # Sort by start time
        segments.sort(key=lambda x: x.start_time)

        resolved = [segments[0]]

        for segment in segments[1:]:
            last_segment = resolved[-1]

            # Check for overlap
            if segment.start_time < last_segment.end_time:
                # Adjust timing to avoid overlap
                gap = 0.1  # 100ms gap between captions
                last_segment.end_time = segment.start_time - gap

                # Ensure minimum duration
                if last_segment.end_time - last_segment.start_time < 0.5:
                    last_segment.end_time = last_segment.start_time + 0.5
                    segment.start_time = last_segment.end_time + gap

            resolved.append(segment)

        return resolved

    def generate_srt_file(
        self, segments: List[CaptionSegment], output_path: str
    ) -> str:
        """Generate SRT subtitle file."""

        def format_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

        srt_content = []

        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment.start_time)
            end_time = format_timestamp(segment.end_time)

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(segment.text)
            srt_content.append("")  # Empty line between segments

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))

        print(f"SRT file saved: {output_path}")
        return output_path

    def generate_ass_file(
        self,
        segments: List[CaptionSegment],
        output_path: str,
        style: Optional[CaptionStyle] = None,
    ) -> str:
        """Generate ASS (Advanced SubStation Alpha) subtitle file with styling."""

        if style is None:
            style = self.default_style

        # ASS header
        ass_content = [
            "[Script Info]",
            "Title: Auto-generated Captions",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        ]

        # Convert colors to ASS format (BGR)
        def color_to_ass(color: str) -> str:
            if color.startswith("#"):
                color = color[1:]
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            return f"&H00{b:02X}{g:02X}{r:02X}"

        primary_color = color_to_ass(
            style.font_color if style.font_color.startswith("#") else "#FFFFFF"
        )
        outline_color = color_to_ass(
            style.border_color if style.border_color.startswith("#") else "#000000"
        )
        back_color = color_to_ass(
            style.background_color
            if style.background_color and style.background_color.startswith("#")
            else "#000000"
        )

        # Style definition
        alignment = (
            2 if style.position == "bottom" else 8 if style.position == "top" else 5
        )

        style_line = f"Style: Default,{style.font_family},{style.font_size},{primary_color},&H000000FF,{outline_color},{back_color},0,0,0,0,100,100,0,0,1,{style.border_width},1 if style.shadow else 0,{alignment},10,10,10,1"
        ass_content.append(style_line)

        ass_content.extend(
            [
                "",
                "[Events]",
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            ]
        )

        # Format timestamp for ASS
        def format_ass_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            centiseconds = int((secs % 1) * 100)
            secs = int(secs)
            return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

        # Add dialogue lines
        for segment in segments:
            start_time = format_ass_timestamp(segment.start_time)
            end_time = format_ass_timestamp(segment.end_time)
            text = segment.text.replace("\n", "\\N")  # ASS line break

            dialogue_line = (
                f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}"
            )
            ass_content.append(dialogue_line)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ass_content))

        print(f"ASS file saved: {output_path}")
        return output_path

    def burn_captions_to_video(
        self,
        video_path: str,
        subtitle_path: str,
        output_path: str,
        style: Optional[CaptionStyle] = None,
    ) -> str:
        """Burn captions directly into video using FFmpeg."""

        if style is None:
            style = self.default_style

        # Build FFmpeg filter for subtitle burning
        if subtitle_path.endswith(".srt"):
            # For SRT files
            subtitle_filter = f"subtitles={subtitle_path}:force_style='FontName={style.font_family},FontSize={style.font_size},PrimaryColour=&H{style.font_color.replace('#', '')},BorderStyle=1,Outline={style.border_width}'"
        else:
            # For ASS files
            subtitle_filter = f"ass={subtitle_path}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            subtitle_filter,
            "-c:a",
            "copy",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            output_path,
        ]

        try:
            print(f"Burning captions to video: {output_path}")
            subprocess.run(cmd, check=True, capture_output=True)
            print("Caption burning completed successfully")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Caption burning failed: {e.stderr.decode()}")
            raise


# Convenience functions
def generate_auto_captions(
    video_path: str,
    output_dir: str,
    language: Optional[str] = None,
    style: Optional[CaptionStyle] = None,
    formats: List[str] = ["srt", "ass"],
) -> Dict[str, str]:
    """
    Generate auto-captions in multiple formats.

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        language: Language code (auto-detected if None)
        style: Caption styling
        formats: List of formats to generate ('srt', 'ass', 'burned')

    Returns:
        Dict mapping format to output file path
    """
    generator = AutoCaptionsGenerator()

    # Generate caption segments
    segments = generator.create_smart_captions(video_path, language, style=style)

    if not segments:
        print("No captions generated")
        return {}

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    results = {}

    # Generate requested formats
    if "srt" in formats:
        srt_path = os.path.join(output_dir, f"{base_name}_captions.srt")
        generator.generate_srt_file(segments, srt_path)
        results["srt"] = srt_path

    if "ass" in formats:
        ass_path = os.path.join(output_dir, f"{base_name}_captions.ass")
        generator.generate_ass_file(segments, ass_path, style)
        results["ass"] = ass_path

    if "burned" in formats:
        # Use SRT for burning if available, otherwise generate temp SRT
        if "srt" in results:
            subtitle_file = results["srt"]
        else:
            subtitle_file = os.path.join(output_dir, f"{base_name}_temp.srt")
            generator.generate_srt_file(segments, subtitle_file)

        burned_path = os.path.join(output_dir, f"{base_name}_with_captions.mp4")
        generator.burn_captions_to_video(video_path, subtitle_file, burned_path, style)
        results["burned"] = burned_path

        # Clean up temp file if created
        if "srt" not in results:
            try:
                os.unlink(subtitle_file)
            except OSError:
                pass

    return results
