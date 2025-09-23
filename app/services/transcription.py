import os
# Removed unused json import
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def get_optimal_whisper_model():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        if gpu_memory > 12:
            return "openai/whisper-large-v3"
        elif gpu_memory > 8:
            return "openai/whisper-large-v2"
        elif gpu_memory > 6:
            return "openai/whisper-medium"
        else:
            return "openai/whisper-small"
    else:
        return "openai/whisper-base"


MODEL_ID = get_optimal_whisper_model()
print(f"Loading Whisper model: {MODEL_ID} on {DEVICE}")

# Global variables for lazy loading
_model = None
_processor = None
_pipe = None


def get_whisper_pipeline():
    """Lazy load Whisper pipeline to save memory when not needed."""
    global _model, _processor, _pipe

    if _pipe is None:
        print("Initializing Whisper pipeline...")

        _model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        _model.to(DEVICE)

        _processor = AutoProcessor.from_pretrained(MODEL_ID)

        _pipe = pipeline(
            "automatic-speech-recognition",
            model=_model,
            tokenizer=_processor.tokenizer,
            feature_extractor=_processor.feature_extractor,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
            return_timestamps=True,
            chunk_length_s=30,  # Process in 30-second chunks
            batch_size=8 if torch.cuda.is_available() else 1,
        )
        print("Whisper pipeline ready!")

    return _pipe


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None
    words: Optional[List[Dict]] = None


def extract_audio_from_video(video_path: str, output_path: Optional[str] = None) -> str:
    """Extract high-quality audio from video for better transcription."""
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    # WAV format for best quality, 16kHz sample rate (optimal for Whisper)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",  # No video
        "-acodec",
        "pcm_s16le",  # High quality PCM
        "-ar",
        "16000",  # 16kHz sample rate
        "-ac",
        "1",  # Mono
        "-af",
        "loudnorm",  # Audio normalization
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Verify the file was created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            raise subprocess.CalledProcessError(1, cmd, b"Empty audio file created")
    except subprocess.CalledProcessError as e:
        print(f"Audio extraction failed: {e.stderr.decode() if e.stderr else str(e)}")
        # Fallback to simpler extraction
        try:
            cmd_simple = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                output_path
            ]
            subprocess.run(cmd_simple, check=True, capture_output=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
            else:
                raise FileNotFoundError("Failed to create valid audio file")
        except Exception as fallback_error:
            print(f"Fallback extraction also failed: {fallback_error}")
            raise


def preprocess_audio(audio_path: str) -> str:
    """Preprocess audio for optimal transcription quality."""
    output_path = tempfile.NamedTemporaryFile(
        suffix="_processed.wav", delete=False
    ).name

    # Audio preprocessing pipeline
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-af",
        "highpass=f=80,lowpass=f=8000,loudnorm,areverse,silenceremove=1:0:-50dB,areverse,silenceremove=1:0:-50dB",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError:
        # Fallback: return original if preprocessing fails
        return audio_path


def transcribe_with_timestamps(
    audio_path: str, language: Optional[str] = None
) -> List[TranscriptionSegment]:
    """Enhanced transcription with detailed timestamps and confidence scores."""
    pipe = get_whisper_pipeline()

    # Set language if specified (improves accuracy)
    generate_kwargs = {"language": language} if language else {}

    try:
        result = pipe(
            audio_path, generate_kwargs=generate_kwargs, return_timestamps=True
        )

        segments = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                segments.append(
                    TranscriptionSegment(
                        start=chunk["timestamp"][0] or 0.0,
                        end=chunk["timestamp"][1] or 0.0,
                        text=chunk["text"].strip(),
                        confidence=getattr(chunk, "confidence", None),
                    )
                )
        else:
            # Fallback for older format
            segments.append(
                TranscriptionSegment(
                    start=0.0, end=0.0, text=result["text"].strip(), confidence=None
                )
            )

        return segments

    except Exception as e:
        print(f"Transcription failed: {e}")
        # Return a segment with a more user-friendly error message
        return [
            TranscriptionSegment(
                start=0.0, end=0.0, text="[Audio transcription unavailable - continuing without transcript]", confidence=None
            )
        ]


def detect_language(audio_path: str) -> str:
    """Detect the primary language in the audio."""
    pipe = get_whisper_pipeline()

    try:
        # Use a small sample for language detection
        sample_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            audio_path,
            "-t",
            "30",
            "-ar",
            "16000",
            sample_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        result = pipe(sample_path, generate_kwargs={"task": "transcribe"})

        # Try to detect language from model output
        if hasattr(result, "language"):
            return result.language

        # Fallback: attempt to infer from text patterns
        text = result.get("text", "")
        if re.search(r"[äöüß]", text.lower()):
            return "de"  # German
        elif re.search(r"[àèéêîôùûç]", text.lower()):
            return "fr"  # French
        elif re.search(r"[ñáéíóúü]", text.lower()):
            return "es"  # Spanish
        else:
            return "en"  # Default to English

    except Exception:
        return "en"  # Default fallback


def transcribe_audio(
    audio_path: str,
    start_time: float = None,
    end_time: float = None,
    language: Optional[str] = None,
    enhance_quality: bool = True,
) -> str:
    """
    Transcription with preprocessing and quality improvements.
    """
    temp_files = []

    try:
        working_audio = audio_path

        # Extract segment if specified
        if start_time is not None or end_time is not None:
            segment_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            temp_files.append(segment_file)

            start_arg = ["-ss", str(start_time)] if start_time else []
            duration_arg = ["-t", str(end_time - (start_time or 0))] if end_time else []

            cmd = (
                ["ffmpeg", "-y"]
                + start_arg
                + ["-i", audio_path]
                + duration_arg
                + ["-vn", "-ar", "16000", "-ac", "1", segment_file]
            )

            subprocess.run(cmd, check=True, capture_output=True)
            working_audio = segment_file

        # Enhance audio quality if requested
        if enhance_quality:
            processed_audio = preprocess_audio(working_audio)
            if processed_audio != working_audio:
                temp_files.append(processed_audio)
                working_audio = processed_audio

        # Auto-detect language if not specified
        if language is None:
            language = detect_language(working_audio)
            print(f"Detected language: {language}")

        # Transcribe with timestamps
        segments = transcribe_with_timestamps(working_audio, language)

        # Combine all text
        transcript_text = " ".join(seg.text for seg in segments if seg.text.strip())

        # Return meaningful result even if empty
        if not transcript_text.strip():
            return "[No speech detected in audio segment]"

        return transcript_text

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass


def transcribe_video_segments(
    video_path: str, segments: List[Tuple[float, float]], language: Optional[str] = None
) -> Dict[int, str]:
    """Transcribe multiple video segments efficiently."""
    # Extract audio once
    audio_path = extract_audio_from_video(video_path)

    try:
        results = {}
        for i, (start_time, end_time) in enumerate(segments):
            text = transcribe_audio(
                audio_path, start_time, end_time, language, enhance_quality=True
            )
            results[i] = text

        return results
    finally:
        # Clean up extracted audio
        try:
            os.unlink(audio_path)
        except OSError:
            pass
