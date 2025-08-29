import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

MODEL_ID = "openai/whisper-large-v3"

print("Loading Whisper model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=TORCH_DTYPE,
    device=DEVICE,
)


def transcribe_audio(audio_path: str, start_time: float = None, end_time: float = None):
    """
    Transcribe a whole audio file or a segment.
    start_time & end_time in seconds. If None, transcribe entire audio.
    """
    if start_time is not None or end_time is not None:
        import tempfile
        import subprocess

        segment_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        start_time_arg = f"-ss {start_time}" if start_time else ""
        end_time_arg = f"-to {end_time}" if end_time else ""
        cmd = f"ffmpeg -y {start_time_arg} {end_time_arg} -i {audio_path} -vn -acodec mp3 {segment_file}"
        subprocess.run(cmd, shell=True, check=True)
        audio_path = segment_file

    result = pipe(audio_path, return_timestamps=True)
    return result["text"]

