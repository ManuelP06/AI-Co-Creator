# AI Co-Creator

AI video content creation platform. Transform long-form videos into short-form content for TikTok, YouTube Shorts, Instagram Reels, and more.

## Core Features

- **Scene Detection** - Hybrid AI models for shot boundary detection
- **Transcription** - Whisper AI with audio preprocessing
- **Content Creation** - AI scene analysis and scoring
- **Auto Captions** - Subtitle generation in multiple formats
- **Multi-Platform Rendering** - Exports for all major social platforms
- **Clean UI** - Minimal, extensible interface

## Quick Start

### Prerequisites
- Python 3.9+
- FFmpeg
- 8GB+ RAM (16GB+ recommended)
- CUDA GPU (optional, for acceleration)

### Installation
```bash
git clone <repository-url>
cd AI-Co-Creator
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python init_db.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker (Recommended)
```bash
docker-compose up --build
```

## Usage

The software can be used in two ways:

### 1. Web Interface (Recommended for General Use and mTesting, Still a bit Buggy)
Serve the UI and open in your browser:
```bash
cd ui && python -m http.server 3000
# Open http://localhost:3000
```

Complete workflow:
1. Upload video files
2. Configure content type and objectives
3. Process with AI pipeline
4. Review and edit timeline
5. Generate auto-captions
6. Render for multiple platforms

### 2. Swagger API Interface (Advanced Control & Insights)
For detailed control and debugging, use the interactive API documentation:
```bash
# Open http://127.0.0.1:8000/docs
```

The Swagger UI provides:
- **Complete API control** - Access all endpoints directly
- **Pipeline monitoring** - Step-by-step processing insights
- **More configuration** - Fine-tune detection thresholds, quality settings
- **Real-time debugging** - View detailed logs and processing status
- **Batch operations** - Process multiple videos efficiently

### API Examples
```python
import requests

# Upload and process video
files = {"file": open("video.mp4", "rb")}
response = requests.post("http://localhost:8000/api/v1/upload/", files=files)
video_id = response.json()["video_id"]

# Create content
requests.post(f"http://localhost:8000/api/v1/shots/{video_id}/detect")
requests.post(f"http://localhost:8000/api/v1/transcript/{video_id}/transcribe")

content = requests.post(f"http://localhost:8000/api/v1/editor/{video_id}",
    json={
        "content_type": "interview",
        "target_platforms": ["youtube_shorts", "tiktok"],
        "objective": "Create highlights"
    })
```

## API Documentation

- Interactive docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## Configuration

Copy and configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

Key settings:
```env
DATABASE_URL=sqlite:///./videos.db
SECRET_KEY=your-secret-key-here
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
```

## Supported Platforms

- **YouTube Shorts** (1080x1920, 60s max)
- **TikTok** (1080x1920, 60s max)
- **Instagram Reels** (1080x1920, 90s max)
- **LinkedIn** (1920x1080, 300s max)
- **Twitter** (1280x720, 140s max)

## Content Types

- Interview
- Podcast
- Educational
- Entertainment
- Product Demo
- Marketing
- Tutorial

## Architecture

```
app/
├── routers/          # API endpoints
├── services/         # Core AI services
│   ├── editor.py
│   ├── shot_detection.py
│   ├── transcription.py
│   ├── auto_captions.py
│   ├── renderer.py
│   └── video_understanding.py
├── models.py         # Database models
└── main.py          # FastAPI app

ui/                   # Web interface
outputs/             # Rendered videos
uploads/             # Source videos
```


## License

MIT License