# AI Co-Creator

Professional AI-powered content automation platform for video processing, analysis, and automated editing.

## Features

- **Video Upload**: Upload and manage video files with metadata
- **Shot Detection**: Automatic shot detection and segmentation using OpenCV
- **Transcription**: Audio-to-text transcription with Whisper
- **Video Analysis**: AI-powered content analysis with VideoLLaMA3 *Currently buggy, requires 10GB+ VRAM*
- **Automated Editing**: Timeline generation optimized for social platforms
- **Video Rendering**: Export videos in multiple formats and resolutions
- **Authentication**: JWT-based user management
- **REST API**: Complete FastAPI backend with async support
- **Multi-Platform Export**: YouTube Shorts, TikTok, Instagram Reels

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, Alembic
- **AI/ML**: PyTorch 2.8.0, Whisper, OpenCV
- **Database**: SQLite (dev), PostgreSQL (prod)
- **Authentication**: JWT with bcrypt
- **Caching**: Redis with fallback
- **Deployment**: Docker, Docker Compose

## Quick Setup

### Prerequisites
- Python 3.9+
- 8GB+ RAM (16GB+ recommended for video analysis)
- CUDA 11.8+ and 10GB+ VRAM (optional, for GPU acceleration and AI analysis)
- FFmpeg (for video processing)

### Automated Installation
```bash
git clone https://github.com/yourusername/AI-Co-Creator.git
cd AI-Co-Creator
chmod +x setup.sh
./setup.sh
```

### Manual Installation
```bash
# 1. Clone repository
git clone https://github.com/yourusername/AI-Co-Creator.git
cd AI-Co-Creator

# 2. Environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env
# Generate secure secret key
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))" >> .env

# 4. Database initialization
python init_db.py

# 5. Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Test installation
python test_api.py
```

### Docker Setup
```bash
docker-compose up --build
```

## API Usage

### Authentication
```python
import requests

# Register
response = requests.post("http://localhost:8000/api/v1/auth/register", json={
    "username": "user", "email": "user@example.com", "password": "password"
})

# Login
response = requests.post("http://localhost:8000/api/v1/auth/login", data={
    "username": "user", "password": "password"
})
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
```

### Video Processing
```python
# Upload video
files = {"file": open("video.mp4", "rb")}
response = requests.post("http://localhost:8000/api/v1/upload/",
                        files=files, data={"username": "user"}, headers=headers)
video_id = response.json()["video_id"]

# Process video
requests.post(f"http://localhost:8000/api/v1/shots/{video_id}/detect", headers=headers)
# Transcribe audio
requests.post(f"http://localhost:8000/api/v1/transcript/{video_id}/transcribe", headers=headers)

# Analyze video content (requires 10GB+ VRAM, currently unstable)
requests.post(f"http://localhost:8000/api/v1/analysis/{video_id}", headers=headers)

# Generate edit
response = requests.post(f"http://localhost:8000/api/v1/editor/{video_id}/create-edit",
                        json={"content_type": "interview", "platform": "youtube_shorts"},
                        headers=headers)

# Render video
requests.post(f"http://localhost:8000/api/v1/renderer/{video_id}/render",
             json={"quality": "high", "format": "portrait"}, headers=headers)
```

## Project Structure

```
AI-Co-Creator/
├── app/
│   ├── core/           # Authentication, middleware, utilities
│   ├── routers/        # API endpoints
│   ├── services/       # Business logic
│   ├── models.py       # Database models
│   ├── schemas.py      # Pydantic schemas
│   ├── config.py       # Configuration
│   └── main.py         # FastAPI application
├── alembic/            # Database migrations
├── uploads/            # Video uploads
├── outputs/            # Rendered videos
├── requirements.txt    # Dependencies
├── .env.example        # Environment template
├── init_db.py          # Database setup
├── setup.sh            # Automated setup
└── docker-compose.yml  # Container orchestration
```

## Configuration

### Environment Variables
```env
# Database
DATABASE_URL=sqlite:///./videos.db

# Security
SECRET_KEY=your-generated-secret-key
DEBUG=false

# AI Models
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b

# Caching
REDIS_URL=redis://localhost:6379/0
```

### Default Users
- **Admin**: username `admin`, password `admin123`
- **Test**: username `testuser`, password `testpass123`

Change passwords in production.

## Production Deployment

### Docker Production
```bash
# Create production environment
cp .env.example .env.prod
# Edit .env.prod with production values

# Deploy
docker-compose -f docker-compose.yml up -d
```

### Manual Production Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt gunicorn

# 2. Database migration
alembic upgrade head

# 3. Start with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Configuration
- Set `DEBUG=false`
- Use PostgreSQL: `DATABASE_URL=postgresql://user:pass@host:5432/db`
- Generate secure `SECRET_KEY`
- Configure CORS for specific domains
- Enable HTTPS with reverse proxy

## API Documentation

Start the server and visit:
- **Interactive API Documentation**: http://127.0.0.1:8000/docs
- **Alternative Documentation**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/health

The Swagger UI at `/docs` provides a complete interface to test all endpoints including:
- Video upload and management
- Shot detection and analysis
- Audio transcription
- Video analysis (AI content understanding)
- Automated editing and timeline generation
- Video rendering and export

## Verification

After setup, verify everything works:

```bash
# 1. Check server is running
curl http://localhost:8000/health
# Expected: {"status":"healthy","version":"1.0.0"}

# 2. Run API tests
python test_api.py
# Expected: 3/3 tests passed

# 3. Check API documentation
curl http://localhost:8000/docs
# Should return HTML documentation page
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Database Issues**
```bash
rm videos.db  # Reset database
python init_db.py
```

**CUDA Issues**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**VideoLLaMA3 Analysis Issues**
- Requires `transformers>=4.46.3` for Qwen2Config support
- Needs 10GB+ VRAM for video analysis
- Currently unstable - may fail with memory errors
- Consider using CPU mode for testing: `force_cpu=True`

**Port in Use**
```bash
lsof -i :8000
kill -9 <PID>
```

### Testing Installation
```bash
# Run API tests
python test_api.py

# Quick health check
curl http://localhost:8000/health

# Test authentication
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpass123"

# Expected output: {"access_token":"...","token_type":"bearer"}
```

## Contributing

1. Fork repository
2. Create feature branch
3. Implement changes
4. Test thoroughly
5. Submit pull request

## License

MIT License - see LICENSE file for details.