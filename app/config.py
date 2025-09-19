from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Database
    database_url: str = "sqlite:///./videos.db"

    # AI Models
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout: int = 300
    video_llama_model: str = "DAMO-NLP-SG/VideoLLaMA3-7B"

    # Redis
    redis_url: Optional[str] = None

    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # File Upload
    max_file_size_mb: int = 500
    allowed_video_extensions: str = "mp4,avi,mov,mkv"
    upload_directory: str = "./uploads"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # GPU Configuration
    cuda_memory_limit_gb: int = 12
    pytorch_cuda_alloc_conf: str = "expandable_segments:True,max_split_size_mb:512"

    # Development
    debug: bool = False
    reload: bool = False

    # API Configuration
    api_title: str = "AI Co-Creator API"
    api_description: str = "AI-powered content creation and video editing platform"
    api_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.upload_directory, exist_ok=True)

# Set PyTorch CUDA configuration
if settings.pytorch_cuda_alloc_conf:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", settings.pytorch_cuda_alloc_conf)