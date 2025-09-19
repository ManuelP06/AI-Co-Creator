from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from enum import Enum

class UserBase(BaseModel):
    username: str
    email: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    is_superuser: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class VideoUploadResponse(BaseModel):
    video_id: int
    filename: str
    file_path: str
    file_size: int
    
    class Config:
        from_attributes = True

class ShotResponse(BaseModel):
    id: int
    shot_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    transcript: Optional[str] = None
    analysis: Optional[str] = None
    
    class Config:
        from_attributes = True

class ShotListResponse(BaseModel):
    video_id: int
    shots: List[ShotResponse]

class ContentType(str, Enum):
    INTERVIEW = "interview"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    PRODUCT_DEMO = "product_demo"

class VideoClipSchema(BaseModel):
    id: int
    source_path: str
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    duration: float = Field(..., ge=0.0)
    content_score: float = Field(default=0.0, ge=0.0, le=10.0)
    transcript: str = ""
    summary: str = ""
    tags: List[str] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_times(self):
        if self.end_time <= self.start_time:
            raise ValueError('end_time must be greater than start_time')
        if abs(self.duration - (self.end_time - self.start_time)) > 0.1:
            raise ValueError('duration must match end_time - start_time')
        return self

class TimelineItem(BaseModel):
    clip_id: int = Field(..., ge=0)
    order: int = Field(..., ge=1)
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    highlight_reason: str = ""
    
    @model_validator(mode='after')
    def validate_timeline_item(self):
        if self.end_time <= self.start_time:
            raise ValueError('end_time must be greater than start_time')
        return self

class TimelinePlan(BaseModel):
    total_duration: float = Field(..., ge=0.0)
    items: List[TimelineItem] = Field(..., min_items=1)
    
    @model_validator(mode='after')
    def validate_timeline_consistency(self):
        if not self.items:
            return self
        
        orders = [item.order for item in self.items]
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError('Timeline items must have consecutive order numbers starting from 1')
        
        return self

class StoryBeat(BaseModel):
    order: int = Field(..., ge=1)
    title: str
    summary: str
    supporting_clips: List[int] = Field(default_factory=list)
    duration_target: Optional[float] = Field(default=None, ge=0.0)

class Storyboard(BaseModel):
    theme: str
    target_audience: Optional[str] = ""
    strategy: Optional[str] = ""
    narrative_arc: Optional[str] = ""
    beats: List[StoryBeat] = Field(default_factory=list)

class EditorAgentResponse(BaseModel):
    storyboard: Optional[Storyboard] = None
    timeline: TimelinePlan

class EditingProjectRequest(BaseModel):
    name: str
    content_type: ContentType
    target_platform: Literal["youtube_shorts", "tiktok", "instagram_reels", "custom"] = "youtube_shorts"
    video_ids: List[int] = Field(..., min_items=1)
    brief: Optional[str] = None
    
    @field_validator('video_ids')
    @classmethod
    def validate_video_ids(cls, v):
        if not all(vid > 0 for vid in v):
            raise ValueError('All video IDs must be positive integers')
        if len(set(v)) != len(v):
            raise ValueError('Video IDs must be unique')
        return v

class EditingProjectResponse(BaseModel):
    project_id: int
    name: str
    content_type: ContentType
    target_platform: str
    source_videos: int
    total_clips: int
    selected_clips: int
    final_duration: float

class ContentAnalysisResponse(BaseModel):
    total_clips: int = Field(..., ge=0)
    total_duration: float = Field(..., ge=0.0)
    average_score: float = Field(..., ge=0.0, le=10.0)
    max_score: float = Field(..., ge=0.0, le=10.0)
    high_quality_clips: int = Field(..., ge=0)
    usable_clips: int = Field(..., ge=0)
    content_density: float = Field(..., ge=0.0, le=1.0)

class VideoFormatSchema(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0) 
    name: str
    ratio: str
    
    @model_validator(mode='after')
    def validate_format(self):
        calculated_ratio = self.width / self.height
        ratio_map = {
            "9:16": 9/16,
            "16:9": 16/9,
            "1:1": 1.0
        }
        
        if self.ratio in ratio_map:
            expected_ratio = ratio_map[self.ratio]
            if abs(calculated_ratio - expected_ratio) > 0.01:
                raise ValueError(f'Width/height does not match specified ratio {self.ratio}')
        
        return self

class RenderRequest(BaseModel):
    output_filename: Optional[str] = Field(None, description="Custom output filename")
    video_format: Literal["portrait", "landscape", "square"] = Field(default="portrait", description="Video format")
    quality: Literal["low", "medium", "high"] = Field(default="high", description="Render quality")
    use_gpu: bool = Field(default=False, description="Use GPU acceleration")
    max_duration: Optional[float] = Field(default=59.0, description="Maximum video duration in seconds")
    
    @field_validator('output_filename')
    @classmethod
    def validate_output_filename(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('output_filename cannot be empty')
            if not v.endswith(('.mp4', '.mov', '.avi')):
                raise ValueError('output_filename must have a valid video file extension')
        return v

class RenderResponse(BaseModel):
    job_id: str
    status: str
    video_id: int
    output_filename: Optional[str] = None
    created_at: str
    settings: Optional[Dict[str, Any]] = None

class RenderJobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = Field(..., ge=0, le=100)
    output_path: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None
    duration: Optional[float] = None
    settings: Optional[Dict[str, Any]] = None

class OutputFile(BaseModel):
    filename: str
    size_mb: float = Field(..., ge=0.0)
    created_at: str

class OutputListResponse(BaseModel):
    files: List[OutputFile]

class SystemInfo(BaseModel):
    ffmpeg_available: bool
    video_formats: List[Dict[str, str]]
    quality_settings: Dict[str, Dict[str, Any]]
    output_directory: str

class JobListResponse(BaseModel):
    jobs: List[Dict[str, Any]]
    total: int

class ErrorDetail(BaseModel):
    error_type: str
    error_code: Optional[str] = None
    message: str
    context: Optional[Dict[str, Any]] = None
    suggestions: List[str] = Field(default_factory=list)

class OperationStatus(BaseModel):
    success: bool
    message: str
    details: Optional[ErrorDetail] = None
    execution_time: Optional[float] = None

def get_platform_video_format(platform: str) -> VideoFormatSchema:
    """Get recommended video format for platform"""
    formats = {
        "tiktok": VideoFormatSchema(
            width=1080, height=1920, name="portrait", ratio="9:16"
        ),
        "instagram": VideoFormatSchema(
            width=1080, height=1920, name="portrait", ratio="9:16"
        ),
        "youtube": VideoFormatSchema(
            width=1080, height=1920, name="portrait", ratio="9:16"
        ),
        "youtube_shorts": VideoFormatSchema(
            width=1080, height=1920, name="portrait", ratio="9:16"
        )
    }
    return formats.get(platform, formats["tiktok"])

def get_platform_limits(platform: str) -> Dict[str, float]:
    """Get platform limits as dictionary"""
    limits = {
        "youtube_shorts": {"max_duration": 59.0, "ideal_clips": 6, "max_clip_duration": 8.0},
        "tiktok": {"max_duration": 59.0, "ideal_clips": 7, "max_clip_duration": 7.0},
        "instagram_reels": {"max_duration": 59.0, "ideal_clips": 6, "max_clip_duration": 8.0},
        "custom": {"max_duration": 300.0, "ideal_clips": 20, "max_clip_duration": 15.0}
    }
    return limits.get(platform, limits["custom"])

# Model rebuilds
EditorAgentResponse.model_rebuild()
Storyboard.model_rebuild()
TimelinePlan.model_rebuild()
StoryBeat.model_rebuild()
TimelineItem.model_rebuild()
VideoClipSchema.model_rebuild()