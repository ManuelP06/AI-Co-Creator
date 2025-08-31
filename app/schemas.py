from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum

class UserBase(BaseModel):
    username: str

class UserResponse(UserBase):
    id: int
    
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
    """Content types for different editing strategies"""
    INTERVIEW = "interview"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    PRODUCT_DEMO = "product_demo"

class TransitionType(str, Enum):
    """Professional video transitions"""
    HARD_CUT = "hard_cut"
    FADE = "fade"
    SLIDE = "slide"
    ZOOM = "zoom"

class VideoClipSchema(BaseModel):
    """Schema for video clip data"""
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
    """Individual timeline item"""
    clip_id: int = Field(..., ge=0)
    order: int = Field(..., ge=1)
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    highlight_reason: str = ""
    transition_type: TransitionType = TransitionType.HARD_CUT
    
    @model_validator(mode='after')
    def validate_timeline_item(self):
        if self.end_time <= self.start_time:
            raise ValueError('end_time must be greater than start_time')
        return self

class StoryBeat(BaseModel):
    """Storyboard beat structure"""
    order: int = Field(..., ge=1)
    title: str
    summary: str
    supporting_clips: List[int] = Field(default_factory=list)
    duration_target: Optional[float] = Field(default=None, ge=0.0)

class Storyboard(BaseModel):
    """Complete storyboard structure"""
    theme: str
    target_audience: Optional[str] = ""
    strategy: Optional[str] = ""
    narrative_arc: Optional[str] = ""
    beats: List[StoryBeat] = Field(..., min_items=1)

class TimelinePlan(BaseModel):
    """Complete timeline plan"""
    total_duration: float = Field(..., ge=0.0)
    items: List[TimelineItem] = Field(..., min_items=1)
    
    @model_validator(mode='after')
    def validate_timeline_consistency(self):
        if not self.items:
            return self
        
        # Validate order sequence
        orders = [item.order for item in self.items]
        if sorted(orders) != list(range(1, len(orders) + 1)):
            raise ValueError('Timeline items must have consecutive order numbers starting from 1')
        
        # Validate total duration consistency
        calculated_duration = sum(item.end_time - item.start_time for item in self.items)
        if abs(self.total_duration - calculated_duration) > 0.5:
            raise ValueError('total_duration must match sum of item durations')
        
        return self

class EditorAgentResponse(BaseModel):
    """Response from editor agent"""
    storyboard: Optional[Storyboard] = None
    timeline: TimelinePlan

class EditingProjectRequest(BaseModel):
    """Request to create editing project"""
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
    """Response for editing project creation"""
    project_id: int
    name: str
    content_type: ContentType
    target_platform: str
    source_videos: int
    total_clips: int
    selected_clips: int
    final_duration: float

class ContentAnalysisResponse(BaseModel):
    """Content analysis metrics"""
    total_clips: int = Field(..., ge=0)
    total_duration: float = Field(..., ge=0.0)
    average_score: float = Field(..., ge=0.0, le=10.0)
    max_score: float = Field(..., ge=0.0, le=10.0)
    high_quality_clips: int = Field(..., ge=0)
    usable_clips: int = Field(..., ge=0)
    content_density: float = Field(..., ge=0.0, le=1.0)

class MultiVideoProjectResponse(BaseModel):
    """Response for multi-video project analytics"""
    project_id: int
    source_videos: int
    total_clips: int
    selected_clips: int
    final_duration: float
    content_type: str
    platform: str

class ProjectAnalyticsResponse(BaseModel):
    """Analytics for multi-video projects"""
    source_videos: int
    total_source_clips: int
    total_source_duration: float
    average_quality: float = Field(..., ge=0.0, le=10.0)
    high_quality_clips: int
    recommended_highlights: int
    estimated_output_duration: float

class VideoFormatSchema(BaseModel):
    """Video format specification"""
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0) 
    name: str
    ratio: str
    platform: str
    
    @model_validator(mode='after')
    def validate_format(self):
        # Validate common aspect ratios
        calculated_ratio = self.width / self.height
        ratio_map = {
            "9:16": 9/16,
            "16:9": 16/9,
            "1:1": 1.0,
            "4:3": 4/3
        }
        
        if self.ratio in ratio_map:
            expected_ratio = ratio_map[self.ratio]
            if abs(calculated_ratio - expected_ratio) > 0.01:
                raise ValueError(f'Width/height does not match specified ratio {self.ratio}')
        
        return self

class CaptionStyle(BaseModel):
    """Caption styling options"""
    font: str = "Arial Bold"
    size: str = "32"
    color: str = "white"
    outline: str = "black"
    outline_width: str = "3"
    
    @field_validator('size', 'outline_width')
    @classmethod
    def validate_numeric_string(cls, v):
        try:
            num_val = int(v)
            if num_val <= 0:
                raise ValueError('Size and outline_width must be positive numbers')
        except ValueError:
            raise ValueError('Size and outline_width must be valid positive integers as strings')
        return v

class RenderRequest(BaseModel):
    """Request for video rendering"""
    video_id: int = Field(..., ge=1)
    output_path: str
    platform: Literal["tiktok", "instagram", "youtube"] = "tiktok"
    quality: Literal["low", "medium", "high"] = "high"
    use_gpu: bool = False
    auto_captions: bool = True
    caption_style: Optional[CaptionStyle] = None
    
    @field_validator('output_path')
    @classmethod
    def validate_output_path(cls, v):
        if not v.strip():
            raise ValueError('output_path cannot be empty')
        if not v.endswith(('.mp4', '.mov', '.avi')):
            raise ValueError('output_path must have a valid video file extension')
        return v

class ViralMetrics(BaseModel):
    """Viral content performance metrics"""
    total_clips: int = Field(..., ge=0)
    total_duration: float = Field(..., ge=0.0)
    avg_viral_score: float = Field(..., ge=0.0, le=10.0)
    max_viral_score: float = Field(..., ge=0.0, le=10.0)
    hook_quality: float = Field(..., ge=0.0, le=10.0)
    viral_clips_count: int = Field(..., ge=0)
    duration_efficiency: float = Field(..., ge=0.0, le=1.0)
    optimal_for_viral: bool

class RenderResponse(BaseModel):
    """Response from video rendering"""
    success: bool
    output_path: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    viral_metrics: Optional[ViralMetrics] = None
    error: Optional[str] = None

class OptimizationRequest(BaseModel):
    """Request for viral optimization"""
    video_id: int = Field(..., ge=1)
    platform: Literal["tiktok", "instagram", "youtube"] = "tiktok"
    max_duration: float = Field(default=59.0, gt=0.0, le=180.0)
    min_viral_score: float = Field(default=2.0, ge=0.0, le=10.0)

class OptimizationResponse(BaseModel):
    """Response from viral optimization"""
    success: bool
    original_clips: int
    optimized_clips: int
    final_duration: float
    viral_metrics: ViralMetrics
    timeline: TimelinePlan
    error: Optional[str] = None

class FFmpegCapabilities(BaseModel):
    """FFmpeg system capabilities"""
    available: bool
    version: Optional[str] = None
    codecs: Optional[Dict[str, bool]] = None
    filters: Optional[Dict[str, bool]] = None
    formats: Optional[Dict[str, bool]] = None
    gpu: Optional[Dict[str, bool]] = None
    error: Optional[str] = None

class TimelineExportRequest(BaseModel):
    """Request for timeline export"""
    video_id: int = Field(..., ge=1)
    format: Literal["json", "edl", "fcpxml"] = "json"
    include_metadata: bool = True

class TimelineExportResponse(BaseModel):
    """Response for timeline export"""
    success: bool
    export_data: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    format: str
    error: Optional[str] = None

class MultiVideoProjectRequest(BaseModel):
    """Request for multi-video project creation"""
    video_ids: List[int] = Field(..., min_items=1)
    content_type: ContentType
    target_platform: Literal["youtube_shorts", "tiktok", "instagram_reels", "custom"] = "youtube_shorts"
    brief: Optional[str] = None
    
    @field_validator('video_ids')
    @classmethod
    def validate_unique_video_ids(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('Video IDs must be unique')
        return v

class ProjectClipAnalysis(BaseModel):
    """Analysis of clips within a project"""
    clip_id: int
    source_video_id: int
    content_score: float = Field(..., ge=0.0, le=10.0)
    duration: float = Field(..., ge=0.0)
    transcript_excerpt: str = ""
    selection_reason: str = ""

class ProjectAnalysisResponse(BaseModel):
    """Detailed project analysis"""
    project_summary: ProjectAnalyticsResponse
    clip_analysis: List[ProjectClipAnalysis]
    recommendations: List[str]
    platform_optimization: Dict[str, Any]

class ClipSelectionCriteria(BaseModel):
    """Criteria for clip selection"""
    min_score: float = Field(default=2.0, ge=0.0, le=10.0)
    prefer_dialogue: bool = True
    maintain_context: bool = True
    ideal_clip_length: float = Field(default=5.0, gt=0.0)
    prefer_hooks: bool = False

class EditingStrategy(BaseModel):
    """Editing strategy configuration"""
    content_type: ContentType
    selection_criteria: ClipSelectionCriteria
    platform_optimizations: Dict[str, Any] = Field(default_factory=dict)
    narrative_priority: float = Field(default=1.0, ge=0.0, le=1.0)
    viral_priority: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def validate_priorities(self):
        total = self.narrative_priority + self.viral_priority
        if abs(total - 1.0) > 0.01:
            raise ValueError('narrative_priority + viral_priority must equal 1.0')
        return self

class ProjectConfiguration(BaseModel):
    """Complete project configuration"""
    name: str
    content_type: ContentType
    target_platform: str
    strategy: EditingStrategy
    brief: Optional[str] = None
    custom_limits: Optional[Dict[str, float]] = None
    
    @field_validator('custom_limits')
    @classmethod
    def validate_custom_limits(cls, v):
        if v is not None:
            required_keys = ['max_duration', 'ideal_clips', 'max_clip_duration']
            if not all(key in v for key in required_keys):
                raise ValueError(f'custom_limits must contain: {required_keys}')
            if any(val <= 0 for val in v.values()):
                raise ValueError('All custom limit values must be positive')
        return v

class AdvancedEditRequest(BaseModel):
    """Advanced editing request with full configuration"""
    video_ids: List[int] = Field(..., min_items=1)
    config: ProjectConfiguration
    render_immediately: bool = False
    export_formats: List[str] = Field(default=["json"])

class AdvancedEditResponse(BaseModel):
    """Advanced editing response"""
    project_id: int
    editor_response: EditorAgentResponse
    project_analytics: ProjectAnalyticsResponse
    rendered_video_path: Optional[str] = None
    export_files: Dict[str, str] = Field(default_factory=dict)

class PlatformLimits(BaseModel):
    """Platform-specific video limits"""
    max_duration: float = Field(..., gt=0.0)
    ideal_clips: int = Field(..., ge=1)
    max_clip_duration: float = Field(..., gt=0.0)
    recommended_format: VideoFormatSchema

class PlatformOptimization(BaseModel):
    """Platform optimization settings"""
    platform: str
    limits: PlatformLimits
    caption_style: CaptionStyle
    viral_keywords: List[str] = Field(default_factory=list)
    engagement_hooks: List[str] = Field(default_factory=list)

class BatchRenderRequest(BaseModel):
    """Request for batch rendering multiple videos"""
    video_ids: List[int] = Field(..., min_items=1)
    platforms: List[str] = Field(default=["tiktok"])
    content_type: ContentType = ContentType.INTERVIEW
    quality: Literal["low", "medium", "high"] = "high"
    use_gpu: bool = False
    output_directory: str
    
    @field_validator('platforms')
    @classmethod
    def validate_platforms(cls, v):
        valid_platforms = ["tiktok", "instagram", "youtube", "youtube_shorts"]
        invalid = [p for p in v if p not in valid_platforms]
        if invalid:
            raise ValueError(f'Invalid platforms: {invalid}. Valid: {valid_platforms}')
        return v

class BatchRenderJob(BaseModel):
    """Individual batch render job"""
    video_id: int
    platform: str
    status: Literal["pending", "processing", "completed", "failed"]
    output_path: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[float] = None

class BatchRenderResponse(BaseModel):
    """Response for batch rendering"""
    batch_id: str
    total_jobs: int
    completed_jobs: List[BatchRenderJob]
    failed_jobs: List[BatchRenderJob]
    summary: Dict[str, Any]

class QualityMetrics(BaseModel):
    """Video quality assessment"""
    resolution_score: float = Field(..., ge=0.0, le=10.0)
    audio_quality: float = Field(..., ge=0.0, le=10.0)
    visual_clarity: float = Field(..., ge=0.0, le=10.0)
    compression_efficiency: float = Field(..., ge=0.0, le=1.0)
    overall_rating: float = Field(..., ge=0.0, le=10.0)

class PerformanceMetrics(BaseModel):
    """Rendering performance metrics"""
    render_time: float = Field(..., ge=0.0)
    cpu_usage: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    memory_usage: Optional[float] = Field(default=None, ge=0.0)
    gpu_utilization: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    output_file_size: int = Field(..., ge=0)

class RenderJobStatus(BaseModel):
    """Status of rendering job"""
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    estimated_time_remaining: Optional[float] = None
    quality_metrics: Optional[QualityMetrics] = None
    performance_metrics: Optional[PerformanceMetrics] = None
    error_details: Optional[str] = None

class SystemCapabilities(BaseModel):
    """System rendering capabilities"""
    ffmpeg: FFmpegCapabilities
    gpu_available: bool
    max_concurrent_renders: int
    supported_formats: List[str]
    supported_platforms: List[str]
    memory_limit_gb: Optional[float] = None

class RenderConfiguration(BaseModel):
    """Global render configuration"""
    default_quality: Literal["low", "medium", "high"] = "high"
    prefer_gpu: bool = False
    max_render_time: int = 1800  # 30 minutes
    temp_directory: Optional[str] = None
    cleanup_temp_files: bool = True

class ErrorDetail(BaseModel):
    """Detailed error information"""
    error_type: str
    error_code: Optional[str] = None
    message: str
    context: Optional[Dict[str, Any]] = None
    suggestions: List[str] = Field(default_factory=list)

class OperationStatus(BaseModel):
    """Generic operation status"""
    success: bool
    message: str
    details: Optional[ErrorDetail] = None
    execution_time: Optional[float] = None

# Utility functions
def validate_platform_duration(platform: str, duration: float) -> bool:
    """Validate duration against platform limits"""
    limits = {
        "tiktok": 60.0,
        "instagram": 60.0, 
        "youtube": 60.0,
        "youtube_shorts": 60.0,
        "custom": 300.0
    }
    return duration <= limits.get(platform, 60.0)

def get_platform_video_format(platform: str) -> VideoFormatSchema:
    """Get recommended video format for platform"""
    formats = {
        "tiktok": VideoFormatSchema(
            width=1080, height=1920, name="9x16", 
            ratio="9:16", platform="TikTok"
        ),
        "instagram": VideoFormatSchema(
            width=1080, height=1920, name="9x16",
            ratio="9:16", platform="Instagram Reels"
        ),
        "youtube": VideoFormatSchema(
            width=1080, height=1920, name="9x16",
            ratio="9:16", platform="YouTube Shorts"
        ),
        "youtube_shorts": VideoFormatSchema(
            width=1080, height=1920, name="9x16",
            ratio="9:16", platform="YouTube Shorts"
        )
    }
    return formats.get(platform, formats["tiktok"])

def get_default_caption_style(platform: str) -> CaptionStyle:
    """Get default caption style for platform"""
    styles = {
        "tiktok": CaptionStyle(
            font="Arial Bold", size="36", color="white",
            outline="black", outline_width="3"
        ),
        "instagram": CaptionStyle(
            font="Helvetica Bold", size="32", color="white",
            outline="black", outline_width="2"
        ),
        "youtube": CaptionStyle(
            font="Roboto Bold", size="28", color="yellow",
            outline="black", outline_width="2"
        ),
        "youtube_shorts": CaptionStyle(
            font="Roboto Bold", size="28", color="yellow",
            outline="black", outline_width="2"
        )
    }
    return styles.get(platform, styles["tiktok"])

def get_platform_limits(platform: str) -> Dict[str, float]:
    """Get platform limits as dictionary"""
    limits = {
        "youtube_shorts": {"max_duration": 59.0, "ideal_clips": 6, "max_clip_duration": 8.0},
        "tiktok": {"max_duration": 59.0, "ideal_clips": 7, "max_clip_duration": 7.0},
        "instagram_reels": {"max_duration": 59.0, "ideal_clips": 6, "max_clip_duration": 8.0},
        "custom": {"max_duration": 300.0, "ideal_clips": 20, "max_clip_duration": 15.0}
    }
    return limits.get(platform, limits["custom"])

# Model rebuilds for forward references
EditorAgentResponse.model_rebuild()
Storyboard.model_rebuild()
TimelinePlan.model_rebuild()
StoryBeat.model_rebuild()
TimelineItem.model_rebuild()
VideoClipSchema.model_rebuild()
EditingProjectResponse.model_rebuild()
AdvancedEditResponse.model_rebuild()
BatchRenderResponse.model_rebuild()
RenderJobStatus.model_rebuild()
ProjectAnalysisResponse.model_rebuild()
PlatformLimits.model_rebuild()
PlatformOptimization.model_rebuild()