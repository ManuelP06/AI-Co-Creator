from pydantic import BaseModel, Field, validator
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
    PODCAST_HIGHLIGHTS = "podcast_highlights"
    INTERVIEW_CLIPS = "interview_clips"
    EDUCATIONAL_SUMMARY = "educational_summary"
    ENTERTAINMENT_COMPILATION = "entertainment_compilation"
    PRODUCT_DEMO = "product_demo"
    TESTIMONIAL_REEL = "testimonial_reel"
    EVENT_HIGHLIGHTS = "event_highlights"

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
    
    @validator('end_time')
    def end_time_must_be_after_start(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v

class TimelineItem(BaseModel):
    """Individual timeline item"""
    clip_id: int = Field(..., ge=0)
    order: int = Field(..., ge=1)
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., gt=0.0)
    highlight_reason: str = ""
    transition_type: TransitionType = TransitionType.HARD_CUT
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be greater than start_time')
        return v

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

class CaptionStyle(BaseModel):
    """Caption styling options"""
    font: str = "Arial Bold"
    size: str = "32"
    color: str = "white"
    outline: str = "black"
    outline_width: str = "3"

class RenderRequest(BaseModel):
    """Request for video rendering"""
    video_id: int = Field(..., ge=1)
    output_path: str
    platform: Literal["tiktok", "instagram", "youtube"] = "tiktok"
    quality: Literal["low", "medium", "high"] = "high"
    use_gpu: bool = False
    auto_captions: bool = True
    caption_style: Optional[CaptionStyle] = None

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

class LegacyViralMetrics(BaseModel):
    """Legacy viral metrics format - DEPRECATED"""
    total_clips: int = Field(..., ge=0)
    total_duration: float = Field(..., ge=0.0)
    avg_viral_score: float = Field(..., ge=0.0, le=10.0)
    max_viral_score: float = Field(..., ge=0.0, le=10.0)
    hook_quality: float = Field(..., ge=0.0, le=10.0)
    viral_clips_count: int = Field(..., ge=0)
    duration_efficiency: float = Field(..., ge=0.0, le=1.0)
    optimal_for_viral: bool

class MultiVideoProjectRequest(BaseModel):
    """Request for multi-video project creation"""
    video_ids: List[int] = Field(..., min_items=1)
    content_type: ContentType
    target_platform: Literal["youtube_shorts", "tiktok", "instagram_reels", "custom"] = "youtube_shorts"
    brief: Optional[str] = None

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
        )
    }
    return styles.get(platform, styles["tiktok"])

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

class ProjectConfiguration(BaseModel):
    """Complete project configuration"""
    name: str
    content_type: ContentType
    target_platform: str
    strategy: EditingStrategy
    brief: Optional[str] = None
    custom_limits: Optional[Dict[str, float]] = None

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
    content_type: ContentType = ContentType.PODCAST_HIGHLIGHTS
    quality: Literal["low", "medium", "high"] = "high"
    use_gpu: bool = False
    output_directory: str

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

# Rebuild models to resolve forward references
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