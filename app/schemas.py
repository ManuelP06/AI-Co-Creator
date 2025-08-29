from pydantic import BaseModel, Field
from typing import List, Optional, Literal


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

    class Config:
        from_attributes = True


class ShotListResponse(BaseModel):
    video_id: int
    shots: List[ShotResponse]


class TimelineItem(BaseModel):
    clip_id: int = Field(..., ge=0)
    order: int = Field(..., ge=1)
    transition: Literal["cut", "fade", "dissolve", "wipe"] = "cut"
    lut: Literal["none", "cinematic", "warm", "vivid", "bw"] = "none"
    speed: float = 1.0
    highlight_reason: str = ""


class StoryBeat(BaseModel):
    order: int
    title: str
    summary: str
    supporting_clips: List[int] = []


class Storyboard(BaseModel):
    theme: str
    beats: List["StoryBeat"] = Field(..., min_items=1)


class TimelinePlan(BaseModel):
    items: List["TimelineItem"] = Field(..., min_items=1)


class EditorAgentResponse(BaseModel):
    storyboard: Optional[Storyboard] = None
    timeline: TimelinePlan


# Resolve forward references
EditorAgentResponse.model_rebuild()
Storyboard.model_rebuild()
TimelinePlan.model_rebuild()
