from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from sqlalchemy.orm import Session
from ollama import Client

from app import models
from app.schemas import EditorAgentResponse, TimelinePlan, Storyboard, TimelineItem

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 180
OLLAMA_RETRIES = 3
TEMPERATURE = 0.25

PLATFORM_LIMITS = {
    "youtube_shorts": {"max_duration": 59.0, "ideal_clips": 6, "max_clip_duration": 8.0},
    "tiktok": {"max_duration": 59.0, "ideal_clips": 7, "max_clip_duration": 7.0},
    "instagram_reels": {"max_duration": 59.0, "ideal_clips": 6, "max_clip_duration": 8.0},
    "custom": {"max_duration": 300.0, "ideal_clips": 20, "max_clip_duration": 15.0}
}

logger = logging.getLogger(__name__)
client = Client(host=OLLAMA_HOST)


class ContentType(Enum):
    """Supported content types for different use cases"""
    INTERVIEW = "interview"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    PRODUCT_DEMO = "product_demo"


class TransitionType(Enum):
    """Professional video transitions"""
    HARD_CUT = "hard_cut"
    FADE = "fade"
    SLIDE = "slide"
    ZOOM = "zoom"


@dataclass
class VideoClip:
    """Clean video clip representation"""
    id: int
    source_path: str
    start_time: float
    end_time: float
    content_score: float = 0.0
    transcript: str = ""
    analysis: str = ""
    tags: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_path": self.source_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "content_score": self.content_score,
            "transcript": self.transcript,
            "summary": self.analysis,
            "tags": self.tags
        }


@dataclass
class EditingProject:
    """Project container for multi-video editing"""
    id: int
    name: str
    content_type: ContentType
    target_platform: str
    source_videos: List[Dict[str, Any]] = field(default_factory=list)
    clips: List[VideoClip] = field(default_factory=list)
    brief: Optional[str] = None
    
    def add_source_video(self, video_id: int, video_path: str, priority: int = 1) -> None:
        """Add source video to project"""
        self.source_videos.append({
            "id": video_id,
            "path": video_path,
            "priority": priority
        })
    
    def get_platform_limits(self) -> Dict[str, float]:
        """Get platform-specific limits"""
        return PLATFORM_LIMITS.get(self.target_platform, PLATFORM_LIMITS["custom"])


class ContentAnalyzer(ABC):
    """Base class for content analysis strategies"""
    
    @abstractmethod
    def calculate_content_score(self, clip: VideoClip) -> float:
        """Calculate content relevance score (0-10)"""
        pass
    
    @abstractmethod
    def get_selection_criteria(self) -> Dict[str, Any]:
        """Get criteria for clip selection"""
        pass


class ViralContentAnalyzer(ContentAnalyzer):
    """Analyzer for viral social media content"""
    
    def calculate_content_score(self, clip: VideoClip) -> float:
        content = f"{clip.transcript} {clip.analysis}".lower()
        
        viral_indicators = [
            "shocking", "unbelievable", "secret", "truth", "exposed",
            "amazing", "incredible", "transformation", "before after",
            "you won't believe", "wait for it", "plot twist",
            "breaking", "exclusive", "reaction", "drama"
        ]
        
        score = 0.0
        for indicator in viral_indicators:
            if indicator in content:
                score += 1.2
        
        if any(hook in content for hook in ["wait", "stop", "look", "imagine"]):
            score += 2.0
        
        return min(10.0, score)
    
    def get_selection_criteria(self) -> Dict[str, Any]:
        return {
            "min_score": 3.0,
            "prefer_hooks": True,
            "maintain_context": False,
            "ideal_clip_length": 3.0
        }


class EditorEngine:
    """Core editing engine with clean architecture"""
    
    def __init__(self, content_analyzer: ContentAnalyzer):
        self.analyzer = content_analyzer
        self.clips: List[VideoClip] = []
        self.selected_clips: List[VideoClip] = []
    
    def load_clips_from_shots(self, shots: List[Any], video_path: str) -> None:
        """Load clips from database shots"""
        self.clips = []
        
        for shot in shots:
            clip = VideoClip(
                id=shot.id,
                source_path=video_path,
                start_time=float(shot.start_time or 0.0),
                end_time=float(shot.end_time or 0.0),
                transcript=shot.transcript or "",
                analysis=shot.analysis or ""
            )
            clip.content_score = self.analyzer.calculate_content_score(clip)
            self.clips.append(clip)
    
    def load_clips_from_multiple_videos(self, video_shots_map: Dict[str, List[Any]]) -> None:
        """Load clips from multiple source videos"""
        self.clips = []
        
        for video_path, shots in video_shots_map.items():
            for shot in shots:
                clip = VideoClip(
                    id=shot.id,
                    source_path=video_path,
                    start_time=float(shot.start_time or 0.0),
                    end_time=float(shot.end_time or 0.0),
                    transcript=shot.transcript or "",
                    analysis=shot.analysis or ""
                )
                clip.content_score = self.analyzer.calculate_content_score(clip)
                self.clips.append(clip)
    
    def select_best_clips(self, target_duration: float, max_clips: int) -> List[VideoClip]:
        """Select best clips while maintaining narrative flow"""
        criteria = self.analyzer.get_selection_criteria()
        min_score = criteria.get("min_score", 2.0)
        maintain_context = criteria.get("maintain_context", True)
        
        candidates = [c for c in self.clips if c.content_score >= min_score]
        
        if maintain_context:
            candidates.sort(key=lambda c: (c.source_path, c.start_time))
        else:
            candidates.sort(key=lambda c: c.content_score, reverse=True)
        
        selected = []
        total_duration = 0.0
        
        for clip in candidates:
            if len(selected) >= max_clips:
                break
            
            if total_duration + clip.duration <= target_duration:
                selected.append(clip)
                total_duration += clip.duration
            elif target_duration - total_duration > 0.5:  
                remaining_time = target_duration - total_duration
                trimmed_clip = VideoClip(
                    id=clip.id,
                    source_path=clip.source_path,
                    start_time=clip.start_time,
                    end_time=clip.start_time + remaining_time,
                    content_score=clip.content_score,
                    transcript=clip.transcript,
                    analysis=clip.analysis,
                    tags=clip.tags + ["trimmed"]
                )
                selected.append(trimmed_clip)
                break
        
        self.selected_clips = selected
        return selected
    
    def generate_timeline(self) -> Dict[str, Any]:
        """Generate clean timeline structure"""
        items = []
        
        for i, clip in enumerate(self.selected_clips):
            items.append({
                "clip_id": clip.id,
                "order": i + 1,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "source_path": clip.source_path,
                "highlight_reason": f"Score: {clip.content_score:.1f}",
                "transition_type": TransitionType.HARD_CUT.value
            })
        
        total_duration = sum(clip.duration for clip in self.selected_clips)
        
        return {
            "total_duration": total_duration,
            "clip_count": len(self.selected_clips),
            "items": items
        }


class PromptTemplateManager:
    """Manages prompts for different content types"""
    
    @staticmethod
    def get_base_prompt() -> str:
        return """
You are a professional video editor AI creating engaging short-form content.

CORE PRINCIPLES:
1. Maintain narrative coherence and logical flow
2. Select the most engaging and valuable moments
3. Respect platform duration limits
4. Create content that provides genuine value to viewers
5. Ensure smooth transitions between clips

QUALITY STANDARDS:
- Each clip must serve a purpose in the overall narrative
- Maintain consistent pacing throughout the video
- Ensure clear audio and visual quality
- Create compelling hooks while staying authentic
- End with strong calls-to-action or memorable moments
"""
    
    @staticmethod
    def get_content_specific_prompt(content_type: ContentType) -> str:
        prompts = {
            ContentType.INTERVIEW: """
INTERVIEW CLIPS STRATEGY:
- Capture the most insightful questions and answers
- Show dynamic between interviewer and guest
- Include moments that reveal personality or expertise
- Maintain professional tone while being engaging
- Focus on actionable insights and key takeaways
""",
            ContentType.EDUCATIONAL: """
EDUCATIONAL CONTENT STRATEGY:
- Break down complex topics into digestible segments
- Maintain logical learning progression
- Include clear explanations and examples
- Ensure each clip builds on the previous one
- End with actionable next steps or key principles
"""
        }
        return prompts.get(content_type, prompts[ContentType.INTERVIEW])


class LLMService:
    """Service for LLM interactions with improved error handling"""
    
    def __init__(self, client: Client):
        self.client = client
    
    def generate_edit_plan(self, 
                          clips: List[VideoClip], 
                          project: EditingProject) -> Dict[str, Any]:
        """Generate editing plan using LLM"""
        
        prompt = self._build_editing_prompt(clips, project)
        
        for attempt in range(OLLAMA_RETRIES + 1):
            try:
                response = self.client.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(project.content_type)
                        },
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": TEMPERATURE,
                        "num_ctx": 8192,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    },
                    format="json"
                )
                
                raw_content = response.get("message", {}).get("content", "")
                return self._parse_llm_response(raw_content)
                
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < OLLAMA_RETRIES:
                    time.sleep(1.0 + attempt)
                else:
                    raise RuntimeError(f"LLM service failed after {OLLAMA_RETRIES} retries: {e}")
    
    def _get_system_prompt(self, content_type: ContentType) -> str:
        """Get system prompt based on content type"""
        base = PromptTemplateManager.get_base_prompt()
        specific = PromptTemplateManager.get_content_specific_prompt(content_type)
        return f"{base}\n\n{specific}"
    
    def _build_editing_prompt(self, clips: List[VideoClip], project: EditingProject) -> str:
        """Build comprehensive editing prompt"""
        limits = project.get_platform_limits()
        
        clips_summary = "\n".join([
            f"Clip {clip.id}: {clip.start_time:.1f}s-{clip.end_time:.1f}s "
            f"(Score: {clip.content_score:.1f}) - {clip.transcript[:100]}..."
            for clip in clips[:20]  # Limit for context
        ])
        
        schema_example = {
            "storyboard": {
                "theme": "content_theme",
                "target_audience": "target_demographic", 
                "strategy": "editing_approach",
                "narrative_arc": "beginning_middle_end_structure",
                "beats": [
                    {
                        "order": 1,
                        "title": "Opening Hook",
                        "summary": "clip_purpose_and_content",
                        "supporting_clips": [1, 2],
                        "duration_target": 5.0
                    }
                ]
            },
            "timeline": {
                "total_duration": 45.5,
                "items": [
                    {
                        "clip_id": 1,
                        "order": 1,
                        "start_time": 12.5,
                        "end_time": 17.5,
                        "highlight_reason": "strong_opening_statement",
                        "transition_type": "hard_cut"
                    }
                ]
            }
        }
        
        return f"""
PROJECT DETAILS:
- Content Type: {project.content_type.value}
- Platform: {project.target_platform}
- Max Duration: {limits['max_duration']}s
- Ideal Clips: {limits['ideal_clips']}
- Brief: {project.brief or 'Create engaging highlight reel'}

AVAILABLE CLIPS:
{clips_summary}

REQUIREMENTS:
1. Create a coherent narrative that flows naturally
2. Select clips that work well together
3. Maintain chronological order when logical
4. Ensure total duration stays under {limits['max_duration']}s
5. Focus on highest-scoring content while maintaining flow

OUTPUT FORMAT (JSON only):
{json.dumps(schema_example, indent=2)}

Return valid JSON with both 'storyboard' and 'timeline' keys.
"""
    
    def _parse_llm_response(self, raw_content: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Try direct JSON parse first
            return json.loads(raw_content)
        except json.JSONDecodeError:
            # Extract JSON from markdown or mixed content
            import re
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError("No valid JSON found in LLM response")


class VideoEditor:    
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService(client)
        self.analyzer_map = {
            ContentType.INTERVIEW: ViralContentAnalyzer(),
            ContentType.EDUCATIONAL: ViralContentAnalyzer(), 
            ContentType.ENTERTAINMENT: ViralContentAnalyzer(),
            ContentType.PRODUCT_DEMO: ViralContentAnalyzer(),
        }
    
    def create_project(self, 
                      name: str,
                      content_type: ContentType,
                      target_platform: str,
                      brief: Optional[str] = None) -> EditingProject:
        """Create new editing project"""
        project = EditingProject(
            id=int(time.time()),
            name=name,
            content_type=content_type,
            target_platform=target_platform,
            brief=brief
        )
        return project
    
    def add_video_to_project(self, project: EditingProject, video_id: int) -> None:
        """Add video source to project"""
        video = self.db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        project.add_source_video(video_id, video.file_path)
        
        shots = (
            self.db.query(models.Shot)
            .filter(models.Shot.video_id == video_id)
            .order_by(models.Shot.shot_index.asc())
            .all()
        )
        
        if not shots:
            raise ValueError(f"No shots found for video {video_id}. Run shot detection first.")
        
        analyzer = self.analyzer_map.get(project.content_type, ViralContentAnalyzer())
        
        for shot in shots:
            clip = VideoClip(
                id=shot.id,
                source_path=video.file_path,
                start_time=float(shot.start_time or 0.0),
                end_time=float(shot.end_time or 0.0),
                transcript=shot.transcript or "",
                analysis=shot.analysis or ""
            )
            clip.content_score = analyzer.calculate_content_score(clip)
            project.clips.append(clip)
    
    def generate_edit(self, project: EditingProject) -> EditorAgentResponse:
        """Generate final edit plan"""
        if not project.clips:
            raise ValueError("No clips available for editing")
        
        limits = project.get_platform_limits()
        
        analyzer = self.analyzer_map.get(project.content_type, ViralContentAnalyzer())
        engine = EditorEngine(analyzer)
        engine.clips = project.clips
        
        selected_clips = engine.select_best_clips(
            target_duration=limits["max_duration"],
            max_clips=limits["ideal_clips"]
        )
        
        if not selected_clips:
            raise ValueError("No clips met selection criteria")
        
        llm_response = self.llm_service.generate_edit_plan(selected_clips, project)
        
        timeline_data = llm_response.get("timeline", {})
        storyboard_data = llm_response.get("storyboard")
        
        timeline_items = []
        for item_data in timeline_data.get("items", []):
            timeline_item = TimelineItem(
                clip_id=item_data["clip_id"],
                order=item_data["order"],
                start_time=item_data["start_time"],
                end_time=item_data["end_time"],
                highlight_reason=item_data.get("highlight_reason", "Selected content")
            )
            timeline_items.append(timeline_item)
        
        total_duration = sum(item.end_time - item.start_time for item in timeline_items)
        
        timeline = TimelinePlan(
            total_duration=total_duration,
            items=timeline_items
        )
        
        storyboard = None
        if storyboard_data:
            try:
                storyboard = Storyboard.model_validate(storyboard_data)
            except Exception as e:
                logger.warning(f"Storyboard validation failed: {e}")
        
        self._save_project_results(project, timeline, storyboard)
        
        logger.info(f"Edit generated: {len(timeline_items)} clips, {total_duration:.1f}s duration")
        
        return EditorAgentResponse(
            storyboard=storyboard,
            timeline=timeline
        )
    
    def _save_project_results(self, 
                             project: EditingProject, 
                             timeline: TimelinePlan, 
                             storyboard: Optional[Storyboard]) -> None:
        """Save project results to database"""
        # For multi-video projects, save to the first video for now
        # In production, we need a separate projects table
        if project.source_videos:
            video_id = project.source_videos[0]["id"]
            video = self.db.query(models.Video).filter(models.Video.id == video_id).first()
            
            if video:
                video.timeline_json = timeline.model_dump()
                if storyboard:
                    video.storyboard_json = storyboard.model_dump()
                
                self.db.add(video)
                self.db.commit()


def run_editor_agent(db: Session, video_id: int, user_brief: Optional[str] = None) -> EditorAgentResponse:
    """Legacy function - creates single video reel"""
    editor = VideoEditor(db)
    
    content_type = ContentType.INTERVIEW
    if user_brief:
        brief_lower = user_brief.lower()
        if "entertainment" in brief_lower:
            content_type = ContentType.ENTERTAINMENT
        elif "education" in brief_lower:
            content_type = ContentType.EDUCATIONAL
        elif "interview" in brief_lower:
            content_type = ContentType.INTERVIEW
        elif "product_demo" in brief_lower:
            content_type = ContentType.PRODUCT_DEMO
    
    project = editor.create_project(
        name=f"Video_{video_id}_highlights",
        content_type=content_type,
        target_platform="youtube_shorts",
        brief=user_brief
    )
    
    editor.add_video_to_project(project, video_id)
    
    return editor.generate_edit(project)


def get_timeline_json(db: Session, video_id: int) -> Dict[str, Any]:
    """Get timeline JSON for video"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise ValueError("Video not found")
    if not video.timeline_json:
        raise ValueError("No timeline found for video")
    return video.timeline_json


def analyze_content_potential(db: Session, video_id: int) -> Dict[str, Any]:
    """Analyze content potential with clean metrics"""
    shots = (
        db.query(models.Shot)
        .filter(models.Shot.video_id == video_id)
        .order_by(models.Shot.shot_index.asc())
        .all()
    )
    
    if not shots:
        return {"error": "No shots found"}
    
    analyzer = ViralContentAnalyzer()
    
    clips = []
    for shot in shots:
        clip = VideoClip(
            id=shot.id,
            source_path="",
            start_time=float(shot.start_time or 0.0),
            end_time=float(shot.end_time or 0.0),
            transcript=shot.transcript or "",
            analysis=shot.analysis or ""
        )
        clip.content_score = analyzer.calculate_content_score(clip)
        clips.append(clip)
    
    scores = [clip.content_score for clip in clips]
    total_duration = sum(clip.duration for clip in clips)
    
    return {
        "total_clips": len(clips),
        "total_duration": total_duration,
        "average_score": sum(scores) / len(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "high_quality_clips": len([s for s in scores if s >= 5.0]),
        "usable_clips": len([s for s in scores if s >= 2.0]),
        "content_density": len([s for s in scores if s >= 3.0]) / len(scores) if scores else 0
    }


def create_multi_video_project(db: Session,
                              video_ids: List[int],
                              content_type: str,
                              target_platform: str = "youtube_shorts",
                              brief: Optional[str] = None) -> Dict[str, Any]:
    """Create project from multiple videos"""
    editor = VideoEditor(db)
    
    try:
        content_enum = ContentType(content_type)
    except ValueError:
        content_enum = ContentType.PODCAST_HIGHLIGHTS
    
    project = editor.create_project(
        name=f"Multi_video_project_{int(time.time())}",
        content_type=content_enum,
        target_platform=target_platform,
        brief=brief
    )
    
    for video_id in video_ids:
        editor.add_video_to_project(project, video_id)
    
    result = editor.generate_edit(project)
    
    return {
        "project_id": project.id,
        "source_videos": len(project.source_videos),
        "total_clips": len(project.clips),
        "selected_clips": len(result.timeline.items) if result.timeline else 0,
        "final_duration": result.timeline.total_duration if result.timeline else 0,
        "content_type": content_type,
        "platform": target_platform
    }


def get_project_analytics(db: Session, video_ids: List[int]) -> Dict[str, Any]:
    """Get analytics for multi-video project"""
    total_clips = 0
    total_duration = 0.0
    quality_scores = []
    
    for video_id in video_ids:
        shots = db.query(models.Shot).filter(models.Shot.video_id == video_id).all()
        
        analyzer = ViralContentAnalyzer()
        
        for shot in shots:
            clip = VideoClip(
                id=shot.id,
                source_path="",
                start_time=float(shot.start_time or 0.0),
                end_time=float(shot.end_time or 0.0),
                transcript=shot.transcript or "",
                analysis=shot.analysis or ""
            )
            score = analyzer.calculate_content_score(clip)
            quality_scores.append(score)
            total_clips += 1
            total_duration += clip.duration
    
    return {
        "source_videos": len(video_ids),
        "total_source_clips": total_clips,
        "total_source_duration": total_duration,
        "average_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
        "high_quality_clips": len([s for s in quality_scores if s >= 5.0]),
        "recommended_highlights": min(total_clips, 10),
        "estimated_output_duration": min(total_duration * 0.15, 59.0)  
    }