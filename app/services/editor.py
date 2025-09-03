from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import statistics
import re

from sqlalchemy.orm import Session
from ollama import Client

from app import models
from app.schemas import EditorAgentResponse, TimelinePlan, Storyboard, TimelineItem

# Configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 300
OLLAMA_RETRIES = 3
TEMPERATURE = 0.1

PLATFORM_CONFIGS = {
    "tiktok": {
        "max_duration": 60.0,
        "optimal_duration": 60.0,
        "ideal_clips": 8,
        "max_clip_duration": 12.0,
        "hook_duration": 3.0,
        "pacing": "fast",
        "aspect_ratio": "9:16",
        "engagement_keywords": ["viral", "trending", "crazy", "shocking", "secret"]
    },
    "youtube_shorts": {
        "max_duration": 60.0,
        "optimal_duration": 45.0,
        "ideal_clips": 6,
        "max_clip_duration": 15.0,
        "hook_duration": 5.0,
        "pacing": "medium",
        "aspect_ratio": "9:16",
        "engagement_keywords": ["discover", "learn", "revealed", "truth", "exposed"]
    },
    "instagram_reels": {
        "max_duration": 90.0,
        "optimal_duration": 30.0,
        "ideal_clips": 5,
        "max_clip_duration": 10.0,
        "hook_duration": 2.0,
        "pacing": "fast",
        "aspect_ratio": "9:16",
        "engagement_keywords": ["aesthetic", "trending", "viral", "relatable", "mood"]
    },
    "linkedin": {
        "max_duration": 300.0,
        "optimal_duration": 120.0,
        "ideal_clips": 8,
        "max_clip_duration": 30.0,
        "hook_duration": 10.0,
        "pacing": "slow",
        "aspect_ratio": "16:9",
        "engagement_keywords": ["insights", "professional", "strategy", "growth", "leadership"]
    }
}

logger = logging.getLogger(__name__)
client = Client(host=OLLAMA_HOST)


class ContentType(Enum):
    INTERVIEW = "interview"
    PODCAST = "podcast"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    DOCUMENTARY = "documentary"


@dataclass
class VideoClip:
    """Video clip with quality metrics"""
    id: int
    source_path: str
    start_time: float
    end_time: float
    transcript: str = ""
    analysis: str = ""
    
    content_score: float = 0.0
    engagement_score: float = 0.0
    audio_quality: float = 5.0  # Default reasonable quality
    visual_quality: float = 5.0
    
    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)
    
    @property
    def composite_score(self) -> float:
        """Calculate weighted composite score"""
        return (
            self.content_score * 0.4 +
            self.engagement_score * 0.3 +
            self.audio_quality * 0.15 +
            self.visual_quality * 0.15
        )
    
    def calculate_hook_potential(self) -> float:
        """Calculate potential as opening hook"""
        hook_keywords = ["wait", "stop", "look", "imagine", "what if", "secret", "truth"]
        text_lower = f"{self.transcript} {self.analysis}".lower()
        
        hook_score = sum(2.0 for keyword in hook_keywords if keyword in text_lower)
        return min(10.0, hook_score + self.engagement_score * 0.5)


@dataclass
class EditingProject:
    id: int
    name: str
    content_type: ContentType
    target_platform: str
    clips: List[VideoClip] = field(default_factory=list)
    brief: Optional[str] = None
    
    # Quality thresholds
    min_content_score: float = 3.0
    min_engagement_score: float = 2.0
    min_composite_score: float = 4.0
    
    def get_platform_config(self) -> Dict[str, Any]:
        return PLATFORM_CONFIGS.get(self.target_platform, PLATFORM_CONFIGS["youtube_shorts"])


class ContentAnalyzer:
    """Unified content analyzer"""
    
    def __init__(self, content_type: ContentType):
        self.content_type = content_type
        self.engagement_patterns = self._get_engagement_patterns()
    
    def _get_engagement_patterns(self) -> List[str]:
        """Get engagement patterns by content type"""
        patterns = {
            ContentType.INTERVIEW: ["question", "answer", "insight", "story", "advice", "truth"],
            ContentType.EDUCATIONAL: ["learn", "discover", "explain", "show", "example", "tip"],
            ContentType.ENTERTAINMENT: ["funny", "crazy", "amazing", "shocking", "epic", "reaction"],
            ContentType.DOCUMENTARY: ["fact", "evidence", "truth", "reveal", "expose", "secret"]
        }
        return patterns.get(self.content_type, patterns[ContentType.INTERVIEW])
    
    def analyze_clip(self, clip: VideoClip) -> VideoClip:
        """Analyze clip and calculate scores"""
        clip.content_score = self._calculate_content_score(clip)
        clip.engagement_score = self._calculate_engagement_score(clip)
        clip.audio_quality = self._estimate_audio_quality(clip)
        clip.visual_quality = self._estimate_visual_quality(clip)
        return clip
    
    def _calculate_content_score(self, clip: VideoClip) -> float:
        """Calculate content relevance score"""
        content = f"{clip.transcript} {clip.analysis}".lower()
        
        # Pattern matching
        pattern_score = sum(1.5 for pattern in self.engagement_patterns if pattern in content)
        
        # Hook indicators
        hook_indicators = ["wait", "stop", "look", "imagine", "what if", "did you know"]
        hook_score = sum(2.0 for hook in hook_indicators if hook in content)
        
        # Filler word penalty
        filler_words = ["um", "uh", "like", "you know", "basically"]
        filler_penalty = sum(0.3 for filler in filler_words if content.count(filler) > 2)
        
        # Duration factor
        duration_factor = 1.0
        if clip.duration < 2.0:
            duration_factor = 0.7
        elif clip.duration > 15.0:
            duration_factor = 0.8
        
        score = (pattern_score + hook_score - filler_penalty) * duration_factor
        return max(0.0, min(10.0, score))
    
    def _calculate_engagement_score(self, clip: VideoClip) -> float:
        """Calculate engagement potential"""
        content = f"{clip.transcript} {clip.analysis}".lower()
        
        viral_words = [
            ("shocking", 3.0), ("unbelievable", 2.5), ("secret", 2.0), ("truth", 2.0),
            ("exposed", 2.5), ("revealed", 2.0), ("crazy", 2.5), ("amazing", 1.5)
        ]
        
        engagement_score = sum(weight for word, weight in viral_words if word in content)
        
        # Question bonus
        if any(q in content for q in ["?", "what", "why", "how"]):
            engagement_score += 1.5
        
        # Emotion indicators
        emotions = ["angry", "sad", "happy", "excited", "shocked", "surprised"]
        emotion_score = sum(1.0 for emotion in emotions if emotion in content)
        
        return min(10.0, engagement_score + emotion_score)
    
    def _estimate_audio_quality(self, clip: VideoClip) -> float:
        """Estimate audio quality based on transcript"""
        if not clip.transcript.strip():
            return 2.0
        
        word_count = len(clip.transcript.split())
        if clip.duration == 0:
            return 2.0
        
        words_per_second = word_count / clip.duration
        
        # Base score from speech clarity
        if 1.5 <= words_per_second <= 3.5:
            base_score = 7.0
        elif words_per_second < 1.0:
            base_score = 4.0  # Too slow/unclear
        elif words_per_second > 5.0:
            base_score = 4.0  # Too fast/unclear
        else:
            base_score = 6.0
        
        # Penalty for transcription errors
        error_indicators = ["[inaudible]", "[unclear]", "um um", "uh uh"]
        error_penalty = sum(1.0 for error in error_indicators if error in clip.transcript.lower())
        
        return max(1.0, min(10.0, base_score - error_penalty))
    
    def _estimate_visual_quality(self, clip: VideoClip) -> float:
        """Estimate visual quality (basic heuristic)"""
        # Basic quality estimation - can be enhanced with actual video analysis
        base_score = 6.0
        
        # Duration-based adjustments
        if clip.duration < 1.0:
            base_score -= 2.0  # Too short might indicate issues
        elif 20.0 < clip.duration < 30.0:
            base_score -= 0.5  # Very long clips might have quality issues
        
        return max(1.0, min(10.0, base_score))


class ClipSelector:
    """Handles intelligent clip selection"""
    
    def __init__(self, project: EditingProject):
        self.project = project
        self.config = project.get_platform_config()
    
    def select_clips(self, clips: List[VideoClip]) -> List[VideoClip]:
        """Select best clips for timeline"""
        if not clips:
            return []
        
        # Filter by quality thresholds
        quality_clips = [
            clip for clip in clips
            if (clip.content_score >= self.project.min_content_score and
                clip.engagement_score >= self.project.min_engagement_score and
                clip.composite_score >= self.project.min_composite_score)
        ]
        
        if not quality_clips:
            # Fallback to best available clips
            quality_clips = sorted(clips, key=lambda c: c.composite_score, reverse=True)[:self.config["ideal_clips"]]
        
        # Sort by composite score
        quality_clips.sort(key=lambda c: c.composite_score, reverse=True)
        
        # Select clips optimizing for duration and variety
        selected = self._optimize_selection(quality_clips)
        
        return selected[:self.config["ideal_clips"]]
    
    def _optimize_selection(self, clips: List[VideoClip]) -> List[VideoClip]:
        """Optimize clip selection for duration and variety"""
        selected = []
        total_duration = 0.0
        max_duration = self.config["max_duration"]
        
        # Ensure we have a good hook
        hook_clips = [c for c in clips if c.calculate_hook_potential() >= 5.0]
        if hook_clips and not selected:
            best_hook = max(hook_clips, key=lambda c: c.calculate_hook_potential())
            selected.append(best_hook)
            total_duration += best_hook.duration
            clips.remove(best_hook)
        
        # Add remaining clips by score, respecting duration
        for clip in clips:
            if total_duration + clip.duration <= max_duration:
                selected.append(clip)
                total_duration += clip.duration
            elif max_duration - total_duration > 1.0:
                # Try to fit a shorter segment
                remaining_time = max_duration - total_duration
                if clip.duration > remaining_time:
                    trimmed_clip = self._trim_clip(clip, remaining_time)
                    if trimmed_clip:
                        selected.append(trimmed_clip)
                break
        
        return selected
    
    def _trim_clip(self, clip: VideoClip, target_duration: float) -> Optional[VideoClip]:
        """Trim clip to target duration"""
        if target_duration < 1.0:
            return None
        
        # Create trimmed version
        trimmed = VideoClip(
            id=clip.id,
            source_path=clip.source_path,
            start_time=clip.start_time,
            end_time=clip.start_time + target_duration,
            transcript=clip.transcript[:int(len(clip.transcript) * target_duration / clip.duration)],
            analysis=clip.analysis,
            content_score=clip.content_score * 0.9,  # Slight penalty for trimming
            engagement_score=clip.engagement_score,
            audio_quality=clip.audio_quality,
            visual_quality=clip.visual_quality
        )
        
        return trimmed


class LLMService:
    
    def __init__(self):
        self.client = client
    
    def generate_storyboard(self, clips: List[VideoClip], project: EditingProject) -> Dict[str, Any]:
        """Generate storyboard using LLM"""
        prompt = self._build_prompt(clips, project)
        system_prompt = self._get_system_prompt(project)
        
        for attempt in range(OLLAMA_RETRIES):
            try:
                response = self.client.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    options={
                        "temperature": TEMPERATURE,
                        "num_ctx": 8192,
                        "top_p": 0.85
                    },
                    format="json"
                )
                
                content = response.get("message", {}).get("content", "")
                return self._parse_response(content, clips, project)
                
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < OLLAMA_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    return self._generate_fallback(clips, project)
    
    def _build_prompt(self, clips: List[VideoClip], project: EditingProject) -> str:
        """Build concise prompt"""
        config = project.get_platform_config()
        
        # Summarize clips concisely
        clip_summaries = []
        for clip in clips[:10]:  # Limit for context
            summary = (
                f"Clip {clip.id}: {clip.duration:.1f}s "
                f"(Score: {clip.composite_score:.1f}) - "
                f"{clip.transcript[:100]}..."
            )
            clip_summaries.append(summary)
        
        return f"""
Create a {project.target_platform} video edit plan for {project.content_type.value} content.

SPECIFICATIONS:
- Target Duration: {config['optimal_duration']}s
- Hook Duration: {config['hook_duration']}s  
- Ideal Clips: {config['ideal_clips']}
- Platform: {project.target_platform}

AVAILABLE CLIPS:
{chr(10).join(clip_summaries)}

BRIEF: {project.brief or 'Create engaging highlight reel'}

OUTPUT JSON FORMAT:
{{
    "storyboard": {{
        "theme": "content theme",
        "target_audience": "audience description", 
        "strategy": "editing approach",
        "narrative_arc": "story structure"
    }},
    "timeline": {{
        "items": [
            {{
                "clip_id": 1,
                "order": 1,
                "start_time": 12.5,
                "end_time": 17.5,
                "highlight_reason": "why this clip"
            }}
        ]
    }}
}}
"""
    
    def _get_system_prompt(self, project: EditingProject) -> str:
        """Get system prompt"""
        return f"""You are a professional video editor specializing in {project.target_platform} content.
Create engaging, high-quality edits that maximize viewer retention and engagement.
Focus on strong hooks, clear narrative flow, and platform optimization.
Always return valid JSON only."""
    
    def _parse_response(self, content: str, clips: List[VideoClip], project: EditingProject) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            data = json.loads(content)
            
            # Validate structure
            if "storyboard" not in data:
                data["storyboard"] = self._create_default_storyboard(project)
            
            if "timeline" not in data or "items" not in data["timeline"]:
                data["timeline"] = {"items": self._create_default_timeline(clips)}
            
            # Validate clip IDs
            available_ids = {clip.id for clip in clips}
            valid_items = []
            
            for item in data["timeline"]["items"]:
                if item.get("clip_id") in available_ids:
                    valid_items.append(item)
            
            data["timeline"]["items"] = valid_items
            return data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response")
            return self._generate_fallback(clips, project)
    
    def _create_default_storyboard(self, project: EditingProject) -> Dict[str, Any]:
        """Create default storyboard"""
        return {
            "theme": f"{project.content_type.value} highlights",
            "target_audience": "General audience",
            "strategy": "Best moments compilation",
            "narrative_arc": "Hook → Build → Climax"
        }
    
    def _create_default_timeline(self, clips: List[VideoClip]) -> List[Dict[str, Any]]:
        """Create default timeline"""
        items = []
        for i, clip in enumerate(clips[:6]):
            items.append({
                "clip_id": clip.id,
                "order": i + 1,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "highlight_reason": f"High quality content (Score: {clip.composite_score:.1f})"
            })
        return items
    
    def _generate_fallback(self, clips: List[VideoClip], project: EditingProject) -> Dict[str, Any]:
        """Generate fallback response"""
        return {
            "storyboard": self._create_default_storyboard(project),
            "timeline": {"items": self._create_default_timeline(clips)}
        }


class VideoEditor:
    """Main video editor class"""
    
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService()
    
    def create_edit(self, video_id: int, content_type: ContentType, target_platform: str, 
                   brief: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> EditorAgentResponse:
        """Create video edit"""
        
        # Load video and shots
        video = self.db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        shots = (
            self.db.query(models.Shot)
            .filter(models.Shot.video_id == video_id)
            .order_by(models.Shot.shot_index.asc())
            .all()
        )
        
        if not shots:
            raise ValueError(f"No shots found for video {video_id}")
        
        # Create project
        project = EditingProject(
            id=int(time.time() * 1000),
            name=f"Video_{video_id}_Edit",
            content_type=content_type,
            target_platform=target_platform,
            brief=brief
        )
        
        # Apply custom options
        if options:
            project.min_content_score = options.get("min_content_score", 3.0)
            project.min_engagement_score = options.get("min_engagement_score", 2.0)
            project.min_composite_score = options.get("min_composite_score", 4.0)
        
        # Analyze clips
        analyzer = ContentAnalyzer(content_type)
        clips = []
        
        for shot in shots:
            if not self._validate_shot(shot):
                continue
                
            clip = VideoClip(
                id=shot.id,
                source_path=video.file_path,
                start_time=float(shot.start_time),
                end_time=float(shot.end_time),
                transcript=getattr(shot, 'transcript', '') or "",
                analysis=getattr(shot, 'analysis', '') or ""
            )
            
            analyzed_clip = analyzer.analyze_clip(clip)
            clips.append(analyzed_clip)
        
        project.clips = clips
        
        if not clips:
            raise ValueError("No valid clips found after analysis")
        
        # Select best clips
        selector = ClipSelector(project)
        selected_clips = selector.select_clips(clips)
        
        if not selected_clips:
            raise ValueError("No clips met selection criteria")
        
        # Generate timeline
        timeline_items = []
        for i, clip in enumerate(selected_clips):
            timeline_item = TimelineItem(
                clip_id=clip.id,
                order=i + 1,
                start_time=clip.start_time,
                end_time=clip.end_time,
                highlight_reason=self._get_highlight_reason(clip)
            )
            timeline_items.append(timeline_item)
        
        timeline = TimelinePlan(
            total_duration=sum(c.duration for c in selected_clips),
            items=timeline_items
        )
        
        # Generate storyboard
        llm_response = self.llm_service.generate_storyboard(selected_clips, project)
        storyboard_data = llm_response.get("storyboard", {})
        
        storyboard = Storyboard(
            theme=storyboard_data.get("theme", f"{content_type.value} highlights"),
            target_audience=storyboard_data.get("target_audience", "General audience"),
            strategy=storyboard_data.get("strategy", "Best moments compilation"),
            narrative_arc=storyboard_data.get("narrative_arc", "Hook → Build → Climax"),
            beats=[]  # Simplified - no complex beat structure
        )
        
        # Save results
        self._save_results(video, timeline, storyboard)
        
        logger.info(f"Edit created: {len(selected_clips)} clips, {timeline.total_duration:.1f}s")
        
        return EditorAgentResponse(storyboard=storyboard, timeline=timeline)
    
    def _validate_shot(self, shot) -> bool:
        """Validate shot data"""
        required_attrs = ['id', 'start_time', 'end_time']
        
        for attr in required_attrs:
            if not hasattr(shot, attr) or getattr(shot, attr) is None:
                return False
        
        if shot.end_time <= shot.start_time:
            return False
        
        return True
    
    def _get_highlight_reason(self, clip: VideoClip) -> str:
        """Generate highlight reason"""
        reasons = []
        
        if clip.content_score >= 7.0:
            reasons.append("High-value content")
        if clip.engagement_score >= 6.0:
            reasons.append("High engagement")
        if clip.calculate_hook_potential() >= 5.0:
            reasons.append("Strong hook potential")
        
        if not reasons:
            reasons.append(f"Quality content (Score: {clip.composite_score:.1f})")
        
        return " • ".join(reasons)
    
    def _save_results(self, video: models.Video, timeline: TimelinePlan, storyboard: Storyboard) -> None:
        """Save editing results"""
        video.timeline_json = timeline.model_dump()
        video.storyboard_json = storyboard.model_dump()
        
        self.db.add(video)
        self.db.commit()
    
    def analyze_content_potential(self, video_id: int) -> Dict[str, Any]:
        """Analyze content potential"""
        video = self.db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")
        
        shots = self.db.query(models.Shot).filter(models.Shot.video_id == video_id).all()
        if not shots:
            raise ValueError("No shots found")
        
        # Analyze all clips
        analyzer = ContentAnalyzer(ContentType.INTERVIEW)  # Default
        clips = []
        
        for shot in shots:
            if not self._validate_shot(shot):
                continue
                
            clip = VideoClip(
                id=shot.id,
                source_path=video.file_path,
                start_time=float(shot.start_time),
                end_time=float(shot.end_time),
                transcript=getattr(shot, 'transcript', '') or "",
                analysis=getattr(shot, 'analysis', '') or ""
            )
            
            analyzed_clip = analyzer.analyze_clip(clip)
            clips.append(analyzed_clip)
        
        if not clips:
            return {"error": "No valid clips found"}
        
        # Calculate metrics
        scores = [c.composite_score for c in clips]
        
        return {
            "total_clips": len(clips),
            "total_duration": sum(c.duration for c in clips),
            "average_score": statistics.mean(scores),
            "max_score": max(scores),
            "high_quality_clips": len([c for c in clips if c.composite_score >= 7.0]),
            "usable_clips": len([c for c in clips if c.composite_score >= 4.0]),
            "content_density": len([c for c in clips if c.composite_score >= 4.0]) / len(clips),
            "viral_potential": max([c.calculate_hook_potential() for c in clips]),
            "recommendations": self._generate_recommendations(clips)
        }
    
    def _generate_recommendations(self, clips: List[VideoClip]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        avg_quality = statistics.mean([c.composite_score for c in clips])
        if avg_quality < 5.0:
            recommendations.append("Overall content quality is low - consider better source material")
        
        hook_clips = [c for c in clips if c.calculate_hook_potential() >= 6.0]
        if len(hook_clips) < 2:
            recommendations.append("Limited hook potential - add more attention-grabbing moments")
        
        high_engagement = [c for c in clips if c.engagement_score >= 6.0]
        if len(high_engagement) < 3:
            recommendations.append("Add more emotionally compelling content")
        
        return recommendations


# Main entry points
def run_editor_agent(db: Session, video_id: int, user_brief: Optional[str] = None,
                    advanced_options: Optional[Dict[str, Any]] = None) -> EditorAgentResponse:
    """Main entry point for editor agent"""
    
    # Determine content type from brief
    content_type = ContentType.INTERVIEW
    if user_brief:
        brief_lower = user_brief.lower()
        if any(word in brief_lower for word in ["entertainment", "funny", "comedy"]):
            content_type = ContentType.ENTERTAINMENT
        elif any(word in brief_lower for word in ["education", "learn", "teach"]):
            content_type = ContentType.EDUCATIONAL
        elif any(word in brief_lower for word in ["documentary", "investigation"]):
            content_type = ContentType.DOCUMENTARY
    
    # Get platform from options or default
    target_platform = "youtube_shorts"
    if advanced_options and "target_platform" in advanced_options:
        target_platform = advanced_options["target_platform"]
    
    editor = VideoEditor(db)
    return editor.create_edit(
        video_id=video_id,
        content_type=content_type,
        target_platform=target_platform,
        brief=user_brief,
        options=advanced_options
    )


def analyze_content_potential(db: Session, video_id: int) -> Dict[str, Any]:
    """Analyze content potential"""
    editor = VideoEditor(db)
    return editor.analyze_content_potential(video_id)


def get_timeline_json(db: Session, video_id: int) -> Dict[str, Any]:
    """Get timeline JSON"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise ValueError("Video not found")
    if not video.timeline_json:
        raise ValueError("No timeline found")
    
    return video.timeline_json


def create_multi_video_project(db: Session, video_ids: List[int], content_type: str,
                              target_platform: str = "youtube_shorts", brief: Optional[str] = None,
                              advanced_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create multi-video project (simplified)"""
    
    try:
        content_enum = ContentType(content_type)
    except ValueError:
        content_enum = ContentType.INTERVIEW
    
    editor = VideoEditor(db)
    results = []
    
    for video_id in video_ids:
        try:
            result = editor.create_edit(
                video_id=video_id,
                content_type=content_enum,
                target_platform=target_platform,
                brief=brief,
                options=advanced_options
            )
            results.append({
                "video_id": video_id,
                "status": "success",
                "timeline_items": len(result.timeline.items),
                "duration": result.timeline.total_duration
            })
        except Exception as e:
            results.append({
                "video_id": video_id,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "project_id": f"multi_{int(time.time())}",
        "analysis_results": results,
        "edit_result": {"total_videos": len(results)},
        "insights": {"processed_videos": len([r for r in results if r["status"] == "success"])},
        "performance_metrics": {"processing_time": time.time()}
    }