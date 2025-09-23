# Removed unused imports
import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from ollama import Client
from sqlalchemy.orm import Session

from app import models

# Unused imports removed for cleaner code

logger = logging.getLogger(__name__)

# LLM Configuration
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://127.0.0.1:11434"
client = Client(host=OLLAMA_HOST)


class ContentType(Enum):
    INTERVIEW = "interview"
    PODCAST = "podcast"
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    DOCUMENTARY = "documentary"
    PRODUCT_DEMO = "product_demo"
    MARKETING = "marketing"
    TESTIMONIAL = "testimonial"
    TUTORIAL = "tutorial"
    PRESENTATION = "presentation"


class SceneType(Enum):
    HOOK = "hook"  # Attention-grabbing opening
    INTRO = "intro"  # Introduction/setup
    MAIN_CONTENT = "main_content"  # Core content
    TRANSITION = "transition"  # Connecting segments
    CLIMAX = "climax"  # Peak moment/revelation
    CONCLUSION = "conclusion"  # Wrap-up/summary
    CALL_TO_ACTION = "call_to_action"  # Action request


@dataclass
class SceneSegment:
    """Intelligent scene segment with analysis"""

    start_time: float
    end_time: float
    scene_type: SceneType
    confidence: float
    content_summary: str
    engagement_score: float
    viral_potential: float
    key_moments: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    speaker_changes: int = 0
    visual_complexity: float = 5.0
    audio_quality: float = 5.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def composite_score(self) -> float:
        """Overall quality score for this scene"""
        return (
            self.engagement_score * 0.4
            + self.viral_potential * 0.3
            + self.confidence * 0.2
            + self.visual_complexity * 0.05
            + self.audio_quality * 0.05
        )


@dataclass
class ContentProject:
    """Intelligent content creation project"""

    video_id: int
    content_type: ContentType
    target_platforms: List[str]
    objective: str  # What the content should achieve
    target_audience: str
    tone: str  # professional, casual, energetic, etc.
    max_duration: float = 60.0
    min_engagement_score: float = 6.0
    scene_segments: List[SceneSegment] = field(default_factory=list)


class ContentAnalyzer:
    """Advanced content analysis with AI understanding"""

    def __init__(self, content_type: ContentType):
        self.content_type = content_type
        self.engagement_patterns = self._get_engagement_patterns()
        self.viral_indicators = self._get_viral_indicators()

    def _get_engagement_patterns(self) -> Dict[str, List[str]]:
        """Content-specific engagement patterns"""
        patterns = {
            ContentType.INTERVIEW: {
                "hooks": [
                    "question",
                    "surprising",
                    "reveal",
                    "secret",
                    "truth",
                    "never told",
                ],
                "engagement": ["story", "personal", "experience", "lesson", "advice"],
                "transitions": ["but", "however", "then", "suddenly", "what happened"],
            },
            ContentType.EDUCATIONAL: {
                "hooks": ["learn", "discover", "secret", "mistake", "never knew"],
                "engagement": ["example", "demonstration", "proof", "result", "impact"],
                "transitions": ["next", "now", "let's", "here's how", "step"],
            },
            ContentType.PRODUCT_DEMO: {
                "hooks": ["game-changer", "revolutionary", "problem", "solution"],
                "engagement": ["before", "after", "results", "transformation"],
                "transitions": ["watch this", "see how", "notice", "compare"],
            },
            ContentType.MARKETING: {
                "hooks": ["exclusive", "limited", "secret", "exposed", "revealed"],
                "engagement": ["results", "proof", "testimonial", "success"],
                "transitions": ["but wait", "there's more", "imagine", "what if"],
            },
        }
        return patterns.get(self.content_type, patterns[ContentType.INTERVIEW])

    def _get_viral_indicators(self) -> List[Tuple[str, float]]:
        """Viral potential indicators with weights"""
        return [
            ("shocking", 3.0),
            ("unbelievable", 2.8),
            ("secret", 2.5),
            ("truth", 2.3),
            ("exposed", 2.8),
            ("revealed", 2.2),
            ("crazy", 2.5),
            ("amazing", 2.0),
            ("incredible", 2.3),
            ("never", 2.0),
            ("always", 1.8),
            ("everyone", 1.8),
            ("nobody", 2.2),
            ("mistake", 2.0),
            ("wrong", 1.8),
            ("right", 1.5),
            ("perfect", 1.7),
            ("best", 1.8),
            ("worst", 2.2),
            ("failure", 2.0),
            ("success", 1.8),
        ]

    def analyze_transcript_segment(
        self, transcript: str, start_time: float, end_time: float
    ) -> SceneSegment:
        """Analyze transcript segment with AI understanding"""

        # Basic metrics
        duration = end_time - start_time

        # Determine scene type
        scene_type = self._classify_scene_type(transcript, start_time, duration)

        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(transcript)

        # Calculate viral potential
        viral_potential = self._calculate_viral_potential(transcript)

        # Extract key elements
        key_moments = self._extract_key_moments(transcript)
        emotions = self._detect_emotions(transcript)
        topics = self._extract_topics(transcript)

        # Confidence based on content quality
        confidence = self._calculate_confidence(transcript, duration)

        return SceneSegment(
            start_time=start_time,
            end_time=end_time,
            scene_type=scene_type,
            confidence=confidence,
            content_summary=self._generate_summary(transcript),
            engagement_score=engagement_score,
            viral_potential=viral_potential,
            key_moments=key_moments,
            emotions=emotions,
            topics=topics,
            speaker_changes=self._count_speaker_changes(transcript),
            visual_complexity=5.0,  # Default - could be enhanced with video analysis
            audio_quality=self._estimate_audio_quality(transcript, duration),
        )

    def _classify_scene_type(
        self, transcript: str, start_time: float, duration: float
    ) -> SceneType:
        """Classify scene type using AI analysis"""
        text_lower = transcript.lower()

        # Hook indicators (first 20% or strong hook words)
        hook_indicators = [
            "wait",
            "stop",
            "imagine",
            "what if",
            "secret",
            "truth",
            "never",
            "always",
        ]
        if start_time < 10 or any(
            indicator in text_lower for indicator in hook_indicators
        ):
            hook_score = sum(
                2.0 for indicator in hook_indicators if indicator in text_lower
            )
            if hook_score >= 2.0:
                return SceneType.HOOK

        # Call to action indicators
        cta_indicators = [
            "subscribe",
            "like",
            "follow",
            "click",
            "visit",
            "buy",
            "get",
            "download",
        ]
        if any(cta in text_lower for cta in cta_indicators):
            return SceneType.CALL_TO_ACTION

        # Conclusion indicators
        conclusion_indicators = [
            "conclusion",
            "summary",
            "final",
            "end",
            "wrap up",
            "to conclude",
        ]
        if any(conclusion in text_lower for conclusion in conclusion_indicators):
            return SceneType.CONCLUSION

        # Transition indicators
        transition_indicators = [
            "but",
            "however",
            "meanwhile",
            "next",
            "then",
            "so",
            "therefore",
        ]
        if (
            any(transition in text_lower for transition in transition_indicators)
            and duration < 5
        ):
            return SceneType.TRANSITION

        # Climax indicators (high engagement + revelation words)
        climax_indicators = [
            "reveal",
            "discovered",
            "breakthrough",
            "moment",
            "realized",
            "truth",
        ]
        if any(climax in text_lower for climax in climax_indicators):
            return SceneType.CLIMAX

        # Default to main content
        return SceneType.MAIN_CONTENT

    def _calculate_engagement_score(self, transcript: str) -> float:
        """Calculate engagement potential score"""
        text_lower = transcript.lower()

        # Pattern matching
        patterns = self.engagement_patterns.get("engagement", [])
        pattern_score = sum(1.5 for pattern in patterns if pattern in text_lower)

        # Question bonus
        question_score = text_lower.count("?") * 1.5

        # Emotional words
        emotional_words = [
            "love",
            "hate",
            "amazing",
            "terrible",
            "incredible",
            "shocking",
        ]
        emotion_score = sum(1.0 for word in emotional_words if word in text_lower)

        # Personal pronouns (engagement through relatability)
        personal_pronouns = ["you", "your", "we", "us", "our"]
        personal_score = sum(
            0.5 for pronoun in personal_pronouns if pronoun in text_lower
        )

        total_score = pattern_score + question_score + emotion_score + personal_score
        return min(10.0, total_score)

    def _calculate_viral_potential(self, transcript: str) -> float:
        """Calculate viral potential score"""
        text_lower = transcript.lower()

        viral_score = 0.0
        for indicator, weight in self.viral_indicators:
            if indicator in text_lower:
                viral_score += weight

        # Bonus for numbers (stats are viral)
        import re

        numbers = re.findall(r"\b\d+\b", transcript)
        if numbers:
            viral_score += len(numbers) * 0.5

        # Bonus for superlatives
        superlatives = [
            "best",
            "worst",
            "most",
            "least",
            "biggest",
            "smallest",
            "fastest",
            "slowest",
        ]
        viral_score += sum(1.0 for sup in superlatives if sup in text_lower)

        return min(10.0, viral_score)

    def _extract_key_moments(self, transcript: str) -> List[str]:
        """Extract key moments from transcript"""
        sentences = re.split(r"[.!?]+", transcript)
        key_moments = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Score sentences
            score = 0
            sentence_lower = sentence.lower()

            # High-impact words
            impact_words = [
                "discovered",
                "realized",
                "revealed",
                "secret",
                "truth",
                "never",
                "always",
            ]
            score += sum(2 for word in impact_words if word in sentence_lower)

            # Numbers and statistics
            if re.search(r"\b\d+\b", sentence):
                score += 1

            if score >= 2:
                key_moments.append(sentence)

        return key_moments[:5]  # Top 5 key moments

    def _detect_emotions(self, transcript: str) -> List[str]:
        """Detect emotions in transcript"""
        text_lower = transcript.lower()

        emotion_map = {
            "excitement": ["amazing", "incredible", "awesome", "fantastic", "wow"],
            "surprise": ["shocking", "unbelievable", "unexpected", "suddenly"],
            "concern": ["worried", "concerned", "problem", "issue", "trouble"],
            "joy": ["happy", "delighted", "thrilled", "excited", "love"],
            "anger": ["angry", "furious", "frustrated", "hate", "terrible"],
            "fear": ["scared", "afraid", "terrified", "nervous", "anxious"],
            "sadness": ["sad", "disappointed", "devastating", "heartbreaking"],
        }

        detected_emotions = []
        for emotion, indicators in emotion_map.items():
            if any(indicator in text_lower for indicator in indicators):
                detected_emotions.append(emotion)

        return detected_emotions

    def _extract_topics(self, transcript: str) -> List[str]:
        """Extract main topics from transcript"""
        # Simple keyword extraction - could be enhanced with NLP
        text_lower = transcript.lower()

        # Common topic categories
        business_keywords = [
            "business",
            "company",
            "revenue",
            "profit",
            "growth",
            "strategy",
        ]
        tech_keywords = [
            "technology",
            "software",
            "ai",
            "digital",
            "platform",
            "system",
        ]
        personal_keywords = [
            "life",
            "experience",
            "journey",
            "story",
            "lesson",
            "advice",
        ]

        topics = []
        if any(keyword in text_lower for keyword in business_keywords):
            topics.append("business")
        if any(keyword in text_lower for keyword in tech_keywords):
            topics.append("technology")
        if any(keyword in text_lower for keyword in personal_keywords):
            topics.append("personal")

        return topics

    def _calculate_confidence(self, transcript: str, duration: float) -> float:
        """Calculate confidence in classification"""
        base_confidence = 5.0

        # More content = higher confidence
        word_count = len(transcript.split())
        if word_count > 50:
            base_confidence += 2.0
        elif word_count > 20:
            base_confidence += 1.0

        # Appropriate duration
        if 3.0 <= duration <= 15.0:
            base_confidence += 1.0

        # Clear speech patterns
        if not re.search(r"\[unclear\]|\[inaudible\]", transcript.lower()):
            base_confidence += 1.0

        return min(10.0, base_confidence)

    def _generate_summary(self, transcript: str) -> str:
        """Generate content summary"""
        # Simple extractive summary - first sentence + key points
        sentences = re.split(r"[.!?]+", transcript)
        if not sentences:
            return "Content segment"

        first_sentence = sentences[0].strip()
        return (
            first_sentence[:100] + "..."
            if len(first_sentence) > 100
            else first_sentence
        )

    def _count_speaker_changes(self, transcript: str) -> int:
        """Count potential speaker changes"""
        # Simple heuristic - look for speaker patterns
        speaker_indicators = ["interviewer:", "host:", "guest:", "speaker:"]
        return sum(
            1
            for indicator in speaker_indicators
            if indicator.lower() in transcript.lower()
        )

    def _estimate_audio_quality(self, transcript: str, duration: float) -> float:
        """Estimate audio quality from transcript"""
        if not transcript.strip():
            return 2.0

        word_count = len(transcript.split())
        if duration == 0:
            return 2.0

        words_per_second = word_count / duration

        # Optimal speech rate
        if 1.5 <= words_per_second <= 3.5:
            base_score = 8.0
        elif words_per_second < 1.0:
            base_score = 4.0
        elif words_per_second > 5.0:
            base_score = 4.0
        else:
            base_score = 6.0

        # Penalty for unclear audio indicators
        unclear_indicators = ["[unclear]", "[inaudible]", "um um", "uh uh"]
        penalty = sum(
            1.0 for indicator in unclear_indicators if indicator in transcript.lower()
        )

        return max(1.0, min(10.0, base_score - penalty))


class ContentCreator:
    """Main intelligent content creation system"""

    def __init__(self, db: Session, options: Dict[str, Any] = None):
        self.db = db
        self.llm_client = client
        self.options = options or {}

    def analyze_video_content(
        self, video_id: int, content_type: ContentType
    ) -> List[SceneSegment]:
        """Analyze video content and identify intelligent scenes"""

        # Get video and shots
        video = self.db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        shots = (
            self.db.query(models.Shot)
            .filter(models.Shot.video_id == video_id)
            .order_by(models.Shot.start_time.asc())
            .all()
        )

        if not shots:
            raise ValueError("No shots found for analysis")

        logger.info(f"Analyzing {len(shots)} shots for intelligent content creation")

        # Initialize analyzer
        analyzer = ContentAnalyzer(content_type)
        scene_segments = []

        # Analyze each shot
        for shot in shots:
            transcript = getattr(shot, "transcript", "") or ""
            if len(transcript.strip()) < 10:
                continue  # Skip shots with insufficient content

            scene_segment = analyzer.analyze_transcript_segment(
                transcript, shot.start_time, shot.end_time
            )
            scene_segments.append(scene_segment)

        # Post-process and optimize scenes
        optimized_scenes = self._optimize_scene_flow(scene_segments, self.options)

        logger.info(f"Created {len(optimized_scenes)} intelligent scene segments")

        # If no scenes were created, check why and provide better error messages
        if not optimized_scenes:
            logger.warning("No scenes could be created from shots - checking for transcripts")

            # Check if shots have transcripts
            shots_with_transcripts = [s for s in shots if getattr(s, "transcript", "") and len(getattr(s, "transcript", "").strip()) >= 10]

            if not shots_with_transcripts:
                raise ValueError(f"Cannot generate content: No transcripts found for {len(shots)} shots. Please run transcription first.")

            logger.warning("No scenes could be created from shots - creating fallback scene")
            # Create a simple fallback scene using the first shot if available
            if shots:
                first_shot = shots[0]
                fallback_scene = SceneSegment(
                    start_time=first_shot.start_time,
                    end_time=first_shot.end_time,
                    scene_type=SceneType.MAIN_CONTENT,
                    confidence=0.5,
                    content_summary="Available video content",
                    engagement_score=5.0,
                    viral_potential=3.0,
                    key_moments=["Content available for processing"]
                )
                optimized_scenes = [fallback_scene]

        return optimized_scenes

    def _optimize_scene_flow(self, scenes: List[SceneSegment], options: Dict[str, Any] = None) -> List[SceneSegment]:
        """Optimize scene flow for narrative structure"""

        if not scenes:
            return scenes

        options = options or {}
        min_engagement = options.get('min_engagement_score', 1.0)
        min_composite = options.get('min_composite_score', 1.0)

        # Sort by start time
        scenes.sort(key=lambda s: s.start_time)

        # Ensure good narrative structure
        optimized = []

        # Find best hook (prioritize early, high-engagement scenes)
        hook_candidates = [
            s
            for s in scenes[:3]  # First 3 scenes
            if s.engagement_score >= min_engagement or s.scene_type == SceneType.HOOK
        ]

        if hook_candidates:
            try:
                best_hook = max(
                    hook_candidates, key=lambda s: s.engagement_score + s.viral_potential
                )
                optimized.append(best_hook)
                scenes.remove(best_hook)
            except ValueError:
                # Handle empty hook_candidates edge case
                pass

        # Add main content (highest quality scenes)
        main_content_scenes = [
            s
            for s in scenes
            if s.scene_type in [SceneType.MAIN_CONTENT, SceneType.CLIMAX]
            and s.composite_score >= min_composite
        ]

        # Sort by composite score and take best ones
        main_content_scenes.sort(key=lambda s: s.composite_score, reverse=True)
        optimized.extend(main_content_scenes[:8])  # Max 8 main scenes

        # Add conclusion if available
        conclusion_scenes = [s for s in scenes if s.scene_type == SceneType.CONCLUSION]
        if conclusion_scenes:
            try:
                best_conclusion = max(conclusion_scenes, key=lambda s: s.composite_score)
                optimized.append(best_conclusion)
            except ValueError:
                # Handle empty conclusion_scenes edge case
                pass

        # Sort final result by start time to maintain chronological order
        optimized.sort(key=lambda s: s.start_time)

        return optimized

    def create_content(self, project: ContentProject) -> Dict[str, Any]:
        """Create intelligent content for multiple platforms"""

        logger.info(f"Creating intelligent content for project: {project.objective}")

        # Analyze video content
        scene_segments = self.analyze_video_content(
            project.video_id, project.content_type
        )
        project.scene_segments = scene_segments

        # Generate content for each platform
        platform_content = {}

        for platform in project.target_platforms:
            try:
                content = self._create_platform_content(project, platform)
                platform_content[platform] = content
                logger.info(f"✅ Created content for {platform}")
            except Exception as e:
                logger.error(f"❌ Failed to create content for {platform}: {e}")
                platform_content[platform] = {"error": str(e)}

        # Generate master timeline
        master_timeline = self._create_master_timeline(project)

        return {
            "project_id": f"intelligent_{project.video_id}_{int(project.max_duration)}s",
            "content_analysis": {
                "total_scenes": len(scene_segments),
                "avg_engagement": statistics.mean(
                    [s.engagement_score for s in scene_segments]
                )
                if scene_segments
                else 0,
                "avg_viral_potential": statistics.mean(
                    [s.viral_potential for s in scene_segments]
                )
                if scene_segments
                else 0,
                "scene_types": dict(
                    Counter([s.scene_type.value for s in scene_segments])
                ),
            },
            "platform_content": platform_content,
            "master_timeline": master_timeline,
            "timeline": master_timeline,  # Add timeline field for schema compatibility
            "recommendations": self._generate_content_recommendations(project, self.options),
        }

    def _create_platform_content(
        self, project: ContentProject, platform: str
    ) -> Dict[str, Any]:
        """Create platform-specific content"""

        # Platform configurations
        platform_configs = {
            "youtube_shorts": {
                "max_duration": 60,
                "hook_duration": 3,
                "ideal_scenes": 6,
            },
            "tiktok": {"max_duration": 60, "hook_duration": 2, "ideal_scenes": 8},
            "instagram_reels": {
                "max_duration": 90,
                "hook_duration": 2,
                "ideal_scenes": 5,
            },
            "linkedin": {"max_duration": 300, "hook_duration": 5, "ideal_scenes": 8},
            "twitter": {"max_duration": 140, "hook_duration": 3, "ideal_scenes": 6},
        }

        config = platform_configs.get(platform, platform_configs["youtube_shorts"])

        # Select best scenes for platform
        selected_scenes = self._select_scenes_for_platform(
            project.scene_segments, config, self.options
        )

        # Generate AI-powered script
        script = self._generate_platform_script(project, platform, selected_scenes)

        # Create timeline
        timeline_data = self._create_platform_timeline(selected_scenes)

        return {
            "platform": platform,
            "duration": sum(s.duration for s in selected_scenes),
            "scene_count": len(selected_scenes),
            "engagement_score": statistics.mean(
                [s.engagement_score for s in selected_scenes]
            )
            if selected_scenes
            else 0,
            "viral_potential": max([s.viral_potential for s in selected_scenes], default=0)
            if selected_scenes
            else 0,
            "selected_scenes": [
                {
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "scene_type": s.scene_type.value,
                    "summary": s.content_summary,
                    "score": s.composite_score,
                }
                for s in selected_scenes
            ],
            "ai_script": script,
            "timeline": timeline_data,
        }

    def _select_scenes_for_platform(
        self, scenes: List[SceneSegment], config: Dict[str, Any], options: Dict[str, Any] = None
    ) -> List[SceneSegment]:
        """Select optimal scenes for platform"""

        max_duration = config["max_duration"]
        ideal_scenes = config["ideal_scenes"]
        options = options or {}
        min_composite = options.get('min_composite_score', 1.0)

        # Early return if no scenes available
        if not scenes:
            return []

        # Filter high-quality scenes
        quality_scenes = [s for s in scenes if s.composite_score >= min_composite]
        if not quality_scenes:
            quality_scenes = sorted(
                scenes, key=lambda s: s.composite_score, reverse=True
            )[:ideal_scenes]

        # Ensure we have a good hook
        hook_scenes = [
            s
            for s in quality_scenes
            if s.scene_type == SceneType.HOOK or s.start_time < 10
        ]
        if not hook_scenes and quality_scenes:
            # Use highest engagement scene as hook
            try:
                hook_scenes = [max(quality_scenes, key=lambda s: s.engagement_score)]
            except ValueError:
                # Handle empty quality_scenes edge case
                hook_scenes = []

        selected = []
        current_duration = 0.0

        # Add best hook first (if available)
        if hook_scenes:
            try:
                best_hook = max(
                    hook_scenes, key=lambda s: s.engagement_score + s.viral_potential
                )
                selected.append(best_hook)
                current_duration += best_hook.duration
                if best_hook in quality_scenes:
                    quality_scenes.remove(best_hook)
            except ValueError:
                # Handle empty hook_scenes edge case
                pass

        # Add remaining scenes by score, respecting duration
        quality_scenes.sort(key=lambda s: s.composite_score, reverse=True)

        for scene in quality_scenes:
            if (
                current_duration + scene.duration <= max_duration
                and len(selected) < ideal_scenes
            ):
                selected.append(scene)
                current_duration += scene.duration

        # Sort by start time to maintain chronological order
        selected.sort(key=lambda s: s.start_time)

        return selected

    def _generate_platform_script(
        self, project: ContentProject, platform: str, scenes: List[SceneSegment]
    ) -> str:
        """Generate AI-powered script for platform"""

        # Prepare scene summaries
        scene_summaries = []
        for i, scene in enumerate(scenes):
            summary = f"Scene {i+1} ({scene.duration:.1f}s): {scene.content_summary} [Engagement: {scene.engagement_score:.1f}, Viral: {scene.viral_potential:.1f}]"
            scene_summaries.append(summary)

        prompt = f"""
Create an engaging {platform} script for {project.content_type.value} content.

OBJECTIVE: {project.objective}
TARGET AUDIENCE: {project.target_audience}
TONE: {project.tone}
PLATFORM: {platform}

AVAILABLE SCENES:
{chr(10).join(scene_summaries)}

Create a script that:
1. Hooks viewers immediately
2. Maintains engagement throughout
3. Delivers on the objective
4. Fits {platform} best practices
5. Uses the best available scenes

Format as a timeline script with timestamps and scene descriptions.
"""

        try:
            response = self.llm_client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert {platform} content creator specializing in viral {project.content_type.value} content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.3},
            )

            return response.get("message", {}).get(
                "content", "Script generation failed"
            )

        except Exception as e:
            logger.error(f"Script generation failed: {e}")
            return f"Auto-generated script for {platform} using {len(scenes)} selected scenes"

    def _create_platform_timeline(
        self, scenes: List[SceneSegment]
    ) -> Dict[str, Any]:
        """Create timeline data for platform"""

        timeline_items = []
        for i, scene in enumerate(scenes):
            timeline_items.append(
                {
                    "order": i + 1,
                    "start_time": scene.start_time,
                    "end_time": scene.end_time,
                    "duration": scene.duration,
                    "scene_type": scene.scene_type.value,
                    "summary": scene.content_summary,
                    "engagement_score": scene.engagement_score,
                    "viral_potential": scene.viral_potential,
                }
            )

        return {
            "total_duration": sum(s.duration for s in scenes),
            "scene_count": len(scenes),
            "items": timeline_items,
        }

    def _create_master_timeline(self, project: ContentProject) -> Dict[str, Any]:
        """Create master timeline with all scenes"""

        # Create timeline items that match the TimelinePlan schema
        timeline_items = []
        for i, scene in enumerate(project.scene_segments):
            timeline_items.append({
                "clip_id": i + 1,  # Use sequential IDs since we don't have real clip IDs
                "order": i + 1,
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "highlight_reason": f"{scene.scene_type.value}: {scene.content_summary[:50]}..."
            })

        return {
            "project_objective": project.objective,
            "content_type": project.content_type.value,
            "total_scenes": len(project.scene_segments),
            "total_duration": sum(s.duration for s in project.scene_segments),
            "scenes": [
                {
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "scene_type": s.scene_type.value,
                    "engagement_score": s.engagement_score,
                    "viral_potential": s.viral_potential,
                    "composite_score": s.composite_score,
                    "summary": s.content_summary,
                    "key_moments": s.key_moments,
                    "emotions": s.emotions,
                    "topics": s.topics,
                }
                for s in project.scene_segments
            ],
            # Add items field for TimelinePlan schema compatibility
            "items": timeline_items
        }

    def _generate_content_recommendations(self, project: ContentProject, options: Dict[str, Any] = None) -> List[str]:
        """Generate intelligent content recommendations"""

        recommendations = []
        scenes = project.scene_segments
        options = options or {}

        if not scenes:
            return ["No scenes available for analysis"]

        # Engagement analysis
        avg_engagement = statistics.mean([s.engagement_score for s in scenes])
        if avg_engagement < 5.0:
            recommendations.append(
                "Low overall engagement - consider adding more compelling moments or questions"
            )

        # Hook analysis
        hook_scenes = [
            s for s in scenes if s.scene_type == SceneType.HOOK or s.start_time < 10
        ]
        min_engagement = options.get('min_engagement_score', 1.0)
        if not hook_scenes or (hook_scenes and max([s.engagement_score for s in hook_scenes], default=0) < min_engagement):
            recommendations.append(
                "Weak opening hook - add attention-grabbing elements in first 10 seconds"
            )

        # Viral potential
        max_viral = max([s.viral_potential for s in scenes], default=0)
        if max_viral < 5.0:
            recommendations.append(
                "Limited viral potential - add surprising facts, numbers, or bold statements"
            )

        # Scene variety
        scene_types = set([s.scene_type for s in scenes])
        if len(scene_types) < 3:
            recommendations.append(
                "Limited scene variety - add different types of content for better flow"
            )

        # Duration optimization
        very_short_scenes = [s for s in scenes if s.duration < 2.0]
        if len(very_short_scenes) > len(scenes) * 0.3:
            recommendations.append(
                "Many very short scenes - consider combining related segments"
            )

        return recommendations


# Main entry points
def create_intelligent_content(
    db: Session,
    video_id: int,
    content_type: str,
    target_platforms: List[str],
    objective: str,
    target_audience: str = "general audience",
    tone: str = "engaging",
    max_duration: float = 60.0,
    options: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Main entry point for intelligent content creation"""

    try:
        content_enum = ContentType(content_type)
    except ValueError:
        content_enum = ContentType.INTERVIEW

    project = ContentProject(
        video_id=video_id,
        content_type=content_enum,
        target_platforms=target_platforms,
        objective=objective,
        target_audience=target_audience,
        tone=tone,
        max_duration=max_duration,
    )

    creator = ContentCreator(db, options or {})
    result = creator.create_content(project)

    # Save timeline and storyboard to database
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if video:
        video.timeline_json = result.get("timeline", {})
        video.storyboard_json = result.get("storyboard")
        db.commit()

    return result


def analyze_content_potential(
    db: Session, video_id: int, content_type: str = "interview"
) -> Dict[str, Any]:
    """Analyze content potential for intelligent creation"""

    try:
        content_enum = ContentType(content_type)
    except ValueError:
        content_enum = ContentType.INTERVIEW

    creator = ContentCreator(db)
    scenes = creator.analyze_video_content(video_id, content_enum)

    if not scenes:
        return {"error": "No scenes found for analysis"}

    return {
        "total_clips": len(scenes),
        "total_duration": sum(s.duration for s in scenes),
        "average_score": statistics.mean([s.engagement_score for s in scenes]) if scenes else 0,
        "max_score": max([s.engagement_score for s in scenes], default=0),
        "high_quality_clips": len([s for s in scenes if s.composite_score >= 7.0]),
        "usable_clips": len([s for s in scenes if s.composite_score >= 5.0]),
        "content_density": len(scenes) / sum(s.duration for s in scenes) if scenes and sum(s.duration for s in scenes) > 0 else 0,
        "viral_potential": statistics.mean([s.viral_potential for s in scenes]) if scenes else 0,
        "recommendations": [
            "Add more engaging content" if len([s for s in scenes if s.engagement_score >= 7.0]) < len(scenes) * 0.3 else "Good engagement level",
            "Consider shorter clips" if statistics.mean([s.duration for s in scenes]) > 30 else "Good clip duration",
            "Focus on high-energy moments" if statistics.mean([s.viral_potential for s in scenes]) < 5.0 else "Strong viral potential"
        ]
    }
