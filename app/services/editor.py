from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, Any, Optional, List

from sqlalchemy.orm import Session
from ollama import Client

from app import models
from app.schemas import EditorAgentResponse, TimelinePlan, Storyboard

OLLAMA_MODEL = "llama3.1:8b-instruct"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 120
OLLAMA_RETRIES = 2
TEMPERATURE = 0.35

logger = logging.getLogger(__name__)
client = Client(host=OLLAMA_HOST)


def _gather_context(db: Session, video_id: int) -> Dict[str, Any]:
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise ValueError("Video not found")

    shots = (
        db.query(models.Shot)
        .filter(models.Shot.video_id == video_id)
        .order_by(models.Shot.shot_index.asc())
        .all()
    )
    if not shots:
        raise ValueError("No shots found for video. Run shot detection first.")

    context = {
        "video": {"id": video.id, "file_path": video.file_path},
        "shots": [
            {
                "clip_id": s.id,
                "shot_index": s.shot_index,
                "start_time": float(s.start_time or 0.0),
                "end_time": float(s.end_time or 0.0),
                "duration": max(0.0, float((s.end_time or 0) - (s.start_time or 0))),
                "transcript": (s.transcript or "")[:6000],
                "analysis": (s.analysis or "")[:4000],
            }
            for s in shots
        ],
    }
    return context


def _build_prompt(context: Dict[str, Any], user_brief: Optional[str]) -> str:
    def fmt_shot(sh: Dict[str, Any]) -> str:
        return (
            f"- Clip {sh['clip_id']} "
            f"[{sh['start_time']:.2f}s–{sh['end_time']:.2f}s | dur {sh['duration']:.2f}s]\n"
            f"  transcript: {sh['transcript']}\n"
            f"  analysis: {sh['analysis']}\n"
        )

    shots_txt = "\n".join(fmt_shot(sh) for sh in context["shots"])
    brief_txt = (user_brief or "").strip() or "Platform: YouTube Shorts/TikTok. Goal: maximize hook & retention."

    schema_hint = {
        "storyboard": {
            "theme": "string",
            "beats": [
                {"order": 1, "title": "string", "summary": "string", "supporting_clips": [0, 1]}
            ],
        },
        "timeline": {
            "items": [
                {
                    "clip_id": 0,
                    "order": 1,
                    "start_time": 0.0,    
                    "end_time": 0.0,       
                    "highlight_reason": "string"
                }
            ]
        },
    }

    return f"""
You are a senior viral short-form video editor.

TASK:
- Re-order clips based on transcripts/analysis to maximize retention.
- Hard cut boring parts; keep only the most interesting/viral segments.
- Ensure a strong hook in the first 2–4 seconds.

CONSTRAINTS:
- Output VALID JSON ONLY (no markdown, no commentary).
- Conform to this key structure: {json.dumps(schema_hint)}
- If unsure, make a best-effort guess; do not leave fields null.

USER_BRIEF:
{brief_txt}

CLIPS:
{shots_txt}

Return a single JSON object with keys "storyboard" and "timeline".
"""


def _extract_first_json_blob(text: str) -> Optional[str]:
    """Extract first JSON object/array from raw text."""
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    return m.group(1) if m else None


def _call_ollama_json(prompt: str) -> Dict[str, Any]:
    last_err = None
    for attempt in range(OLLAMA_RETRIES + 1):
        try:
            res = client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "Respond with strict JSON that validates."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": TEMPERATURE, "num_ctx": 8192},
                format="json",
            )
            raw = (res or {}).get("message", {}).get("content", "")
            try:
                return json.loads(raw)
            except Exception:
                blob = _extract_first_json_blob(raw)
                if blob:
                    return json.loads(blob)
                raise ValueError("Model did not return valid JSON.")
        except Exception as e:
            last_err = e
            logger.warning("Ollama JSON call failed (attempt %s/%s): %s",
                           attempt + 1, OLLAMA_RETRIES + 1, e)
            if attempt < OLLAMA_RETRIES:
                time.sleep(1.0 + attempt * 0.5)
    raise RuntimeError(f"Ollama call ultimately failed: {last_err}")


def _postprocess_timeline_items(db: Session, video_id: int, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Harden timeline items from the LLM:
    - Filter to valid Shot IDs for this video
    - Coerce types and bounds (times)
    - Ensure stable, unique ordering
    """
    valid_shots = {
        s.id: (float(s.start_time or 0.0), float(s.end_time or 0.0))
        for s in db.query(models.Shot).filter(models.Shot.video_id == video_id).all()
    }

    cleaned: List[Dict[str, Any]] = []
    seen_orders = set()
    next_order = 1

    for it in items or []:
        cid = it.get("clip_id")
        if cid not in valid_shots:
            continue

        default_start, default_end = valid_shots[cid]
        st = float(it.get("start_time", default_start))
        en = float(it.get("end_time", default_end))
        if en <= st:
            en = max(st + 0.25, default_end)

        order = int(it.get("order") or 0)
        if order <= 0 or order in seen_orders:
            order = next_order
        seen_orders.add(order)
        next_order = order + 1

        cleaned.append({
            "clip_id": cid,
            "order": order,
            "start_time": st,
            "end_time": en,
            "highlight_reason": (it.get("highlight_reason") or "").strip()[:256],
        })

    cleaned.sort(key=lambda x: x["order"])
    return cleaned


def run_editor_agent(db: Session, video_id: int, user_brief: Optional[str] = None) -> EditorAgentResponse:
    ctx = _gather_context(db, video_id)
    prompt = _build_prompt(ctx, user_brief)
    raw = _call_ollama_json(prompt)

    if not isinstance(raw, dict) or "timeline" not in raw:
        raise ValueError("Model response missing 'timeline' key.")

    timeline_part = raw.get("timeline")
    storyboard_part = raw.get("storyboard")

    try:
        timeline_model = TimelinePlan.model_validate(timeline_part)
    except Exception as e:
        logger.error("Timeline validation failed: %s", e)
        raise

    items_clean = _postprocess_timeline_items(db, video_id, timeline_model.items)
    timeline_model.items = items_clean  

    storyboard_model = None
    if storyboard_part:
        try:
            storyboard_model = Storyboard.model_validate(storyboard_part)
        except Exception as e:
            logger.warning("Storyboard validation failed, ignoring storyboard: %s", e)

    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise ValueError("Video not found (race condition).")

    video.timeline_json = timeline_model.model_dump()
    if storyboard_model:
        video.storyboard_json = storyboard_model.model_dump()

    db.add(video)
    db.commit()

    return EditorAgentResponse(storyboard=storyboard_model, timeline=timeline_model)


def get_timeline_json(db: Session, video_id: int) -> Dict[str, Any]:
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise ValueError("Video not found")
    if not video.timeline_json:
        raise ValueError("No timeline found for video")
    return video.timeline_json