from enum import Enum
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Video
from app.services.shot_detection import process_shots_for_video
from app.services.transcription import transcribe_audio
from app.services.video_understanding import run_video_analysis

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class ProcessingStep(str, Enum):
    shot_detection = "shot_detection"
    transcription = "transcription"
    content_analysis = "content_analysis"


@router.get("/videos")
def list_videos(db: Session = Depends(get_db)):
    """Get list of all uploaded videos with their processing status"""
    videos = db.query(Video).all()

    video_list = []
    for video in videos:
        # Check processing status
        has_shots = bool(video.shots and len(video.shots) > 0)
        has_transcript = bool(video.transcript and video.transcript.strip())
        has_analysis = bool(video.analysis and video.analysis.strip())

        # Count shots with transcripts
        shots_with_transcript = 0
        if video.shots:
            shots_with_transcript = len([s for s in video.shots if s.transcript and s.transcript.strip()])

        video_list.append({
            "id": video.id,
            "filename": video.original_filename,
            "file_path": video.file_path,
            "file_size": video.file_size,
            "created_at": video.created_at,
            "updated_at": video.updated_at,
            "pipeline_status": {
                "has_shots": has_shots,
                "shots_count": len(video.shots) if video.shots else 0,
                "shot_count": len(video.shots) if video.shots else 0,  # Keep for compatibility
                "transcription_complete": has_transcript,
                "has_transcript": has_transcript,  # Keep for compatibility
                "shots_with_transcript": shots_with_transcript,
                "analysis_complete": has_analysis,
                "has_analysis": has_analysis,  # Keep for compatibility
                "processing": False  # Add processing flag
            }
        })

    return {"videos": video_list, "total": len(video_list)}


@router.get("/videos/{video_id}/status")
def get_video_status(video_id: int, db: Session = Depends(get_db)):
    """Get detailed processing status for a specific video"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    shots_info = []
    if video.shots:
        for shot in video.shots:
            shots_info.append({
                "id": shot.id,
                "shot_index": shot.shot_index,
                "start_time": shot.start_time,
                "end_time": shot.end_time,
                "duration": shot.end_time - shot.start_time,
                "has_transcript": bool(shot.transcript and shot.transcript.strip()),
                "transcript_length": len(shot.transcript) if shot.transcript else 0,
                "has_analysis": bool(shot.analysis and shot.analysis.strip())
            })

    # Calculate status for UI compatibility
    total_shots = len(shots_info)
    shots_with_transcript = len([s for s in shots_info if s["has_transcript"]])
    shots_with_analysis = len([s for s in shots_info if s["has_analysis"]])
    has_video_transcript = bool(video.transcript and video.transcript.strip())
    has_video_analysis = bool(video.analysis and video.analysis.strip())

    return {
        "video_id": video.id,
        "filename": video.original_filename,
        "file_path": video.file_path,
        "file_size": video.file_size,
        "created_at": video.created_at,
        "updated_at": video.updated_at,
        "shots": shots_info,
        "video_transcript": video.transcript,
        "video_analysis": video.analysis,
        # UI expects these fields at root level
        "shots_count": total_shots,
        "transcription_complete": has_video_transcript or (total_shots > 0 and shots_with_transcript == total_shots),
        "analysis_complete": has_video_analysis,
        "processing": False,
        # Keep original summary for compatibility
        "summary": {
            "total_shots": total_shots,
            "shots_with_transcript": shots_with_transcript,
            "shots_with_analysis": shots_with_analysis,
            "has_video_transcript": has_video_transcript,
            "has_video_analysis": has_video_analysis
        }
    }


@router.post("/videos/{video_id}/process")
def process_video_step(
    video_id: int,
    step: ProcessingStep,
    db: Session = Depends(get_db),
    # Shot detection parameters
    shot_method: str = Query("transnet", description="transnet or pyscene"),
    shot_threshold: float = Query(0.5, description="Detection threshold"),
    # Transcription parameters
    per_shot: bool = Query(True, description="Transcribe per shot or full video"),
    # Content analysis parameters
    content_type: Optional[str] = Query(None, description="Content type for analysis"),
    force_reprocess: bool = Query(False, description="Force reprocess even if data exists")
):
    """Process a specific step for a video"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    result = {"video_id": video_id, "step": step, "status": "success"}

    try:
        if step == ProcessingStep.shot_detection:
            # Check if shots already exist
            if video.shots and len(video.shots) > 0 and not force_reprocess:
                result["message"] = f"Video already has {len(video.shots)} shots. Use force_reprocess=true to regenerate."
                result["shot_count"] = len(video.shots)
            else:
                # Clear existing shots if force reprocessing
                if force_reprocess and video.shots:
                    for shot in video.shots:
                        db.delete(shot)
                    db.commit()

                shots = process_shots_for_video(
                    db, video_id, video.file_path, shot_method, shot_threshold
                )
                result["message"] = f"Generated {len(shots)} shots using {shot_method}"
                result["shot_count"] = len(shots)
                result["method"] = shot_method
                result["threshold"] = shot_threshold

        elif step == ProcessingStep.transcription:
            if per_shot:
                # Check if video has shots
                if not video.shots or len(video.shots) == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="No shots found. Run shot detection first."
                    )

                transcribed_count = 0
                skipped_count = 0
                error_count = 0

                for shot in video.shots:
                    # Skip if already transcribed and not force reprocessing
                    if (shot.transcript and shot.transcript.strip()
                        and not shot.transcript.startswith("Transcription error")
                        and not force_reprocess):
                        skipped_count += 1
                        continue

                    try:
                        transcript = transcribe_audio(
                            video.file_path, shot.start_time, shot.end_time
                        )
                        shot.transcript = transcript
                        db.add(shot)
                        transcribed_count += 1
                    except Exception as e:
                        shot.transcript = f"Transcription error: {str(e)}"
                        db.add(shot)
                        error_count += 1

                db.commit()
                result["message"] = f"Transcribed {transcribed_count} shots, skipped {skipped_count}, errors {error_count}"
                result["transcribed"] = transcribed_count
                result["skipped"] = skipped_count
                result["errors"] = error_count

            else:
                # Full video transcription
                if (video.transcript and video.transcript.strip()
                    and not video.transcript.startswith("Transcription error")
                    and not force_reprocess):
                    result["message"] = "Video already has transcript. Use force_reprocess=true to regenerate."
                else:
                    try:
                        transcript = transcribe_audio(video.file_path)
                        video.transcript = transcript
                        db.add(video)
                        db.commit()
                        result["message"] = f"Full video transcribed ({len(transcript)} characters)"
                        result["transcript_length"] = len(transcript)
                    except Exception as e:
                        video.transcript = f"Transcription error: {str(e)}"
                        db.add(video)
                        db.commit()
                        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        elif step == ProcessingStep.content_analysis:
            # Check if video has shots
            if not video.shots or len(video.shots) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No shots found. Run shot detection first."
                )

            if (video.analysis and video.analysis.strip() and not force_reprocess):
                result["message"] = "Video already has analysis. Use force_reprocess=true to regenerate."
            else:
                try:
                    # Run shot-by-shot analysis which is what the function is designed for
                    run_video_analysis(db, video, per_shot=True)

                    # Create a summary analysis from all shot analyses
                    shot_analyses = []
                    for shot in video.shots:
                        if shot.analysis and shot.analysis.strip():
                            shot_analyses.append(shot.analysis)

                    if shot_analyses:
                        # Combine analyses into a comprehensive video analysis
                        video.analysis = f"Video contains {len(shot_analyses)} analyzed segments: " + " | ".join(shot_analyses[:5])
                        if len(shot_analyses) > 5:
                            video.analysis += f" and {len(shot_analyses) - 5} more segments."
                    else:
                        video.analysis = "Video analysis completed but no content was analyzed successfully"

                    db.add(video)
                    db.commit()
                    result["message"] = f"Content analysis completed for {len(shot_analyses)} shots"
                    result["content_type"] = content_type or "general"
                    result["analyzed_shots"] = len(shot_analyses)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Content analysis failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Processing failed: {str(e)}"
        raise HTTPException(status_code=500, detail=str(e))

    return result


@router.post("/videos/{video_id}/process-pipeline")
def process_full_pipeline(
    video_id: int,
    db: Session = Depends(get_db),
    steps: List[ProcessingStep] = Query(..., description="Steps to process in order"),
    # Parameters for each step
    shot_method: str = Query("transnet"),
    shot_threshold: float = Query(0.5),
    per_shot: bool = Query(True),
    content_type: Optional[str] = Query(None),
    force_reprocess: bool = Query(False)
):
    """Process multiple steps in sequence for a video"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    results = []

    for step in steps:
        try:
            # Call the individual step processing
            step_result = process_video_step(
                video_id, step, db, shot_method, shot_threshold,
                per_shot, content_type, force_reprocess
            )
            results.append(step_result)
        except Exception as e:
            error_result = {
                "video_id": video_id,
                "step": step,
                "status": "error",
                "message": str(e)
            }
            results.append(error_result)
            # Stop processing on error
            break

    return {
        "video_id": video_id,
        "pipeline_status": "completed" if all(r["status"] == "success" for r in results) else "partial",
        "steps_completed": len([r for r in results if r["status"] == "success"]),
        "total_steps": len(steps),
        "results": results
    }


@router.delete("/videos/{video_id}/data")
def clear_video_data(
    video_id: int,
    db: Session = Depends(get_db),
    clear_shots: bool = Query(False, description="Clear shot data"),
    clear_transcript: bool = Query(False, description="Clear transcript data"),
    clear_analysis: bool = Query(False, description="Clear analysis data")
):
    """Clear specific data for a video (useful for reprocessing)"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    cleared = []

    if clear_shots and video.shots:
        shot_count = len(video.shots)
        for shot in video.shots:
            db.delete(shot)
        cleared.append(f"shots ({shot_count})")

    if clear_transcript:
        if video.transcript:
            video.transcript = None
            cleared.append("video transcript")

        if video.shots:
            shot_transcript_count = len([s for s in video.shots if s.transcript])
            for shot in video.shots:
                shot.transcript = None
            if shot_transcript_count > 0:
                cleared.append(f"shot transcripts ({shot_transcript_count})")

    if clear_analysis:
        if video.analysis:
            video.analysis = None
            cleared.append("video analysis")

        if video.shots:
            shot_analysis_count = len([s for s in video.shots if s.analysis])
            for shot in video.shots:
                shot.analysis = None
            if shot_analysis_count > 0:
                cleared.append(f"shot analysis ({shot_analysis_count})")

    if cleared:
        db.add(video)
        db.commit()

    return {
        "video_id": video_id,
        "cleared": cleared,
        "message": f"Cleared: {', '.join(cleared)}" if cleared else "Nothing to clear"
    }