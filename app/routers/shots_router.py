from enum import Enum
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Video
from app.schemas import ShotResponse
from app.services.shot_detection import process_shots_for_video


class DetectionMethod(str, Enum):
    transnet = "transnet"
    pyscene = "pyscene"


router = APIRouter(prefix="/shots", tags=["shots"])


@router.post("/{video_id}/detect", response_model=List[ShotResponse])
def generate_shots(
    video_id: int,
    db: Session = Depends(get_db),
    method: DetectionMethod = Query(
        DetectionMethod.transnet, description="Detection method: transnet or pyscene"
    ),
    threshold: float = Query(
        0.5,
        description="Threshold: 0.0-1.0 for TransNetV2, 10.0-50.0 for PySceneDetect",
    ),
):
    """
    Shot boundary detection for a video using TransNetV2 or PySceneDetect.
    """
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if method == DetectionMethod.transnet and not (0.0 <= threshold <= 1.0):
        raise HTTPException(
            status_code=400, detail="TransNetV2 threshold must be between 0.0 and 1.0"
        )
    elif method == DetectionMethod.pyscene and not (10.0 <= threshold <= 100.0):
        raise HTTPException(
            status_code=400,
            detail="PySceneDetect threshold should be between 10.0 and 100.0",
        )

    try:
        db_shots = process_shots_for_video(
            db,
            video_id=video.id,
            video_path=video.file_path,
            method=method.value,
            threshold=threshold,
        )

        return [
            ShotResponse(
                id=shot.id,
                shot_index=shot.shot_index,
                start_frame=shot.start_frame,
                end_frame=shot.end_frame,
                start_time=shot.start_time,
                end_time=shot.end_time,
                transcript=shot.transcript or "",
            )
            for shot in db_shots
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shot detection failed: {str(e)}")


@router.get("/{video_id}")
def get_shots(video_id: int, db: Session = Depends(get_db)):
    """Get existing shots for a video"""
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    if not video.shots:
        raise HTTPException(
            status_code=404, detail="No shots found. Run shot detection first."
        )

    return [
        ShotResponse(
            id=shot.id,
            shot_index=shot.shot_index,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
            start_time=shot.start_time,
            end_time=shot.end_time,
            transcript=shot.transcript or "",
        )
        for shot in video.shots
    ]
