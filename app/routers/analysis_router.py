from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app import models
from app.services.video_understanding import run_video_analysis

router = APIRouter(prefix="/analysis", tags=["analysis"])

@router.post("/{video_id}")
def analyze_video_endpoint(
    video_id: int,
    per_shot: bool = Query(True, description="Analysing per shot (True) or full video (False)"),
    db: Session = Depends(get_db),
):
    print(f"Starting analysis for video ID: {video_id}")
    
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    print(f"Found video: {video.id}, file_path: {video.file_path}")

    if per_shot:
        try:
            shots = video.shots 
            shot_count = len(shots) if shots else 0
            print(f"Found {shot_count} shots for video")
            
            if shot_count == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="No shots found for this video. Run shot detection first."
                )
                
        except Exception as e:
            print(f"Error checking shots: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Could not access shots for video: {str(e)}"
            )

    try:
        run_video_analysis(db, video, per_shot=per_shot)
        

        if per_shot:
            db.refresh(video)
            analyzed_shots = [s for s in video.shots if s.analysis and s.analysis.strip()]
            result_message = f"Successfully analyzed {len(analyzed_shots)} shots"
        else:
            db.refresh(video)
            result_message = f"Successfully analyzed full video, analysis length: {len(video.analysis) if video.analysis else 0}"
        
        return {
            "status": "success", 
            "video_id": video.id, 
            "per_shot": per_shot,
            "message": result_message
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )

@router.get("/{video_id}/results")
def get_analysis_results(
    video_id: int,
    db: Session = Depends(get_db),
):
    """Get analysis results for a video"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    shot_analyses = []
    if hasattr(video, 'shots') and video.shots:
        for shot in video.shots:
            shot_analyses.append({
                "shot_id": shot.id,
                "shot_index": shot.shot_index,
                "start_time": shot.start_time,
                "end_time": shot.end_time,
                "analysis": shot.analysis
            })
    
    return {
        "video_id": video.id,
        "video_analysis": video.analysis,
        "shot_count": len(shot_analyses),
        "shot_analyses": shot_analyses
    }