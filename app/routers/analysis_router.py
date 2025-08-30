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

@router.delete("/{video_id}/clear")
def clear_video_analyses(
    video_id: int,
    clear_all: bool = Query(False, description="Clear all analyses, not just errors"),
    db: Session = Depends(get_db),
):
    """Clear analysis results for a specific video"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Clear shot analyses
        shots = video.shots
        cleared_count = 0
        
        if clear_all:
            # Clear all analyses
            for shot in shots:
                if shot.analysis:
                    shot.analysis = None
                    cleared_count += 1
        else:
            # Only clear error analyses
            error_patterns = [
                "Failed to load model",
                "CUDA out of memory",
                "Analysis error",
                "Processing error",
                "OOM error"
            ]
            
            for shot in shots:
                if shot.analysis and any(pattern in shot.analysis for pattern in error_patterns):
                    shot.analysis = None
                    cleared_count += 1
        
        # Clear video-level analysis if exists
        video_cleared = False
        if hasattr(video, 'analysis') and video.analysis:
            if clear_all or any(pattern in video.analysis for pattern in error_patterns):
                video.analysis = None
                video_cleared = True
        
        db.commit()
        
        return {
            "status": "success",
            "video_id": video_id,
            "cleared_shot_analyses": cleared_count,
            "cleared_video_analysis": video_cleared,
            "message": f"Cleared {cleared_count} shot analyses" + (" and video analysis" if video_cleared else "")
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to clear analyses: {str(e)}")

@router.get("/{video_id}/stats")
def get_analysis_stats(
    video_id: int,
    db: Session = Depends(get_db),
):
    """Get statistics about analysis results for a video"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        shots = video.shots
        total_shots = len(shots)
        with_analysis = sum(1 for shot in shots if shot.analysis and shot.analysis.strip())
        empty_analysis = sum(1 for shot in shots if not shot.analysis or not shot.analysis.strip())
        
        error_patterns = [
            "Failed to load model",
            "CUDA out of memory", 
            "Analysis error",
            "Processing error",
            "OOM error"
        ]
        
        error_analyses = sum(1 for shot in shots 
                           if shot.analysis and any(pattern in shot.analysis for pattern in error_patterns))
        
        successful_analyses = with_analysis - error_analyses
        
        return {
            "video_id": video_id,
            "total_shots": total_shots,
            "with_analysis": with_analysis,
            "empty_analysis": empty_analysis,
            "error_analyses": error_analyses,
            "successful_analyses": successful_analyses,
            "has_video_analysis": bool(hasattr(video, 'analysis') and video.analysis)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")