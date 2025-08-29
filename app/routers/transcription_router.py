from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app import models
from app.services.transcription import transcribe_audio

router = APIRouter(prefix="/transcription", tags=["transcription"])

@router.post("/{video_id}")
def transcribe_video_endpoint(
    video_id: int,
    per_shot: bool = Query(True, description="Transcribe per shot (True) or full video (False)"),
    db: Session = Depends(get_db),
):
    """Transcribe audio for a video - either per shot or full video"""
    print(f"Starting transcription for video ID: {video_id}")
    
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    print(f"Found video: {video.id}, file_path: {video.file_path}")

    try:
        if per_shot:
            shots = video.shots
            shot_count = len(shots) if shots else 0
            print(f"Found {shot_count} shots for video")
            
            if shot_count == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="No shots found for this video. Run shot detection first."
                )
            
            transcribed_count = 0
            error_count = 0
            
            for i, shot in enumerate(shots):
                print(f"Transcribing shot {i+1}/{shot_count} (Shot ID: {shot.id})")
                
                if shot.transcript and shot.transcript.strip() and not shot.transcript.startswith("Transcription error"):
                    print(f"Shot {i+1} already has transcription, skipping...")
                    continue
                
                try:
                    transcript_result = transcribe_audio(
                        audio_path=video.file_path,
                        start_time=shot.start_time,
                        end_time=shot.end_time
                    )
                    
                    print(f"Got transcription result: '{transcript_result[:100]}...' (length: {len(transcript_result)})")
                    
                    shot.transcript = transcript_result
                    db.add(shot)
                    transcribed_count += 1
                    
                    if transcribed_count % 10 == 0:
                        print(f"Committing after {transcribed_count} shots...")
                        db.commit()
                        
                except Exception as e:
                    print(f"Error transcribing shot {shot.shot_index}: {e}")
                    shot.transcript = f"Transcription error: {str(e)}"
                    db.add(shot)
                    error_count += 1
            
            db.commit()
            print(f"Transcription complete. Processed: {transcribed_count}, Errors: {error_count}")
            result_message = f"Successfully transcribed {transcribed_count} shots, {error_count} errors"
            
        else:
            print("Transcribing full video...")
            
            if video.transcript and video.transcript.strip() and not video.transcript.startswith("Transcription error"):
                result_message = "Video already has transcription"
            else:
                try:
                    transcript_result = transcribe_audio(video.file_path)
                    video.transcript = transcript_result
                    db.add(video)
                    db.commit()
                    result_message = f"Successfully transcribed full video, length: {len(transcript_result)} characters"
                except Exception as e:
                    video.transcript = f"Transcription error: {str(e)}"
                    db.add(video)
                    db.commit()
                    result_message = f"Transcription failed: {str(e)}"
        
        return {
            "status": "success", 
            "video_id": video.id, 
            "per_shot": per_shot,
            "message": result_message
        }
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Transcription failed: {str(e)}"
        )

@router.get("/{video_id}/results")
def get_transcription_results(
    video_id: int,
    db: Session = Depends(get_db),
):
    """Get transcription results for a video"""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    shot_transcriptions = []
    if hasattr(video, 'shots') and video.shots:
        for shot in video.shots:
            shot_transcriptions.append({
                "shot_id": shot.id,
                "shot_index": shot.shot_index,
                "start_time": shot.start_time,
                "end_time": shot.end_time,
                "transcript": shot.transcript
            })
    
    return {
        "video_id": video.id,
        "video_transcript": video.transcript,
        "shot_count": len(shot_transcriptions),
        "shot_transcriptions": shot_transcriptions
    }