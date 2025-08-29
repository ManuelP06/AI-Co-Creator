
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.compiler import render_timeline_to_file, OUTPUT_DIR
from app import models
import os

router = APIRouter(prefix="/render", tags=["render"])

@router.post("/{video_id}")
def render_video_endpoint(
    video_id: int,
    output_name: str = Query(None, description="optional output filename, e.g. final.mp4"),
    quality: str = Query("high", description="high/medium/low"),
    use_gpu: bool = Query(False, description="Try hardware GPU encoding"),
    lut_file: str = Query(None, description="Optional path to .cube LUT file on server"),
    db: Session = Depends(get_db)
):
    # ensure video exists
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    try:
        out_name = output_name or f"video_{video_id}_render.mp4"
        output_path = render_timeline_to_file(db, video_id, output_filename=out_name, quality=quality, use_gpu=use_gpu, lut_file=lut_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering failed: {e}")

    # Return location for download (static file path)
    return {"status": "ok", "output_path": output_path, "filename": os.path.basename(output_path)}

@router.get("/download/{filename}")
def download_rendered_file(filename: str):
    full = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="File not found")
    # For production serve via nginx or a proper static file server.
    return {"download_path": full}
