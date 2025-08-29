from sqlalchemy.orm import Session
from app.models import Shot
import cv2
from transnetv2_pytorch import TransNetV2
import tempfile
import subprocess
import os

_transnet_model = None

def get_transnet_model():
    global _transnet_model
    if _transnet_model is None:
        print("Loading TransNetV2 model...")
        _transnet_model = TransNetV2()
        print("TransNetV2 model loaded successfully")
    return _transnet_model


def detect_shots_transnet(video_path: str, threshold: float = 0.5):
    """
    Shot detection using TransNetV2.
    threshold: confidence threshold for shot boundaries (0.0 to 1.0)
    """
    model = get_transnet_model()
    
    temp_video = None
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        print(f"Video info: {frame_count} frames at {fps} FPS")
        
        if fps > 0 and frame_count > 0:
            video_for_detection = video_path
        else:
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            cmd = f"ffmpeg -i {video_path} -r 25 -c:v libx264 -preset fast {temp_video}"
            subprocess.run(cmd, shell=True, check=True)
            video_for_detection = temp_video
            
            cap = cv2.VideoCapture(video_for_detection)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        
        video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_for_detection)
        
        shot_boundaries = [i for i, pred in enumerate(single_frame_predictions) if pred > threshold]
        
        shots = []
        start_frame = 0
        
        for i, boundary_frame in enumerate(shot_boundaries):
            shots.append({
                "shot_index": i,
                "start_frame": start_frame,
                "end_frame": boundary_frame,
                "start_time": start_frame / fps,
                "end_time": boundary_frame / fps,
                "confidence": float(single_frame_predictions[boundary_frame])
            })
            start_frame = boundary_frame + 1
        
        if start_frame < frame_count:
            shots.append({
                "shot_index": len(shots),
                "start_frame": start_frame,
                "end_frame": frame_count - 1,
                "start_time": start_frame / fps,
                "end_time": (frame_count - 1) / fps,
                "confidence": 1.0
            })
        
        print(f"TransNetV2 detected {len(shots)} shots")
        return shots
        
    except Exception as e:
        print(f"TransNetV2 detection failed: {e}")
        return detect_shots_pyscene(video_path, 30.0)
        
    finally:
        if temp_video and os.path.exists(temp_video):
            os.unlink(temp_video)


def detect_shots_pyscene(video_path: str, threshold: float = 30.0):
    """
    Fallback shot detection using PySceneDetect.
    """
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    
    print("Using PySceneDetect as fallback...")
    
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()

    shots = []
    for i, (start, end) in enumerate(scene_list):
        shots.append({
            "shot_index": i,
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "start_time": start.get_seconds(), 
            "end_time": end.get_seconds(),
            "confidence": 1.0 
        })
    return shots


def detect_shots(video_path: str, method: str = "transnet", threshold: float = 0.5):
    """
    Shot detection with two methods.
    method: "transnet" or "pyscene"
    threshold: 0.5 for TransNetV2, 30.0 for PySceneDetect
    """
    if method == "transnet":
        return detect_shots_transnet(video_path, threshold)
    else:
        return detect_shots_pyscene(video_path, threshold)


def process_shots_for_video(db: Session, video_id: int, video_path: str, method: str = "transnet", threshold: float = 0.5):
    """
    Shot boundary detection and storage.
    """
    print(f"Processing shots for video {video_id} using {method}")
    
    shots = detect_shots(video_path, method=method, threshold=threshold)

    db_shots = []
    for shot in shots:
        db_shot = Shot(
            video_id=video_id,
            shot_index=shot["shot_index"],
            start_frame=shot["start_frame"],
            end_frame=shot["end_frame"],
            start_time=shot["start_time"],
            end_time=shot["end_time"],
            transcript=None,  
            analysis=None     
        )
        db.add(db_shot)
        db_shots.append(db_shot)

    db.commit()
    print(f"Saved {len(db_shots)} shots to database")
    return db_shots
