import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.models import Shot

# Conditional import with error handling
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError as e:
    print(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False

try:
    from transnetv2_pytorch import TransNetV2

    TRANSNET_AVAILABLE = True
except ImportError as e:
    print(f"TransNetV2 not available: {e}")
    TRANSNET_AVAILABLE = False

try:
    # Removed unused torch import
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

_transnet_model = None


@dataclass
class ShotInfo:
    shot_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    features: Optional[Dict] = None


def get_transnet_model():
    global _transnet_model
    if _transnet_model is None:
        print("Loading TransNetV2 model...")
        _transnet_model = TransNetV2()
        print("TransNetV2 model loaded successfully")
    return _transnet_model


def extract_visual_features(frame) -> Dict:
    """Extract visual features from a frame for better shot boundary detection."""
    if not CV2_AVAILABLE:
        return {}

    # Color histogram features
    hist_b = cv2.calcHist([frame], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([frame], [1], None, [32], [0, 256])
    hist_r = cv2.calcHist([frame], [2], None, [32], [0, 256])

    # Brightness and contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)

    # Edge density for complexity measure
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    return {
        "hist_b": hist_b.flatten(),
        "hist_g": hist_g.flatten(),
        "hist_r": hist_r.flatten(),
        "brightness": brightness,
        "contrast": contrast,
        "edge_density": edge_density,
    }


def compute_frame_difference(features1: Dict, features2: Dict) -> float:
    """Compute similarity between two frames based on visual features."""
    if not features1 or not features2:
        return 0.0

    # Histogram comparison using correlation
    hist_diff = 0.0
    for channel in ["hist_b", "hist_g", "hist_r"]:
        if channel in features1 and channel in features2:
            corr = cv2.compareHist(
                features1[channel], features2[channel], cv2.HISTCMP_CORREL
            )
            hist_diff += 1.0 - corr  # Convert correlation to distance

    hist_diff /= 3.0  # Average across channels

    # Add brightness and contrast differences
    brightness_diff = (
        abs(features1.get("brightness", 0) - features2.get("brightness", 0)) / 255.0
    )
    contrast_diff = (
        abs(features1.get("contrast", 0) - features2.get("contrast", 0)) / 255.0
    )
    edge_diff = abs(features1.get("edge_density", 0) - features2.get("edge_density", 0))

    # Weighted combination
    total_diff = (
        hist_diff * 0.6 + brightness_diff * 0.2 + contrast_diff * 0.1 + edge_diff * 0.1
    )

    return total_diff


def detect_shots_enhanced_cv(video_path: str, threshold: float = 0.3) -> List[ShotInfo]:
    """Enhanced shot detection using OpenCV with multiple visual features."""
    if not CV2_AVAILABLE:
        return detect_shots_pyscene(video_path, 30.0)

    print(f"Enhanced CV shot detection with threshold {threshold}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        return detect_shots_pyscene(video_path, 30.0)

    shots = []
    boundaries = []

    prev_features = None
    frame_idx = 0

    # Sample every 3rd frame for performance
    step = max(1, int(fps / 10))  # Sample ~10 frames per second

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            current_features = extract_visual_features(frame)

            if prev_features is not None:
                diff = compute_frame_difference(prev_features, current_features)

                if diff > threshold:
                    boundaries.append(
                        {
                            "frame": frame_idx,
                            "time": frame_idx / fps,
                            "confidence": min(1.0, diff / threshold),
                            "features": current_features,
                        }
                    )

            prev_features = current_features

        frame_idx += 1

    cap.release()

    # Post-process boundaries to remove too-close detections
    filtered_boundaries = []
    min_shot_length = fps * 0.5  # Minimum 0.5 seconds per shot

    for boundary in boundaries:
        if (
            not filtered_boundaries
            or (boundary["frame"] - filtered_boundaries[-1]["frame"]) > min_shot_length
        ):
            filtered_boundaries.append(boundary)

    # Convert to shots
    start_frame = 0
    for i, boundary in enumerate(filtered_boundaries):
        shots.append(
            ShotInfo(
                shot_index=i,
                start_frame=start_frame,
                end_frame=boundary["frame"],
                start_time=start_frame / fps,
                end_time=boundary["time"],
                confidence=boundary["confidence"],
                features=boundary.get("features"),
            )
        )
        start_frame = boundary["frame"] + 1

    # Add final shot
    if start_frame < frame_count:
        shots.append(
            ShotInfo(
                shot_index=len(shots),
                start_frame=start_frame,
                end_frame=frame_count - 1,
                start_time=start_frame / fps,
                end_time=(frame_count - 1) / fps,
                confidence=1.0,
            )
        )

    print(f"Enhanced CV detected {len(shots)} shots")
    return shots


def detect_shots_transnet(video_path: str, threshold: float = 0.5) -> List[ShotInfo]:
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
            temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            cmd = f"ffmpeg -i {video_path} -r 25 -c:v libx264 -preset fast {temp_video}"
            subprocess.run(cmd, shell=True, check=True)
            video_for_detection = temp_video

            cap = cv2.VideoCapture(video_for_detection)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        _, single_frame_predictions, _ = model.predict_video(video_for_detection)

        shot_boundaries = [
            i for i, pred in enumerate(single_frame_predictions) if pred > threshold
        ]

        shots = []
        start_frame = 0

        for i, boundary_frame in enumerate(shot_boundaries):
            shots.append(
                ShotInfo(
                    shot_index=i,
                    start_frame=start_frame,
                    end_frame=boundary_frame,
                    start_time=start_frame / fps,
                    end_time=boundary_frame / fps,
                    confidence=float(single_frame_predictions[boundary_frame]),
                )
            )
            start_frame = boundary_frame + 1

        if start_frame < frame_count:
            shots.append(
                ShotInfo(
                    shot_index=len(shots),
                    start_frame=start_frame,
                    end_frame=frame_count - 1,
                    start_time=start_frame / fps,
                    end_time=(frame_count - 1) / fps,
                    confidence=1.0,
                )
            )

        print(f"TransNetV2 detected {len(shots)} shots")
        return shots

    except Exception as e:
        print(f"TransNetV2 detection failed: {e}")
        return detect_shots_enhanced_cv(video_path, 0.3)

    finally:
        if temp_video and os.path.exists(temp_video):
            os.unlink(temp_video)


def detect_shots_pyscene(video_path: str, threshold: float = 30.0) -> List[ShotInfo]:
    """
    Fallback shot detection using PySceneDetect.
    """
    from scenedetect import SceneManager, VideoManager
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
        shots.append(
            ShotInfo(
                shot_index=i,
                start_frame=start.get_frames(),
                end_frame=end.get_frames(),
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
                confidence=1.0,
            )
        )
    return shots


def detect_shots_hybrid(video_path: str, threshold: float = 0.4) -> List[ShotInfo]:
    """Hybrid approach combining multiple detection methods for best accuracy."""
    print("Using hybrid shot detection approach")

    results = []

    # Try TransNetV2 first (most accurate for neural cuts)
    if TRANSNET_AVAILABLE:
        try:
            transnet_shots = detect_shots_transnet(video_path, threshold)
            results.append(("transnet", transnet_shots))
        except Exception as e:
            print(f"TransNetV2 failed: {e}")

    # Try enhanced CV method
    try:
        cv_shots = detect_shots_enhanced_cv(video_path, threshold)
        results.append(("enhanced_cv", cv_shots))
    except Exception as e:
        print(f"Enhanced CV failed: {e}")

    # Fallback to PySceneDetect
    if not results:
        try:
            pyscene_shots = detect_shots_pyscene(video_path, 25.0)
            results.append(("pyscene", pyscene_shots))
        except Exception as e:
            print(f"PySceneDetect failed: {e}")
            return []

    # If we have multiple results, combine them intelligently
    if len(results) > 1:
        return combine_shot_results(results, video_path)
    else:
        return results[0][1] if results else []


def combine_shot_results(
    results: List[Tuple[str, List[ShotInfo]]], video_path: str
) -> List[ShotInfo]:
    """Combine multiple shot detection results using consensus."""
    if not results:
        return []

    # If only one result, return it
    if len(results) == 1:
        return results[0][1]

    # Get video info
    cap = cv2.VideoCapture(video_path) if CV2_AVAILABLE else None
    if cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
    else:
        fps = 25.0  # Default
        duration = 60.0  # Default

    # Collect all boundary points from all methods
    all_boundaries = []
    for method_name, shots in results:
        weight = {"transnet": 1.0, "enhanced_cv": 0.8, "pyscene": 0.6}.get(
            method_name, 0.5
        )

        for shot in shots[:-1]:  # Exclude last shot (end boundary)
            all_boundaries.append(
                {
                    "time": shot.end_time,
                    "frame": shot.end_frame,
                    "confidence": shot.confidence * weight,
                    "method": method_name,
                }
            )

    # Cluster nearby boundaries
    consensus_boundaries = []
    tolerance = 1.0  # 1 second tolerance

    sorted_boundaries = sorted(all_boundaries, key=lambda x: x["time"])

    i = 0
    while i < len(sorted_boundaries):
        cluster = [sorted_boundaries[i]]
        j = i + 1

        # Find all boundaries within tolerance
        while (
            j < len(sorted_boundaries)
            and (sorted_boundaries[j]["time"] - sorted_boundaries[i]["time"])
            <= tolerance
        ):
            cluster.append(sorted_boundaries[j])
            j += 1

        # If we have consensus (multiple methods agree), keep this boundary
        if len(cluster) >= 2 or cluster[0]["confidence"] > 0.8:
            avg_time = sum(b["time"] for b in cluster) / len(cluster)
            avg_frame = int(avg_time * fps)
            max_confidence = max(b["confidence"] for b in cluster)

            consensus_boundaries.append(
                {"time": avg_time, "frame": avg_frame, "confidence": max_confidence}
            )

        i = j

    # Convert back to shots
    final_shots = []
    start_frame = 0
    start_time = 0.0

    for i, boundary in enumerate(consensus_boundaries):
        final_shots.append(
            ShotInfo(
                shot_index=i,
                start_frame=start_frame,
                end_frame=boundary["frame"],
                start_time=start_time,
                end_time=boundary["time"],
                confidence=boundary["confidence"],
            )
        )
        start_frame = boundary["frame"] + 1
        start_time = boundary["time"]

    # Add final shot
    if consensus_boundaries:
        final_frame = int(duration * fps) - 1
        final_shots.append(
            ShotInfo(
                shot_index=len(final_shots),
                start_frame=start_frame,
                end_frame=final_frame,
                start_time=start_time,
                end_time=duration,
                confidence=1.0,
            )
        )

    print(
        f"Hybrid detection: {len(final_shots)} consensus shots from {len(all_boundaries)} boundaries"
    )
    return final_shots


def detect_shots(
    video_path: str, method: str = "hybrid", threshold: float = 0.4
) -> List[ShotInfo]:
    """
    Shot detection with multiple methods.
    method: "hybrid", "transnet", "enhanced_cv", or "pyscene"
    threshold: varies by method
    """
    if method == "hybrid":
        return detect_shots_hybrid(video_path, threshold)
    elif method == "transnet":
        return detect_shots_transnet(video_path, threshold)
    elif method == "enhanced_cv":
        return detect_shots_enhanced_cv(video_path, threshold)
    else:
        return detect_shots_pyscene(
            video_path, threshold * 60
        )  # Convert to PySceneDetect scale


def process_shots_for_video(
    db: Session,
    video_id: int,
    video_path: str,
    method: str = "hybrid",
    threshold: float = 0.4,
):
    """
    Shot boundary detection and storage with quality filtering.
    """
    print(f"Processing shots for video {video_id} using {method}")

    shots = detect_shots(video_path, method=method, threshold=threshold)

    # Filter out very short shots (less than 0.5 seconds)
    min_duration = 0.5
    filtered_shots = [
        shot for shot in shots if (shot.end_time - shot.start_time) >= min_duration
    ]

    print(
        f"Filtered {len(shots) - len(filtered_shots)} short shots, keeping {len(filtered_shots)}"
    )

    db_shots = []
    for shot in filtered_shots:
        db_shot = Shot(
            video_id=video_id,
            shot_index=shot.shot_index,
            start_frame=shot.start_frame,
            end_frame=shot.end_frame,
            start_time=shot.start_time,
            end_time=shot.end_time,
            transcript=None,
            analysis=None,
        )
        db.add(db_shot)
        db_shots.append(db_shot)

    db.commit()
    print(f"Saved {len(db_shots)} high-quality shots to database")
    return db_shots
