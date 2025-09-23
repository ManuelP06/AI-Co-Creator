import gc
import os
import sys
import traceback
from contextlib import contextmanager
from typing import Optional, Tuple

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from app.config import settings
from app.core.logging_config import get_logger
from app.services.gpu_manager import gpu_manager

logger = get_logger("video_understanding")

MODEL_NAME = settings.video_llama_model

_model = None
_processor = None
_model_on_cpu = False
_last_config_hash = None


def get_gpu_memory_info():
    """Get detailed GPU memory info with better precision"""
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = total - reserved
            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "free_gb": free,
                "utilization": (reserved / total) * 100,
            }
        except Exception as e:
            print(f"GPU memory check failed: {e}")
    return None


@contextmanager
def memory_context():
    """Context manager for memory cleanup"""
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def aggressive_cleanup():
    """Enhanced memory cleanup with multiple passes"""
    # Multiple cleanup passes for stubborn memory
    for i in range(3):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Force Python garbage collection
    if hasattr(sys, "_clear_type_cache"):
        sys._clear_type_cache()


def force_unload_all_models():
    """Complete model cleanup with proper error handling"""
    global _model, _processor, _model_on_cpu, _last_config_hash

    try:
        if _model is not None:
            # Move to CPU before deletion to avoid meta tensor issues
            try:
                if hasattr(_model, "cpu") and not _model_on_cpu:
                    print("Moving model to CPU before deletion...")
                    _model = _model.cpu()
            except Exception as e:
                print(f"Warning: Could not move model to CPU: {e}")

            del _model
            _model = None

        if _processor is not None:
            del _processor
            _processor = None

        _model_on_cpu = False
        _last_config_hash = None

        aggressive_cleanup()
        print("Model unloaded successfully")

    except Exception as e:
        print(f"Error during model unloading: {e}")
        # Force cleanup even if there were errors
        _model = None
        _processor = None
        _model_on_cpu = False
        _last_config_hash = None
        aggressive_cleanup()


def _load_model_once(force_cpu=False) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    """Enhanced model loading with better error handling"""
    global _model, _processor, _model_on_cpu, _last_config_hash

    # Create configuration hash to detect changes
    current_config_hash = hash((force_cpu, MODEL_NAME))

    # Return existing model if configuration matches exactly
    if (
        _model is not None
        and _processor is not None
        and force_cpu == _model_on_cpu
        and current_config_hash == _last_config_hash
    ):
        print(
            f"Model already loaded in correct mode ({'CPU' if _model_on_cpu else 'GPU'})"
        )
        return _model, _processor

    # Unload if configuration doesn't match
    if _model is not None or _processor is not None:
        print(f"Configuration change detected, reloading model...")
        force_unload_all_models()

    gpu_info = get_gpu_memory_info()

    # Enhanced GPU memory check for RTX 5080
    if not force_cpu and torch.cuda.is_available() and gpu_info:
        if gpu_info["free_gb"] < 10.0:  # More conservative threshold for RTX 5080
            print(
                f"Insufficient GPU memory ({gpu_info['free_gb']:.1f}GB free), using CPU mode"
            )
            force_cpu = True
        elif gpu_info["utilization"] > 85:
            print(
                f"High GPU utilization ({gpu_info['utilization']:.1f}%), using CPU mode"
            )
            force_cpu = True

    # Configure loading parameters
    if force_cpu:
        dtype = torch.float32
        max_mem = {"cpu": "48GB"}
        device_map = "cpu"
        print("Loading in CPU mode with float32 precision")
    else:
        dtype = torch.float16
        max_mem = {0: "12GB", "cpu": "48GB"}  # More conservative for RTX 5080
        device_map = "auto"
        print("Loading in GPU mode with float16 precision")

    print(f"Target device: {'CPU' if force_cpu else 'GPU'}")
    print(f"Memory limits: {max_mem}")
    print(f"Model dtype: {dtype}")

    loading_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "max_memory": max_mem,
        "offload_folder": "./offload_cache",
        "offload_state_dict": True,  # Additional offloading
    }

    try:
        print("Loading model weights...")

        with memory_context():
            _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **loading_kwargs)
            _processor = AutoProcessor.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )

        _model.eval()
        _model_on_cpu = force_cpu
        _last_config_hash = current_config_hash

        gpu_mem = get_gpu_memory_info()
        ram_usage = psutil.virtual_memory().percent

        print(f"Model loaded successfully ({'CPU' if force_cpu else 'GPU'} mode)")
        if gpu_mem:
            print(
                f"GPU: {gpu_mem['allocated_gb']:.1f}GB allocated, {gpu_mem['free_gb']:.1f}GB free"
            )
        print(f"RAM: {ram_usage:.1f}% used")

        return _model, _processor

    except Exception as e:
        print(f"Model loading failed: {e}")
        traceback.print_exc()
        force_unload_all_models()
        raise


def safe_tensor_preparation(inputs, model, target_cpu=False):
    """Safe tensor placement with comprehensive error handling"""

    if inputs is None:
        raise ValueError("Inputs cannot be None")

    if target_cpu or _model_on_cpu:
        target_device = torch.device("cpu")
        target_dtype = torch.float32
        print("Preparing tensors for CPU inference (float32)")
    else:
        try:
            # Get model device/dtype from first parameter
            first_param = next(iter(model.parameters()))
            target_device = first_param.device
            target_dtype = first_param.dtype
            print(f"Preparing tensors for {target_device} inference ({target_dtype})")
        except (StopIteration, RuntimeError):
            print("Warning: Could not determine model device, using CPU fallback")
            target_device = torch.device("cpu")
            target_dtype = torch.float32

    converted_inputs = {}

    for key, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            try:
                # Move to target device
                if tensor.device != target_device:
                    tensor = tensor.to(target_device)

                # Convert dtype for floating point tensors
                if key == "pixel_values" or tensor.dtype.is_floating_point:
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(target_dtype)

                converted_inputs[key] = tensor
                print(f"  {key}: {tensor.shape} → {target_device} ({tensor.dtype})")

            except Exception as e:
                print(f"Error converting {key}: {e}")
                # Emergency CPU fallback
                try:
                    fallback_dtype = (
                        torch.float32
                        if tensor.dtype.is_floating_point
                        else tensor.dtype
                    )
                    converted_inputs[key] = tensor.cpu().to(fallback_dtype)
                    print(f"  {key}: Emergency CPU fallback")
                except Exception as fallback_error:
                    print(f"Critical: Could not convert {key} to CPU: {fallback_error}")
                    raise
        else:
            converted_inputs[key] = tensor

    return converted_inputs


def validate_video_segment(video_path: str, start_time: float, end_time: float) -> bool:
    """Validate video segment before processing"""
    try:
        duration = end_time - start_time

        if duration <= 0:
            print(f"Invalid duration: {duration:.1f}s")
            return False

        if duration > 300:  # 5 minutes max
            print(f"Segment too long: {duration:.1f}s (max 300s)")
            return False

        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False

        return True

    except Exception as e:
        print(f"Validation error: {e}")
        return False


def analyze_shot(
    video_path: str,
    start_time: float,
    end_time: float,
    fps: float = 0.25,
    max_frames: int = 6,
    retry_on_error: bool = True,
    force_cpu: bool = False,
) -> str:
    """Enhanced shot analysis with comprehensive error handling"""

    duration = end_time - start_time
    print(
        f"\nAnalyzing shot: {start_time:.1f}s → {end_time:.1f}s ({duration:.1f}s duration)"
    )

    # Validate segment first
    if not validate_video_segment(video_path, start_time, end_time):
        return "Validation failed: Invalid video segment parameters"

    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"GPU memory before: {gpu_info['free_gb']:.2f}GB free")

    # Initialize variables to prevent scope issues
    inputs = None
    output_ids = None

    try:
        # Load model with appropriate configuration
        model, processor = _load_model_once(force_cpu=force_cpu)

        if model is None or processor is None:
            return "Model loading failed: Model or processor is None"

        # Create conversation
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful video analysis assistant.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": video_path,
                            "fps": fps,
                            "max_frames": max_frames,
                            "start_time": start_time,
                            "end_time": end_time,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Analyze this {duration:.1f}s video segment. Describe the key visual elements, actions, setting, and any visible text or objects. Be concise but thorough.",
                    },
                ],
            },
        ]

        print("Processing video segment...")

        # Process conversation with error handling
        try:
            inputs = processor(
                conversation=conversation,
                return_tensors="pt",
                add_system_prompt=True,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"Processor error: {e}")
            return f"Processing failed: {str(e)}"

        if inputs is None:
            return "Processing failed: Processor returned None"

        # Check GPU memory after processing
        if not force_cpu and not _model_on_cpu:
            gpu_info = get_gpu_memory_info()
            if gpu_info and gpu_info["free_gb"] < 3.0:
                print(
                    f"Low GPU memory ({gpu_info['free_gb']:.1f}GB), switching to CPU for this inference"
                )
                force_unload_all_models()
                model, processor = _load_model_once(force_cpu=True)

        # Prepare tensors safely
        try:
            inputs = safe_tensor_preparation(
                inputs, model, target_cpu=(force_cpu or _model_on_cpu)
            )
        except Exception as e:
            print(f"Tensor preparation failed: {e}")
            return f"Tensor preparation failed: {str(e)}"

        # Enhanced generation config for RTX 5080
        generation_config = {
            "max_new_tokens": 50 if (_model_on_cpu or force_cpu) else 100,
            "do_sample": False,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": False,
            "temperature": None,  # Disable sampling
            "top_p": None,
            "top_k": None,
        }

        print(
            f"Starting inference ({'CPU' if (_model_on_cpu or force_cpu) else 'GPU'} mode)..."
        )

        # Clear cache before inference
        if not (_model_on_cpu or force_cpu):
            torch.cuda.empty_cache()

        # Generation with comprehensive error handling
        try:
            with torch.inference_mode():
                with torch.no_grad():
                    output_ids = model.generate(**inputs, **generation_config)
        except Exception as e:
            print(f"Generation failed: {e}")
            return f"Generation failed: {str(e)}"

        if output_ids is None:
            return "Generation failed: No output generated"

        # Decode results
        try:
            generated_text = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
        except Exception as e:
            print(f"Decoding failed: {e}")
            return f"Decoding failed: {str(e)}"

        # Clean up generated text
        if "assistant" in generated_text.lower():
            result = generated_text.split("assistant")[-1].strip()
        else:
            result = generated_text.strip()

        result = result.replace("</s>", "").strip()

        if not result or len(result) < 5:
            return "Analysis produced empty or very short result"

        print(f"Analysis complete: {len(result)} characters")
        return result

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}")

        # Cleanup current tensors
        if "inputs" in locals() and inputs is not None:
            del inputs
        if "output_ids" in locals() and output_ids is not None:
            del output_ids
        aggressive_cleanup()

        # Retry with CPU if not already tried
        if retry_on_error and not (force_cpu or _model_on_cpu):
            print("Retrying with CPU mode...")
            return analyze_shot(
                video_path,
                start_time,
                end_time,
                fps=max(0.1, fps * 0.5),
                max_frames=max(2, max_frames - 2),
                retry_on_error=False,
                force_cpu=True,
            )
        else:
            return f"Memory error: Segment too large for available resources (tried both GPU and CPU)"

    except RuntimeError as e:
        error_msg = str(e).lower()
        print(f"Runtime error: {e}")

        if "input type" in error_msg and "weight type" in error_msg:
            print("Dtype mismatch detected")
            if retry_on_error and not force_cpu:
                print("Retrying with CPU mode to resolve dtype issue...")
                return analyze_shot(
                    video_path,
                    start_time,
                    end_time,
                    fps=fps,
                    max_frames=max_frames,
                    retry_on_error=False,
                    force_cpu=True,
                )

        return f"Runtime error: {str(e)}"

    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return f"Analysis failed: {str(e)}"

    finally:
        # Safe cleanup of local variables
        try:
            if "inputs" in locals() and inputs is not None:
                del inputs
            if "output_ids" in locals() and output_ids is not None:
                del output_ids
        except:
            pass

        gc.collect()
        if torch.cuda.is_available() and not (_model_on_cpu or force_cpu):
            torch.cuda.empty_cache()


def intelligent_segment_processing(
    video_path: str, start_time: float, end_time: float
) -> str:
    """Enhanced processing with dynamic parameter optimization for RTX 5080"""

    duration = end_time - start_time
    gpu_info = get_gpu_memory_info()

    # Enhanced parameter selection for RTX 5080
    if duration <= 10:
        fps, max_frames = 0.5, 10  # Higher quality for short segments
    elif duration <= 20:
        fps, max_frames = 0.4, 8
    elif duration <= 45:
        fps, max_frames = 0.3, 6
    elif duration <= 90:
        fps, max_frames = 0.25, 4
    else:
        fps, max_frames = 0.2, 3

    force_cpu = False
    if gpu_info:
        if gpu_info["free_gb"] < 8.0:  # Conservative for RTX 5080
            force_cpu = True
            print(f"GPU memory limited ({gpu_info['free_gb']:.1f}GB), using CPU mode")
        elif gpu_info["free_gb"] < 10.0:
            fps = min(fps, 0.2)
            max_frames = min(max_frames, 4)
            print(f"Reducing parameters due to memory constraints")

    print(f"Processing {duration:.1f}s segment: {fps} fps, {max_frames} max frames")

    # Split very long segments
    if duration > 150:  # More aggressive splitting
        print(f"Splitting long segment ({duration:.1f}s)")
        mid_point = start_time + duration / 2

        part1 = analyze_shot(
            video_path,
            start_time,
            mid_point,
            fps=0.15,
            max_frames=3,
            force_cpu=force_cpu,
        )
        aggressive_cleanup()
        part2 = analyze_shot(
            video_path, mid_point, end_time, fps=0.15, max_frames=3, force_cpu=force_cpu
        )

        return f"First half: {part1.strip()} Second half: {part2.strip()}"

    return analyze_shot(
        video_path, start_time, end_time, fps, max_frames, force_cpu=force_cpu
    )


def run_video_analysis(db, video, per_shot: bool = True, prefer_gpu: bool = True):
    """
    Enhanced production video analysis optimized for RTX 5080 + Ryzen 9 9950X + 64GB RAM

    Args:
        db: Database session
        video: Video object with shots
        per_shot: Whether to analyze per shot or full video
        prefer_gpu: Whether to prefer GPU over CPU (will fallback automatically)
    """

    print(f"Starting VideoLLaMA3 analysis for video ID: {video.id}")
    print(f"System specs: RTX 5080 16GB | Ryzen 9 9950X | 64GB RAM")

    ram_info = psutil.virtual_memory()
    gpu_info = get_gpu_memory_info()

    print(
        f"RAM: {ram_info.available / 1024**3:.1f}GB available / {ram_info.total / 1024**3:.1f}GB total"
    )
    if gpu_info:
        print(
            f"GPU: {gpu_info['free_gb']:.1f}GB free / {gpu_info['total_gb']:.1f}GB total"
        )

    # Start fresh
    force_unload_all_models()

    shot_count = 0
    error_count = 0
    total_shots = len(video.shots) if per_shot else 1

    try:
        if per_shot:
            print(f"Processing {total_shots} shots individually")

            for i, shot in enumerate(video.shots):
                print(f"\n{'='*60}")
                print(f"Shot {i+1}/{total_shots} (ID: {shot.id})")
                print(
                    f"Duration: {shot.start_time:.1f}s → {shot.end_time:.1f}s ({shot.end_time - shot.start_time:.1f}s)"
                )

                # Skip already analyzed shots (unless they have errors)
                if (
                    shot.analysis
                    and shot.analysis.strip()
                    and not any(
                        error in shot.analysis
                        for error in [
                            "Analysis error",
                            "Failed to load",
                            "OOM error",
                            "Processing error",
                            "Memory error",
                            "Runtime error",
                            "Processing failed",
                            "Analysis failed",
                            "Model loading failed",
                            "Validation failed",
                        ]
                    )
                ):
                    print(f"Shot {i+1} already analyzed, skipping")
                    shot_count += 1
                    continue

                try:
                    current_gpu = get_gpu_memory_info()
                    if current_gpu:
                        print(f"Pre-analysis GPU: {current_gpu['free_gb']:.2f}GB free")

                    # Enhanced analysis with validation
                    analysis_result = intelligent_segment_processing(
                        video.file_path, shot.start_time, shot.end_time
                    )

                    if analysis_result and len(analysis_result.strip()) > 10:
                        shot.analysis = analysis_result.strip()
                        shot_count += 1
                        print(f"Success: '{analysis_result[:50]}...'")
                    else:
                        shot.analysis = "Analysis produced empty or invalid result"
                        error_count += 1
                        print(f"Empty result for shot {i+1}")

                    db.add(shot)

                    # Save progress more frequently
                    if (shot_count + error_count) % 2 == 0:
                        print("Saving progress...")
                        db.commit()

                except Exception as e:
                    print(f"Error processing shot {i+1}: {e}")
                    shot.analysis = f"Processing failed: {str(e)[:100]}"
                    db.add(shot)
                    error_count += 1

                    # Log detailed error for debugging
                    traceback.print_exc()

                # More frequent cleanup for RTX 5080
                if (shot_count + error_count) % 2 == 0:
                    print("Periodic cleanup...")
                    aggressive_cleanup()

        print("\nFinal save...")
        db.commit()

        print(f"\nAnalysis Complete!")
        print(f"Successful: {shot_count}")
        print(f"Errors: {error_count}")
        success_rate = (
            (shot_count / (shot_count + error_count) * 100)
            if (shot_count + error_count) > 0
            else 0
        )
        print(f"Success rate: {success_rate:.1f}%")

        if error_count > 0:
            print(f"Tip: Consider running with force_cpu=True for problematic segments")
            print(f"Tip: Check video file integrity and segment boundaries")

    except Exception as e:
        print(f"Critical error in analysis pipeline: {e}")
        traceback.print_exc()
        db.rollback()
        raise

    finally:
        print("Final cleanup...")
        force_unload_all_models()


def unload_model():
    """Public interface for model cleanup"""
    force_unload_all_models()


def force_cpu_analysis(db, video):
    """Force CPU-only analysis for maximum stability"""
    print("Running in CPU-only mode for maximum stability")
    return run_video_analysis(db, video, per_shot=True, prefer_gpu=False)


def gpu_analysis_with_fallback(db, video):
    """GPU analysis with intelligent CPU fallback for RTX 5080"""
    print("Running with GPU preference and automatic CPU fallback (RTX 5080 optimized)")
    return run_video_analysis(db, video, per_shot=True, prefer_gpu=True)


def debug_shot_analysis(video_path: str, shot_index: int, shots_data):
    """Debug specific problematic shot"""
    if shot_index >= len(shots_data):
        print(f"Invalid shot index: {shot_index}")
        return

    shot = shots_data[shot_index]
    print(f"Debug analysis for shot {shot_index + 1}")
    print(f"Video: {video_path}")
    print(f"⏱Time: {shot['start_time']:.1f}s → {shot['end_time']:.1f}s")

    # Try with minimal parameters
    result = analyze_shot(
        video_path,
        shot["start_time"],
        shot["end_time"],
        fps=0.1,
        max_frames=2,
        force_cpu=True,
    )

    print(f"Result: {result}")
    return result


# Additional utility functions for troubleshooting


def check_system_resources():
    """Check system resources and provide recommendations"""
    ram_info = psutil.virtual_memory()
    gpu_info = get_gpu_memory_info()

    print("System Resource Check:")
    print(
        f"RAM: {ram_info.available / 1024**3:.1f}GB available / {ram_info.total / 1024**3:.1f}GB total ({ram_info.percent:.1f}% used)"
    )

    if gpu_info:
        print(
            f"GPU: {gpu_info['free_gb']:.1f}GB free / {gpu_info['total_gb']:.1f}GB total ({gpu_info['utilization']:.1f}% used)"
        )

        if gpu_info["free_gb"] < 8.0:
            print("Warning: Low GPU memory - recommend CPU mode")
        elif gpu_info["utilization"] > 90:
            print("Warning: High GPU utilization")
        else:
            print("GPU memory looks good")
    else:
        print("No GPU detected")

    if ram_info.percent > 85:
        print("Warning: High RAM usage")
    else:
        print("RAM usage looks good")


def test_model_loading():
    """Test model loading without video processing"""
    print("Testing model loading...")

    try:
        model, processor = _load_model_once(force_cpu=False)
        print("GPU model loading successful")
        force_unload_all_models()

        model, processor = _load_model_once(force_cpu=True)
        print("CPU model loading successful")
        force_unload_all_models()

        print("Model loading test passed")
        return True

    except Exception as e:
        print(f"Model loading test failed: {e}")
        traceback.print_exc()
        return False
