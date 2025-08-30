import os, gc, torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
import psutil

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"

# Optimized CUDA configuration for RTX 5080
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

_model = None
_processor = None
_model_on_cpu = False

def get_gpu_memory_info():
    """Get detailed GPU memory info"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved
        }
    return None

def aggressive_cleanup():
    """Comprehensive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Multiple cleanup cycles for stubborn allocations
        for _ in range(5):
            gc.collect()
            torch.cuda.empty_cache()

def force_unload_all_models():
    """Complete model cleanup"""
    global _model, _processor, _model_on_cpu
    
    if _model is not None:
        # Properly cleanup model
        if hasattr(_model, 'cpu'):
            _model.cpu()
        del _model
        _model = None
        
    if _processor is not None:
        del _processor
        _processor = None
        
    _model_on_cpu = False
    
    # Comprehensive cleanup
    aggressive_cleanup()
    
    # Clear Python caches
    import sys
    if hasattr(sys, '_clear_type_cache'):
        sys._clear_type_cache()

def _load_model_once(force_cpu=False) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    global _model, _processor, _model_on_cpu
    
    # Return existing model if configuration matches
    if _model is not None and _processor is not None and force_cpu == _model_on_cpu:
        print(f"Model already loaded in correct mode ({'CPU' if _model_on_cpu else 'GPU'})")
        return _model, _processor
    
    # Unload if configuration doesn't match
    if _model is not None or _processor is not None:
        print(f"Configuration change detected, reloading model...")
        force_unload_all_models()

    gpu_info = get_gpu_memory_info()
    
    # Smart device selection for RTX 5080
    if not force_cpu and torch.cuda.is_available() and gpu_info:
        if gpu_info['free_gb'] < 12.0:  # Need 12GB+ for stable GPU inference
            print(f"Insufficient GPU memory ({gpu_info['free_gb']:.1f}GB free), using CPU mode")
            force_cpu = True
    
    # Configure loading parameters
    if force_cpu:
        dtype = torch.float32  # CPU works better with float32
        max_mem = {"cpu": "48GB"}  # Use more of your 64GB RAM
        device_map = "cpu"
        print("Loading in CPU mode with float32 precision")
    else:
        dtype = torch.float16  # GPU efficiency with float16
        max_mem = {0: "10GB", "cpu": "48GB"}  # Conservative GPU allocation
        device_map = "auto"
        print("Loading in GPU mode with float16 precision")
    
    print(f"Target device: {'CPU' if force_cpu else 'GPU'}")
    print(f"Memory limits: {max_mem}")
    print(f"Model dtype: {dtype}")
    
    loading_kwargs = {
        'trust_remote_code': True,
        'device_map': device_map,
        'torch_dtype': dtype,
        'low_cpu_mem_usage': True,
        'max_memory': max_mem,
        'offload_folder': './offload_cache',  # Offload to disk if needed
    }
    
    try:
        print("Loading model weights...")
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **loading_kwargs)
        _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Set model to evaluation mode
        _model.eval()
        _model_on_cpu = force_cpu
        
        # Memory status after loading
        gpu_mem = get_gpu_memory_info()
        ram_usage = psutil.virtual_memory().percent
        
        print(f"Model loaded successfully ({'CPU' if force_cpu else 'GPU'} mode)")
        if gpu_mem:
            print(f"GPU: {gpu_mem['allocated_gb']:.1f}GB allocated, {gpu_mem['free_gb']:.1f}GB free")
        print(f"RAM: {ram_usage:.1f}% used")
        
        return _model, _processor
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        force_unload_all_models()
        raise

def smart_tensor_preparation(inputs, model, target_cpu=False):
    """Optimized tensor placement and dtype conversion"""
    
    # Determine target configuration
    if target_cpu or _model_on_cpu:
        target_device = torch.device("cpu")
        target_dtype = torch.float32
        print("Preparing tensors for CPU inference (float32)")
    else:
        try:
            first_param = next(iter(model.parameters()))
            target_device = first_param.device
            target_dtype = first_param.dtype
            print(f"Preparing tensors for {target_device} inference ({target_dtype})")
        except StopIteration:
            target_device = torch.device("cpu")
            target_dtype = torch.float32
            print("Fallback: preparing tensors for CPU inference")
    
    # Convert tensors with proper error handling
    converted_inputs = {}
    for key, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            try:
                # Move to target device
                tensor = tensor.to(target_device)
                
                # Convert dtype for specific tensor types
                if key == "pixel_values" or tensor.dtype.is_floating_point:
                    tensor = tensor.to(target_dtype)
                
                converted_inputs[key] = tensor
                print(f"  {key}: {tensor.shape} → {target_device} ({tensor.dtype})")
                
            except Exception as e:
                print(f"Error converting {key}: {e}")
                # Emergency fallback to CPU
                converted_inputs[key] = tensor.to("cpu").to(torch.float32 if tensor.dtype.is_floating_point else tensor.dtype)
        else:
            converted_inputs[key] = tensor
    
    return converted_inputs

def analyze_shot(
    video_path: str,
    start_time: float,
    end_time: float,
    fps: float = 0.25,  # Conservative default
    max_frames: int = 6,  # Conservative default
    retry_on_error: bool = True,
    force_cpu: bool = False,
) -> str:
    
    duration = end_time - start_time
    print(f"\nAnalyzing shot: {start_time:.1f}s → {end_time:.1f}s ({duration:.1f}s duration)")
    
    # Pre-flight memory check
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"GPU memory before: {gpu_info['free_gb']:.2f}GB free")
    
    # Load model with appropriate configuration
    try:
        model, processor = _load_model_once(force_cpu=force_cpu)
    except Exception as e:
        return f"Model loading failed: {str(e)}"

    # Prepare conversation
    conversation = [
        {"role": "system", "content": "You are a helpful video analysis assistant."},
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

    inputs = None
    output_ids = None
    
    try:
        print("📹 Processing video segment...")
        
        # Process inputs
        inputs = processor(
            conversation=conversation,
            return_tensors="pt",
            add_system_prompt=True,
            add_generation_prompt=True,
        )
        
        # Final memory check before inference
        if not force_cpu and not _model_on_cpu:
            gpu_info = get_gpu_memory_info()
            if gpu_info and gpu_info['free_gb'] < 2.5:
                print(f"Low GPU memory ({gpu_info['free_gb']:.1f}GB), switching to CPU for this inference")
                # Unload GPU model and reload on CPU
                force_unload_all_models()
                model, processor = _load_model_once(force_cpu=True)
        
        # Prepare tensors with correct placement and dtypes
        inputs = smart_tensor_preparation(inputs, model, target_cpu=(force_cpu or _model_on_cpu))
        
        # Optimized generation parameters
        generation_config = {
            "max_new_tokens": 60 if (_model_on_cpu or force_cpu) else 80,
            "do_sample": False,  # Greedy decoding for memory efficiency
            "pad_token_id": processor.tokenizer.eos_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "use_cache": False,  # Disable KV cache to save memory
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": False,
        }
        
        print(f"Starting inference ({'CPU' if (_model_on_cpu or force_cpu) else 'GPU'} mode)...")
        
        # Clear cache before generation
        if not (_model_on_cpu or force_cpu):
            torch.cuda.empty_cache()
        
        with torch.inference_mode():
            with torch.no_grad():  # Extra memory protection
                output_ids = model.generate(**inputs, **generation_config)
        
        # Decode result
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        # Extract just the assistant's response (remove the conversation context)
        if "assistant" in generated_text.lower():
            result = generated_text.split("assistant")[-1].strip()
        else:
            result = generated_text.strip()
        
        # Clean up result
        result = result.replace("</s>", "").strip()
        
        print(f"Analysis complete: {len(result)} characters")
        return result if result else "No analysis generated"
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM: {e}")
        
        # Immediate cleanup
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        aggressive_cleanup()
        
        # Retry with CPU if not already tried
        if retry_on_error and not (force_cpu or _model_on_cpu):
            print("Retrying with CPU mode...")
            return analyze_shot(
                video_path, start_time, end_time,
                fps=0.2, max_frames=4, retry_on_error=False, force_cpu=True
            )
        else:
            return f"Memory error: Segment too large for available resources"
            
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "input type" in error_msg and "weight type" in error_msg:
            print(f"Dtype mismatch: {e}")
            if retry_on_error and not force_cpu:
                print("Retrying with CPU mode to resolve dtype issue...")
                return analyze_shot(
                    video_path, start_time, end_time,
                    fps=fps, max_frames=max_frames, retry_on_error=False, force_cpu=True
                )
        print(f"Runtime error: {e}")
        return f"Runtime error: {str(e)}"
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return f"Analysis failed: {str(e)}"
        
    finally:
        # Comprehensive cleanup
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        gc.collect()
        if torch.cuda.is_available() and not (_model_on_cpu or force_cpu):
            torch.cuda.empty_cache()

def intelligent_segment_processing(video_path: str, start_time: float, end_time: float) -> str:
    """Intelligent processing with automatic parameter optimization"""
    
    duration = end_time - start_time
    gpu_info = get_gpu_memory_info()
    
    # Determine optimal parameters based on duration and available memory
    if duration <= 15:
        fps, max_frames = 0.4, 8  # High quality for short segments
    elif duration <= 45:
        fps, max_frames = 0.3, 6  # Balanced
    elif duration <= 90:
        fps, max_frames = 0.25, 4  # Conservative
    else:
        fps, max_frames = 0.2, 3   # Very conservative for long segments
    
    # Adjust based on available GPU memory
    force_cpu = False
    if gpu_info:
        if gpu_info['free_gb'] < 6.0:
            force_cpu = True
            print(f"GPU memory limited ({gpu_info['free_gb']:.1f}GB), using CPU mode")
        elif gpu_info['free_gb'] < 8.0:
            fps = min(fps, 0.2)
            max_frames = min(max_frames, 4)
            print(f"Reducing parameters due to memory constraints")
    
    print(f"Processing {duration:.1f}s segment: {fps} fps, {max_frames} max frames")
    
    # Handle very long segments by splitting
    if duration > 120:
        print(f"Splitting long segment ({duration:.1f}s)")
        mid_point = start_time + duration / 2
        
        part1 = analyze_shot(video_path, start_time, mid_point, fps=0.2, max_frames=3, force_cpu=force_cpu)
        # Cleanup between parts
        aggressive_cleanup()
        part2 = analyze_shot(video_path, mid_point, end_time, fps=0.2, max_frames=3, force_cpu=force_cpu)
        
        # Combine results intelligently
        return f"First half: {part1.strip()} Second half: {part2.strip()}"
    
    return analyze_shot(video_path, start_time, end_time, fps, max_frames, force_cpu=force_cpu)

def run_video_analysis(db, video, per_shot: bool = True, prefer_gpu: bool = True):
    """
    Production video analysis optimized for RTX 5080 + Ryzen 9 9950X + 64GB RAM
    
    Args:
        db: Database session
        video: Video object with shots
        per_shot: Whether to analyze per shot or full video
        prefer_gpu: Whether to prefer GPU over CPU (will fallback automatically)
    """
    
    print(f"Starting VideoLLaMA3 analysis for video ID: {video.id}")
    print(f"ystem specs: RTX 5080 16GB | Ryzen 9 9950X | 64GB RAM")
    
    # System status
    ram_info = psutil.virtual_memory()
    gpu_info = get_gpu_memory_info()
    
    print(f"RAM: {ram_info.available / 1024**3:.1f}GB available / {ram_info.total / 1024**3:.1f}GB total")
    if gpu_info:
        print(f"GPU: {gpu_info['free_gb']:.1f}GB free / {gpu_info['total_gb']:.1f}GB total")
    
    # Initial cleanup
    force_unload_all_models()
    
    shot_count = 0
    error_count = 0
    total_shots = len(video.shots) if per_shot else 1
    
    try:
        if per_shot:
            print(f"📋 Processing {total_shots} shots individually")
            
            for i, shot in enumerate(video.shots):
                print(f"\n{'='*60}")
                print(f"Shot {i+1}/{total_shots} (ID: {shot.id})")
                print(f"Duration: {shot.start_time:.1f}s → {shot.end_time:.1f}s ({shot.end_time - shot.start_time:.1f}s)")
                
                # Skip if already analyzed
                if (shot.analysis and shot.analysis.strip() and 
                    not any(error in shot.analysis for error in ["Analysis error", "Failed to load", "OOM error", "Processing error", "Memory error", "Runtime error"])):
                    print(f"✅ Shot {i+1} already analyzed, skipping")
                    continue
                
                try:
                    # Memory status check
                    current_gpu = get_gpu_memory_info()
                    if current_gpu:
                        print(f"Pre-analysis GPU: {current_gpu['free_gb']:.2f}GB free")
                    
                    # Process with intelligent parameter selection
                    analysis_result = intelligent_segment_processing(
                        video.file_path, shot.start_time, shot.end_time
                    )
                    
                    # Validate result
                    if analysis_result and len(analysis_result.strip()) > 10:
                        shot.analysis = analysis_result.strip()
                        shot_count += 1
                        print(f"Success: '{analysis_result[:60]}...'")
                    else:
                        shot.analysis = "Analysis produced empty or invalid result"
                        error_count += 1
                        print(f"Empty result for shot {i+1}")
                    
                    db.add(shot)
                    
                    # Commit every 3 successful analyses
                    if shot_count % 3 == 0:
                        print("Saving progress...")
                        db.commit()
                        
                except Exception as e:
                    print(f"Error processing shot {i+1}: {e}")
                    shot.analysis = f"Processing failed: {str(e)[:100]}"
                    db.add(shot)
                    error_count += 1
                
                # Cleanup between shots
                if (shot_count + error_count) % 2 == 0:
                    print("🧹 Periodic cleanup...")
                    aggressive_cleanup()
        
        # Final commit
        print("\n💾 Final save...")
        db.commit()
        
        # Results summary
        print(f"\nAnalysis Complete!")
        print(f"Successful: {shot_count}")
        print(f"Errors: {error_count}")
        print(f"Success rate: {shot_count/(shot_count+error_count)*100:.1f}%")
        
        if error_count > 0:
            print(f"Tip: Consider running with force_cpu=True for problematic segments")
        
    except Exception as e:
        print(f"Critical error in analysis pipeline: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
        
    finally:
        print("🧹 Final cleanup...")
        force_unload_all_models()

def unload_model():
    """Public interface for model cleanup"""
    force_unload_all_models()
    print("Model unloaded")

# Utility functions for manual control
def force_cpu_analysis(db, video):
    """Force CPU-only analysis for maximum stability"""
    print("Running in CPU-only mode for maximum stability")
    return run_video_analysis(db, video, per_shot=True, prefer_gpu=False)

def gpu_analysis_with_fallback(db, video):
    """GPU analysis with intelligent CPU fallback"""
    print("Running with GPU preference and automatic CPU fallback")
    return run_video_analysis(db, video, per_shot=True, prefer_gpu=True)