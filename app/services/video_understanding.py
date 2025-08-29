import os, gc, torch
from typing import Tuple, Optional
from transformers import AutoModelForCausalLM, AutoProcessor
import psutil

MODEL_NAME = "DAMO-NLP-SG/VideoLLaMA3-7B"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _load_model_once(force_cpu=False) -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    global _model, _processor, _model_on_cpu
    
    if _model is not None and _processor is not None:
        if force_cpu and not _model_on_cpu:
            print("Moving model to CPU...")
            _model = _model.cpu()
            _model_on_cpu = True
            aggressive_cleanup()
        elif not force_cpu and _model_on_cpu:
            gpu_info = get_gpu_memory_info()
            if gpu_info and gpu_info['free_gb'] > 8.0: 
                print("Moving model back to GPU...")
                try:
                    _model = _model.cuda()
                    _model_on_cpu = False
                except torch.cuda.OutOfMemoryError:
                    print("Not enough GPU memory, keeping on CPU")
        
        return _model, _processor

    aggressive_cleanup()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    attn_impl = "flash_attention_2"
    try:
        import flash_attn 
    except Exception:
        attn_impl = "eager"

    max_mem = {}
    if not force_cpu and torch.cuda.is_available():
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            available_gpu = max(8.0, gpu_info['total_gb'] - 2.0)
            max_mem[0] = f"{available_gpu:.0f}GB"
    
    max_mem["cpu"] = "60GB" 
    
    device_map = "cpu" if force_cpu else "auto"

    print(f"Loading VideoLLaMA3 model... (force_cpu={force_cpu}, device_map={device_map})")
    print(f"Memory limits: {max_mem}")
    
    try:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            max_memory=max_mem,
            offload_folder="./model_offload" if not force_cpu else None,  
        )
        
        _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model_on_cpu = force_cpu
        
        gpu_mem = get_gpu_memory_info()
        ram_usage = psutil.virtual_memory().percent
        
        print(f"Model loaded successfully.")
        if gpu_mem:
            print(f"GPU memory: {gpu_mem['reserved_gb']:.1f}GB reserved, {gpu_mem['free_gb']:.1f}GB free")
        print(f"RAM usage: {ram_usage:.1f}%")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        _model = None
        _processor = None
        _model_on_cpu = False
        aggressive_cleanup()
        raise
    
    return _model, _processor

def unload_model():
    global _model, _processor, _model_on_cpu
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    _model_on_cpu = False
    aggressive_cleanup()
    print("Model unloaded from memory")

def smart_tensor_placement(inputs, model, prefer_gpu=True):
    """Smart tensor placement based on available memory"""
    if not prefer_gpu or _model_on_cpu:
        device = "cpu"
    else:
        gpu_info = get_gpu_memory_info()
        if gpu_info and gpu_info['free_gb'] > 1.0:  
            device = next(model.parameters()).device
        else:
            print("Low GPU memory, keeping inputs on CPU")
            device = "cpu"
    
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
            if k == "pixel_values" and not _model_on_cpu:
                inputs[k] = inputs[k].to(model.dtype)
    
    return inputs

def analyze_shot(
    video_path: str,
    start_time: float,
    end_time: float,
    fps: float = 0.5,
    max_frames: int = 16,
    retry_on_oom: bool = True,
) -> str:
    print(f"Analyzing shot {start_time:.1f}s-{end_time:.1f}s")
    
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"GPU mem before: {gpu_info['reserved_gb']*1024:.1f} MB ({gpu_info['free_gb']:.1f}GB free)")
    
    force_cpu = False
    if gpu_info and gpu_info['free_gb'] < 2.0:  
        print("Low GPU memory detected, will try CPU inference")
        force_cpu = True
    
    try:
        model, processor = _load_model_once(force_cpu=force_cpu)
    except Exception as e:
        return f"Failed to load model: {str(e)}"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
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
                    "text": f"Describe this segment from {start_time:.1f}s to {end_time:.1f}s in detail. "
                            f"List salient actions, scene, people, emotions, and on-screen text if any.",
                },
            ],
        },
    ]

    inputs = None
    output_ids = None
    
    try:
        inputs = processor(
            conversation=conversation,
            return_tensors="pt",
            add_system_prompt=True,
            add_generation_prompt=True,
        )

        inputs = smart_tensor_placement(inputs, model, prefer_gpu=not force_cpu)

        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"GPU mem after preprocessing: {gpu_info['reserved_gb']*1024:.1f} MB")

        generation_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,
            "pad_token_id": processor.tokenizer.eos_token_id,
            "use_cache": False,
        }
        
        if _model_on_cpu:
            generation_kwargs["max_new_tokens"] = 128 

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)

        result = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(f"Analysis result length: {len(result)} chars ({'CPU' if _model_on_cpu else 'GPU'} inference)")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM error: {e}")
        
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        aggressive_cleanup()
        
        if retry_on_oom and not force_cpu:
            print("Attempting recovery with CPU inference...")
            return analyze_shot(
                video_path, start_time, end_time, 
                fps=max(0.25, fps/2), max_frames=max(8, max_frames//2), 
                retry_on_oom=False
            )
        else:
            return "OOM error – could not recover"
    
    except Exception as e:
        print(f"Other error during analysis: {e}")
        result = f"Analysis error: {str(e)}"
    
    finally:
        if inputs is not None:
            del inputs
        if output_ids is not None:
            del output_ids
        aggressive_cleanup()
        
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"GPU mem after cleanup: {gpu_info['reserved_gb']*1024:.1f} MB")
    
    return result

def check_segment_memory_requirements(duration: float, fps: float, max_frames: int) -> dict:
    """Estimate memory requirements for a video segment"""
    actual_frames = min(int(duration * fps), max_frames)
    
    frame_size_mb = 3.0 
    processing_overhead = 2.0  
    
    estimated_mb = actual_frames * frame_size_mb * processing_overhead
    
    return {
        'frames': actual_frames,
        'estimated_mb': estimated_mb,
        'estimated_gb': estimated_mb / 1024,
        'duration': duration
    }

def adaptive_segment_processing(video_path: str, start_time: float, end_time: float, 
                              base_fps: float = 0.5, base_max_frames: int = 16) -> str:
    """Process segment with adaptive parameters based on memory"""
    duration = end_time - start_time
    
    memory_req = check_segment_memory_requirements(duration, base_fps, base_max_frames)
    gpu_info = get_gpu_memory_info()
    
    print(f"Segment {start_time:.1f}s-{end_time:.1f}s: {memory_req['frames']} frames, "
          f"~{memory_req['estimated_gb']:.1f}GB estimated")
    
    fps = base_fps
    max_frames = base_max_frames
    
    if gpu_info and memory_req['estimated_gb'] > gpu_info['free_gb']:
        print("Reducing parameters due to memory constraints")
        fps = min(fps, 0.25)
        max_frames = min(max_frames, 8)
        
        if duration > 120:  
            print(f"Splitting long segment ({duration:.1f}s) into chunks")
            mid_point = start_time + duration / 2
            
            result1 = analyze_shot(video_path, start_time, mid_point, fps, max_frames//2)
            result2 = analyze_shot(video_path, mid_point, end_time, fps, max_frames//2)
            
            return f"{result1} {result2}".strip()
    
    return analyze_shot(video_path, start_time, end_time, fps, max_frames)

def run_video_analysis(db, video, per_shot: bool = True):
    """
    VideoLLaMA3 analysis with adaptive memory management
    """
    shot_count = 0
    error_count = 0
    
    print(f"Starting analysis for video ID: {video.id}")
    print(f"Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB")
    
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        print(f"GPU memory: {gpu_info['free_gb']:.1f}GB free / {gpu_info['total_gb']:.1f}GB total")
    
    try:
        if per_shot:
            shots = video.shots
            total_shots = len(shots)
            print(f"Found {total_shots} shots for video")
            
            for i, shot in enumerate(shots):
                print(f"Processing shot {i+1}/{total_shots} (Shot ID: {shot.id})")
                
                if shot.analysis and shot.analysis.strip() and not shot.analysis.startswith("Analysis error"):
                    print(f"Shot {i+1} already has analysis, skipping...")
                    continue
                
                try:
                    analysis_result = adaptive_segment_processing(
                        video.file_path, shot.start_time, shot.end_time
                    )
                    
                    print(f"Got analysis result: '{analysis_result[:100]}...' (length: {len(analysis_result)})")
                    
                    shot.analysis = analysis_result
                    db.add(shot)
                    shot_count += 1
                    
                    if shot_count % 3 == 0:
                        print(f"Committing after {shot_count} shots...")
                        try:
                            db.commit()
                            print("Commit successful")
                        except Exception as e:
                            print(f"Commit failed: {e}")
                            db.rollback()
                            raise
                        
                except Exception as e:
                    print(f"Error processing shot {shot.shot_index}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    shot.analysis = f"Processing error: {str(e)}"
                    db.add(shot)
                    error_count += 1
                
                if (shot_count + error_count) % 20 == 0:
                    print("Performing periodic cleanup...")
                    aggressive_cleanup()
        
        print("Performing final commit...")
        db.commit()
        print(f"Analysis complete. Processed: {shot_count}, Errors: {error_count}")
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise
    finally:
        pass