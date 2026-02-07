#!/usr/bin/env python3
"""
DUET-VLM Inference Script for Qwen2.5-VL

This script runs inference with the DUET-VLM model (VisionZip + PyramidDrop)
and compares performance with the base Qwen2.5-VL model.

Usage:
    # Base Qwen2.5-VL (no token reduction)
    python run_inference.py --image path/to/image.jpg --query "Describe this image"
    
    # VisionZip only
    python run_inference.py --image path/to/image.jpg --query "Describe this image" \
        --visionzip --dominant 0.65 --contextual 0.05
    
    # DUET-VLM (VisionZip + PyramidDrop)
    python run_inference.py --image path/to/image.jpg --query "Describe this image" \
        --visionzip --dominant 0.65 --contextual 0.05 \
        --pdrop --layer_list "[7,14,21]" --ratio_list "[0.75,0.5,0.25]"
"""

import argparse
import time
import torch
from PIL import Image
from transformers import AutoProcessor

# Import the DUET-VLM model
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_qwen2_5vl_duet import Qwen2_5_VLForConditionalGeneration


def load_image(image_path: str) -> Image.Image:
    """Load an image from path or URL."""
    if image_path.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    return image.convert("RGB")


def run_inference(args):
    """Run inference with DUET-VLM model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for AMD GPU and warn about VisionZip compatibility
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if "AMD" in device_name or "Radeon" in device_name or "MI" in device_name:
            print(f"\nWARNING: Detected AMD GPU ({device_name})")
            print("VisionZip may have compatibility issues on ROCm/AMD GPUs.")
            print("If you encounter crashes, try running with --no-visionzip\n")
    
    # Load model and processor
    print(f"\nLoading model: {args.model_path}")
    t0 = time.perf_counter()
    
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto" if args.device_map_auto else None,
        attn_implementation="eager",  # Use eager attention for VisionZip compatibility
    )
    
    if not args.device_map_auto:
        model = model.to(device)
    
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f}s")
    
    # Configure DUET-VLM
    if args.visionzip or args.pdrop:
        print("\n" + "="*60)
        print("DUET-VLM Configuration:")
        print("="*60)
        
        layer_list = eval(args.layer_list) if args.layer_list else None
        ratio_list = eval(args.ratio_list) if args.ratio_list else None
        
        model.configure_duet(
            visionzip_enabled=args.visionzip,
            dominant_ratio=args.dominant,
            contextual_ratio=args.contextual,
            cluster_width=args.cluster_width,
            pdrop_enabled=args.pdrop,
            layer_list=layer_list,
            image_token_ratio_list=ratio_list,
        )
        
        print(f"  VisionZip: {'Enabled' if args.visionzip else 'Disabled'}")
        if args.visionzip:
            print(f"    - Dominant ratio: {args.dominant}")
            print(f"    - Contextual ratio: {args.contextual}")
            print(f"    - Cluster width: {args.cluster_width}")
        
        print(f"  PyramidDrop: {'Enabled' if args.pdrop else 'Disabled'}")
        if args.pdrop and layer_list:
            print(f"    - Layer list: {layer_list}")
            print(f"    - Ratio list: {ratio_list}")
        print("="*60)
    else:
        print("\nRunning base Qwen2.5-VL (no token reduction)")
    
    # Load image
    print(f"\nLoading image: {args.image}")
    image = load_image(args.image)
    print(f"Image size: {image.size}")
    
    # Prepare conversation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.query},
            ],
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Warmup
    if args.warmup > 0:
        print(f"\nWarming up for {args.warmup} iteration(s)...")
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete.")
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Run inference
    print("\nRunning inference...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_start = time.perf_counter()
    
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t_end = time.perf_counter()
    
    # Decode output
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Calculate metrics
    total_time_ms = (t_end - t_start) * 1000
    num_generated_tokens = generated_ids.shape[1]
    tokens_per_sec = num_generated_tokens / (t_end - t_start)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nQuery: {args.query}")
    print(f"\nResponse: {response}")
    print("\n" + "-"*60)
    print("Performance Metrics:")
    print(f"  Total time: {total_time_ms:.1f} ms")
    print(f"  Generated tokens: {num_generated_tokens}")
    print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
    
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Peak GPU memory: {peak_memory_gb:.2f} GB")
    
    # Token reduction stats
    if args.visionzip and hasattr(model, '_original_image_tokens') and model._original_image_tokens:
        original_tokens = model._original_image_tokens
        after_vz = model._image_tokens
        reduction_pct = (1 - after_vz / original_tokens) * 100
        print(f"\nToken Reduction (VisionZip):")
        print(f"  Original visual tokens: {original_tokens}")
        print(f"  After VisionZip: {after_vz}")
        print(f"  Reduction: {reduction_pct:.1f}%")
    
    print("="*60)
    
    return {
        "response": response,
        "total_time_ms": total_time_ms,
        "num_tokens": num_generated_tokens,
        "tokens_per_sec": tokens_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="DUET-VLM Inference for Qwen2.5-VL")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path or name of the Qwen2.5-VL model")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--device-map-auto", action="store_true", help="Use automatic device mapping")
    
    # Input arguments
    parser.add_argument("--image", type=str, required=True, help="Path to image file or URL")
    parser.add_argument("--query", type=str, required=True, help="Question about the image")
    
    # Generation arguments
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations")
    
    # VisionZip arguments
    parser.add_argument("--visionzip", action="store_true", help="Enable VisionZip token reduction")
    parser.add_argument("--dominant", type=float, default=0.65, help="Dominant token ratio")
    parser.add_argument("--contextual", type=float, default=0.05, help="Contextual token ratio")
    parser.add_argument("--cluster-width", type=int, default=4, help="Clustering width multiplier")
    
    # PyramidDrop arguments
    parser.add_argument("--pdrop", action="store_true", help="Enable PyramidDrop progressive dropping")
    parser.add_argument("--layer-list", type=str, default=None, 
                        help="Layers to drop at, e.g., '[7,14,21]'")
    parser.add_argument("--ratio-list", type=str, default=None,
                        help="Keep ratios after each layer, e.g., '[0.75,0.5,0.25]'")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.pdrop and (args.layer_list is None or args.ratio_list is None):
        parser.error("--pdrop requires --layer-list and --ratio-list")
    
    run_inference(args)


if __name__ == "__main__":
    main()
