import argparse
import torch
import time
import json
import os
from torch.profiler import profile as torch_profile, ProfilerActivity

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from transformers.generation.streamers import BaseStreamer
from visionzip import visionzip

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    def _maybe_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    timing = {}

    # Reset CUDA peak memory stats to measure peak usage for this run
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    t0 = time.perf_counter()
    if args.layer_list is not None:
        pdrop_infer = True  # whether to use pdrop infer
    pdrop_infer = False
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, pdrop_infer
    )
    _maybe_sync()
    timing["model_load_ms"] = (time.perf_counter() - t0) * 1000.0

    t_vz0 = time.perf_counter()
    # model = visionzip(model, dominant=54, contextual=10)
    _maybe_sync()
    timing["visionzip_wrap_ms"] = (time.perf_counter() - t_vz0) * 1000.0
    model_class_name = type(model).__name__
    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
        model.model.layer_list = eval(args.layer_list)
        model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
        model.model.image_token_ratio_list.insert(0, 1.0)

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    t_prompt0 = time.perf_counter()
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    timing["prompt_build_ms"] = (time.perf_counter() - t_prompt0) * 1000.0

    image_files = image_parser(args)
    t_imgload0 = time.perf_counter()
    images = load_images(image_files)
    timing["image_load_ms"] = (time.perf_counter() - t_imgload0) * 1000.0
    image_sizes = [x.size for x in images]
    t_imgproc0 = time.perf_counter()
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    )
    timing["image_preprocess_ms"] = (time.perf_counter() - t_imgproc0) * 1000.0
    t_imgmove0 = time.perf_counter()
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    _maybe_sync()
    timing["image_move_to_device_ms"] = (time.perf_counter() - t_imgmove0) * 1000.0

    t_tok0 = time.perf_counter()
    input_ids_cpu = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    timing["prompt_tokenize_ms"] = (time.perf_counter() - t_tok0) * 1000.0
    t_idsmove0 = time.perf_counter()
    input_ids = input_ids_cpu.unsqueeze(0).to(device="cuda", non_blocking=True)
    _maybe_sync()
    timing["tokens_move_to_device_ms"] = (time.perf_counter() - t_idsmove0) * 1000.0

    # Timing helper using a custom streamer to observe token emission
    class _TimingStreamer(BaseStreamer):
        def __init__(self, stop_profiler_on_first_token=None):
            self.time_first_token = None
            self._last_put_time = None
            self.decode_step_durations = []  # time between successive token emissions
            self._stop_profiler_cb = stop_profiler_on_first_token
            self._prof_stopped = False

        def put(self, value):
            now = time.perf_counter()
            if self.time_first_token is None:
                self.time_first_token = now
                if self._stop_profiler_cb is not None and not self._prof_stopped:
                    try:
                        self._stop_profiler_cb()
                        self._prof_stopped = True
                    except Exception:
                        pass
            if self._last_put_time is not None:
                self.decode_step_durations.append(now - self._last_put_time)
            self._last_put_time = now

        def end(self):
            pass

    # Optional PyTorch profiler to capture prefill (TTFB) kernels
    prof = None
    prof_stopped = False
    def _stop_profiler():
        nonlocal prof, prof_stopped
        if prof is not None and not prof_stopped:
            try:
                prof.stop()
            except Exception:
                pass
            prof_stopped = True

    timing_streamer = _TimingStreamer(stop_profiler_on_first_token=_stop_profiler if getattr(args, "pt_profile_ttfb", False) else None)

    # Optional warmup to avoid first-iteration CUDA/kernel init overhead in measured run
    if getattr(args, "warmup", 0) and args.warmup > 0:
        with torch.inference_mode():
            for _ in range(args.warmup):
                _ = model.generate(
                    input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=1,
                    use_cache=True,
                )
                _maybe_sync()

    with torch.inference_mode():
        initial_input_len = input_ids.shape[1]
        _maybe_sync()
        # Start profiler just before generation to isolate prefill
        if getattr(args, "pt_profile_ttfb", False):
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            prof = torch_profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            try:
                prof.start()
            except Exception:
                prof = None
        t_start = time.perf_counter()
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            streamer=timing_streamer,
        )
        _maybe_sync()
        t_end = time.perf_counter()
        # Ensure profiler stopped even if no tokens streamed
        if prof is not None and not prof_stopped:
            _stop_profiler()

    t_decode0 = time.perf_counter()
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    timing["text_decode_ms"] = (time.perf_counter() - t_decode0) * 1000.0
    print(outputs)

    # Report timings
    total_ms = (t_end - t_start) * 1000.0
    prefill_ms = None if timing_streamer.time_first_token is None else (timing_streamer.time_first_token - t_start) * 1000.0
    tokens_after_first = len(timing_streamer.decode_step_durations)
    tokens_generated = int(output_ids.shape[1] - initial_input_len)
    avg_decode_ms = None
    if tokens_after_first > 0:
        avg_decode_ms = (sum(timing_streamer.decode_step_durations) / tokens_after_first) * 1000.0

    print(f"Total time: {total_ms:.1f} ms")
    if prefill_ms is not None:
        print(f"Prefilling time (time-to-first-token): {prefill_ms:.1f} ms")
    else:
        print("Prefilling time (time-to-first-token): N/A")
    if avg_decode_ms is not None:
        print(f"Average decoding time per token: {avg_decode_ms:.1f} ms (over {tokens_after_first} token(s) after first)")
    else:
        print("Average decoding time per token: N/A")

    # Detailed breakdown
    timing["generate_total_ms"] = total_ms
    timing["prefill_ttfb_ms"] = prefill_ms
    timing["avg_decode_per_token_ms"] = avg_decode_ms
    timing["tokens_after_first"] = tokens_after_first
    timing["tokens_generated"] = tokens_generated
    timing["initial_input_tokens"] = int(initial_input_len)
    if prefill_ms is not None and tokens_generated > 0 and t_end > (timing_streamer.time_first_token or t_end):
        timing["decode_throughput_tokens_per_s"] = tokens_after_first / (t_end - (timing_streamer.time_first_token or t_end))
    else:
        timing["decode_throughput_tokens_per_s"] = None

    # GPU memory usage reporting
    if torch.cuda.is_available():
        _maybe_sync()
        try:
            max_alloc = torch.cuda.max_memory_allocated()
            max_res = torch.cuda.max_memory_reserved()
            cur_alloc = torch.cuda.memory_allocated()
            cur_res = torch.cuda.memory_reserved()
            timing["gpu_max_memory_allocated_bytes"] = int(max_alloc)
            timing["gpu_max_memory_reserved_bytes"] = int(max_res)
            timing["gpu_current_memory_allocated_bytes"] = int(cur_alloc)
            timing["gpu_current_memory_reserved_bytes"] = int(cur_res)
            print(f"GPU peak memory allocated: {max_alloc / (1024 ** 3):.2f} GB ({int(max_alloc)} bytes)")
            print(f"GPU peak memory reserved:  {max_res / (1024 ** 3):.2f} GB ({int(max_res)} bytes)")
        except Exception as e:
            timing["gpu_memory_error"] = str(e)

    # print("=== Profiling breakdown (ms) ===")
    # for k in [
    #     "model_load_ms",
    #     "visionzip_wrap_ms",
    #     "prompt_build_ms",
    #     "image_load_ms",
    #     "image_preprocess_ms",
    #     "image_move_to_device_ms",
    #     "prompt_tokenize_ms",
    #     "tokens_move_to_device_ms",
    #     "text_decode_ms",
    #     "generate_total_ms",
    #     "prefill_ttfb_ms",
    #     "avg_decode_per_token_ms",
    # ]:
    #     if k in timing and timing[k] is not None:
    #         print(f"{k}: {timing[k]:.1f}")

    if getattr(args, "profile_json", None):
        try:
            os.makedirs(os.path.dirname(args.profile_json), exist_ok=True) if os.path.dirname(args.profile_json) else None
            with open(args.profile_json, "w") as f:
                json.dump(timing, f, indent=2)
            print(f"Saved profiling JSON to {args.profile_json}")
        except Exception as e:
            print(f"Failed to save profiling JSON: {e}")

    # Export torch profiler trace and optional summary
    if prof is not None:
        try:
            trace_path = args.pt_trace_path if getattr(args, "pt_trace_path", None) else "/tmp/llava_ttfb_trace.json"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True) if os.path.dirname(trace_path) else None
            prof.export_chrome_trace(trace_path)
            print(f"Saved TTFB torch profiler Chrome trace to {trace_path}")
            if getattr(args, "pt_summary", False):
                try:
                    print("\nTop CUDA kernels by self time:")
                    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
                except Exception:
                    pass
                try:
                    print("\nTop CPU ops by self time:")
                    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
                except Exception:
                    pass
        except Exception as e:
            print(f"Failed to export torch profiler trace: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--layer_list", type=str, default= None)
    parser.add_argument("--image_token_ratio_list", type=str, default= None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--profile_json", type=str, default=None)
    parser.add_argument("--pt_profile_ttfb", action='store_true', help="Enable torch.profiler for prefill (TTFB) until first token")
    parser.add_argument("--pt_trace_path", type=str, default=None, help="Path to save Chrome trace (.json)")
    parser.add_argument("--pt_summary", action='store_true', help="Print short torch.profiler CPU/CUDA summary")
    args = parser.parse_args()

    eval_model(args)
