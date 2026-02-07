import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math

# ============= For PDrop Inference ==================
from llava.aligner import greedy_align_and_filter
from llava.utils import longest_common_subarray
from llava.salient_token_finder import salient_tokens_finder
# ============= For PDrop Inference ==================

# ============= VisionZip ==================
from visionzip import visionzip
# ============= VisionZip ==================


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.layer_list is not None:
        pdrop_infer = True  # whether to use pdrop infer
    else:
        pdrop_infer = False
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
                                                                           args.model_base, 
                                                                           model_name,
                                                                           pdrop_infer
                                                                           )

    # ============= VisionZip ==================
    model = visionzip(model, dominant=args.dominant, contextual=args.contextual, cluster_width=args.cluster_width)
    # ============= VisionZip ==================

    model_class_name = type(model).__name__
    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
        model.model.layer_list = eval(args.layer_list)
        model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
        model.model.image_token_ratio_list.insert(0, 1.0)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            original_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Build input ids (keep a 1D copy for salient index computation)
            input_ids_1d = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                if pdrop_infer and args.compute_salient_tokens:
                    # Compute salient token indices for PyramidDrop
                    salient_tokens = salient_tokens_finder(original_prompt)
                    salient_tokens_filtered = [t.lower() for t in salient_tokens]

                    res = longest_common_subarray(tokenizer(original_prompt).input_ids, input_ids_1d.detach().cpu().tolist())
                    if res is not None:
                        qqs_start = res['start_in_qqs_0based']
                        prompt_start = res['start_in_prompt_0based']
                        if qqs_start > 1:
                            rel_seq = [ele for ele in input_ids_1d[prompt_start - (qqs_start - 1):prompt_start]] + res['match']
                            start = prompt_start - (qqs_start - 1)
                        else:
                            rel_seq = res['match']
                            start = prompt_start
                    else:
                        raise NotImplementedError("No match found")
                    rel_words = [tokenizer.decode([w], skip_special_tokens=True) for w in rel_seq]
                    assert len(rel_seq) == len(rel_words), "Length mismatch"
                    salient_token_original = greedy_align_and_filter(rel_words, salient_tokens_filtered)
                    salient_token_indices = [start + idx for idx, word in enumerate(rel_words) if word in salient_token_original]
                    if not len(salient_token_indices):
                        # Ensure at least one token index is present
                        salient_token_indices.append(len(input_ids_1d) - 1)
                else:
                    salient_token_indices = None

                if pdrop_infer:
                    output_ids = model.generate(
                        input_ids_1d.unsqueeze(0).cuda(),
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        idxs=salient_token_indices,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
                else:
                    output_ids = model.generate(
                        input_ids_1d.unsqueeze(0).cuda(),
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--layer_list", type=str, default= None)
    parser.add_argument("--image_token_ratio_list", type=str, default= None)
    parser.add_argument("--compute_salient_tokens", action='store_true', default=False)
    parser.add_argument("--dominant", type=int, default=300)
    parser.add_argument("--contextual", type=int, default=7)
    parser.add_argument("--cluster_width", type=int, default=4)
    args = parser.parse_args()

    eval_model(args)
