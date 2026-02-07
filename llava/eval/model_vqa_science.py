import argparse
import torch
import os
import json
import time
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# ============= For PDrop Inference ==================
from llava.aligner import greedy_align_and_filter
from llava.utils import longest_common_subarray
from llava.salient_token_finder import salient_tokens_finder
# ============= For PDrop Inference ==================

# ============= VisionZip ==================
from visionzip import visionzip
# ============= VisionZip ==================

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.layer_list is not None:
        pdrop_infer = True  # whether to use pdrop infer
    else:
        pdrop_infer = False
    start_time = time.time()
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, 
                                                                           args.model_base, 
                                                                           model_name,
                                                                           pdrop_infer)
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds\n\n")
    
    # ============= VisionZip ==================
    model = visionzip(model, dominant=args.dominant, contextual=args.contextual, cluster_width=args.cluster_width)
    # ============= VisionZip ==================

    model_class_name = type(model).__name__
    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
        print(f"\nApplying PDrop inference to the model\n")
        model.model.layer_list = eval(args.layer_list)
        model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
        model.model.image_token_ratio_list.insert(0, 1.0)
    else:
        print(f"\nNo PDrop inference is needed for this model\n")

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    # record the start time
    start_time = time.time()
    total_tokens = 0
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        original_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        total_tokens += input_ids.shape[1]

        # Determine whether to compute salient tokens:
        # Priority: command-line arg > model config > default (False)
        compute_salient_tokens = args.compute_salient_tokens
        if not compute_salient_tokens:
            compute_salient_tokens = getattr(model.config, 'use_salient_tokens', False)

        # ============= For PDrop Inference ==================
        if 'image' in line and pdrop_infer and compute_salient_tokens:
            salient_tokens_filtered = [t.lower() for t in salient_tokens_finder(original_prompt)]
            res = longest_common_subarray(tokenizer(original_prompt).input_ids, input_ids[0].detach().cpu().tolist())
            if res is not None:
                qqs_start = res['start_in_qqs_0based']
                prompt_start = res['start_in_prompt_0based']
                # qqs_end = res['end_in_qqs_0based']
                # prompt_end = res['end_in_prompt_0based']
                if qqs_start > 1:
                    rel_seq = [ele for ele in input_ids[0][prompt_start - (qqs_start - 1):prompt_start]] + res['match']
                    start = prompt_start - (qqs_start - 1)
                else:
                    rel_seq = res['match']
                    start = prompt_start
            else:
                raise NotImplementedError("No match found")
            # start_seq = [-200, 29871, 13]
            # if -200 in input_ids[0]:
            #     start_seq = [-200, 29871, 13]
            # else:
            #     start_seq = [3148, 1001, 29901]
            # end_seq   = [319, 1799, 9047, 13566, 29901]
            # rel_seq, start, _ = extract_between_sequences(input_ids[0], start_seq, end_seq)
            rel_words = [tokenizer.decode([w], skip_special_tokens=True) for w in rel_seq]
            assert len(rel_seq) == len(rel_words), "Length mismatch"
            salient_token_original = greedy_align_and_filter(rel_words, salient_tokens_filtered)
            salient_token_indices = [start+id for id, word in enumerate(rel_words) if word in salient_token_original]
            if not len(salient_token_indices):
                print(f"HIT!!! for empty list")
            # +++++++++++++++++ For adding last token +++++++++++++++++ 
            salient_token_indices.append(len(input_ids[0]) - 1)
            # print("Indices and the tokens: ", [(st, rw) for st, rw in zip(salient_token_indices, salient_token_original+[':'])], "\n\n")
            # +++++++++++++++++ +++++++++++++++++++++ +++++++++++++++++
            # print("Indices and the tokens: ", [(st, rw) for st, rw in zip(salient_token_indices, salient_token_original)], "\n\n")
        else:
            salient_token_indices = None
        # ============= For PDrop Inference ==================
        

        with torch.inference_mode():
            if pdrop_infer:
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    idxs=salient_token_indices,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            else:
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()
    end_time = time.time()
    print(f"Questions answered per GPU per second: {len(questions) / (end_time - start_time)}")
    print(f"Tokens processed per GPU per second: {total_tokens / (end_time - start_time)}")
    print(f"Time taken to evaluate model: {end_time - start_time} seconds")
    print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--layer_list", type=str, default= None)
    parser.add_argument("--image_token_ratio_list", type=str, default= None)
    parser.add_argument("--compute_salient_tokens", action='store_true', default=False)
    parser.add_argument("--dominant", type=int, default=300)
    parser.add_argument("--contextual", type=int, default=7)
    parser.add_argument("--cluster_width", type=int, default=4)
    args = parser.parse_args()

    eval_model(args)
