import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

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


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, pdrop_infer, compute_salient_tokens=True):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.pdrop_infer = pdrop_infer
        self.compute_salient_tokens = compute_salient_tokens
    
    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        original_prompt = qs

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if image_file.startswith("SEED-Bench-video-image"):
            meta_info = image_file.split("/")[-1].split("_")
            if meta_info[0] == "10":
                sub_dir = "ssv2_8_frame"
            elif meta_info[0] == "11":
                sub_dir = "kitchen_8_frame"
            elif meta_info[0] == "12":
                sub_dir = "breakfast_8_frame"
            else:
                raise ValueError(f"Invalid task number: {meta_info[0]}")
            actual_image_file = "/wekafs/aditysin/PyramidDrop/data/playground/data/eval/seed_bench/SEED-Bench-video-image/" + "v1_video/task{}/{}/{}/4.png".format(meta_info[0], sub_dir, meta_info[1].rstrip('.png'))
            image = Image.open(actual_image_file).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        else:
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        # ================ TODO: MODS ====================
        if self.pdrop_infer and self.compute_salient_tokens:
            # -------------------- NO Reference setting -----------------
            # qs = qs.split("\nReference OCR token:")[0]
            # qs += "\nAnswer the question using a single word or phrase."
            # -----------------------------------------------------------
            salient_tokens_filtered = [t.lower() for t in salient_tokens_finder(original_prompt)]
            # start_seq = [-200, 29871, 13]
            # end_seq   = [319, 1799, 9047, 13566, 29901]
            res = longest_common_subarray(self.tokenizer(original_prompt).input_ids, input_ids.detach().cpu().tolist())
            if res is not None:
                qqs_start = res['start_in_qqs_0based']
                prompt_start = res['start_in_prompt_0based']
                if qqs_start > 1:
                    rel_seq = [ele for ele in input_ids[prompt_start - (qqs_start - 1):prompt_start]] + res['match']
                    start = prompt_start - (qqs_start - 1)
                else:
                    rel_seq = res['match']
                    start = prompt_start
            else:
                raise NotImplementedError("No match found")
            rel_words = [self.tokenizer.decode([w], skip_special_tokens=True) for w in rel_seq]
            assert len(rel_seq) == len(rel_words), "Length mismatch"
            salient_token_original = greedy_align_and_filter(rel_words, salient_tokens_filtered)
            salient_token_indices = [start+id for id, word in enumerate(rel_words) if word in salient_token_original]
            if not len(salient_token_indices):
                print(f"HIT!!! for empty list")
            # +++++++++++++++++ For adding last token +++++++++++++++++ 
            salient_token_indices.append(len(input_ids) - 1)
        # print("Indices and the tokens: ", [(st, rw) for st, rw in zip(salient_token_indices, salient_token_original+[':'])], "\n\n")
        # +++++++++++++++++ +++++++++++++++++++++ +++++++++++++++++
        # print("Indices and the tokens: ", [(st, rw) for st, rw in zip(salient_token_indices, salient_token_original)], "\n\n")
        # ================================================
        else:
            salient_token_indices = None

        return input_ids, image_tensor, image.size, salient_token_indices

    def __len__(self):
        return len(self.questions)


# ================== TODO: MODS ===================
def collate_fn(batch):
    input_ids, image_tensors, image_sizes, salient_tokens = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, salient_tokens
# =================================================

# def collate_fn(batch):
#     input_ids, image_tensors, image_sizes = zip(*batch)
#     input_ids = torch.stack(input_ids, dim=0)
#     image_tensors = torch.stack(image_tensors, dim=0)
#     return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, pdrop_infer, compute_salient_tokens, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, pdrop_infer, compute_salient_tokens)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


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

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    
    # Determine whether to compute salient tokens:
    # Priority: command-line arg > model config > default (False)
    compute_salient_tokens = args.compute_salient_tokens
    if not compute_salient_tokens:
        # Check model config if arg not set
        compute_salient_tokens = getattr(model.config, 'use_salient_tokens', False)
    print(f"Using salient tokens: {compute_salient_tokens}")
    
    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, pdrop_infer, compute_salient_tokens)

    start_time = time.time()
    total_tokens = 0
    for (input_ids, image_tensor, image_sizes, salient_tokens), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        total_tokens += input_ids.shape[1]

        with torch.inference_mode():
            if pdrop_infer:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    idxs=salient_tokens[0],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)
            else:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
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
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer_list", type=str, default= None)
    parser.add_argument("--image_token_ratio_list", type=str, default= None)
    parser.add_argument("--compute_salient_tokens", action='store_true', default=False)
    parser.add_argument("--dominant", type=int, default=300)
    parser.add_argument("--contextual", type=int, default=7)
    parser.add_argument("--cluster_width", type=int, default=4)
    args = parser.parse_args()

    eval_model(args)
