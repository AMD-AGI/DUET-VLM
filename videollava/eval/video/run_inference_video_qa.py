import math
import os
import argparse
import json

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama_pdrop import LlavaLlamaForCausalLM_PDrop
from videollava.train.train import smart_tokenizer_and_embedding_resize
from visionzip import visionzip_video as visionzip


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--dominant", type=int, default=160)
    parser.add_argument("--contextual", type=int, default=32)
    parser.add_argument("--layer_list", type=str, required=True)
    parser.add_argument("--image_token_ratio_list", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_VID_START_TOKEN + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_VID_END_TOKEN + '\n' + qs
    else:
        qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs

    conv_mode = "llava_v1"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # print(video_tensor.shape)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    # print(f"Input ids in get_model_output: {input_ids}")
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    # print(f"Keywords: {keywords}")
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    # print(f"Stopping criteria: {stopping_criteria}")
    # Reduce generation memory footprint to avoid OOM
    # gen_cap = max(64, min(256, args.model_max_length // 2))
    # try:
    #     torch.cuda.empty_cache()
    # except Exception:
    #     pass
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    # print(f"Output ids in get_model_output: {output_ids}")
    # input_token_len = input_ids.shape[1]
    # print(f"Input ids: {input_ids.shape}")
    # print(f"Output ids: {output_ids.shape}")
    # print(f"Input token len: {input_token_len}")
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # print(f"N diff input output: {n_diff_input_output}")
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # print(tokenizer.decode(output_ids[0][0]))
    # print(tokenizer.decode(output_ids[0][-1]))
    # print(outputs)
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, pdrop_infer=True)
    model = model.to(args.device)
    print("Applying VisionZip to the model")
    model = visionzip(model, dominant=args.dominant, contextual=args.contextual)
    print("VisionZip applied to the model")

    model_class_name = type(model).__name__
    if model_class_name == "LlavaLlamaForCausalLM_PDrop":
        model.model.layer_list = eval(args.layer_list)
        model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
        model.model.image_token_ratio_list.insert(0, 1.0)

    # Load both ground truth file containing questions and answers
    # with open(args.gt_file_question) as file:
    #     gt_questions = json.load(file)
    # with open(args.gt_file_answers) as file:
    #     gt_answers = json.load(file)

    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    gt_answers = json.load(open(args.gt_file_answers, "r"))
    answers_by_id = {sample["question_id"]: sample["answer"] for sample in gt_answers}

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results


    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    for sample in tqdm(gt_questions):
        video_name = sample['video_name']
        print(f"Video name: {video_name}")
        question = sample['question']
        print(f"Question: {question}")
        question_id = sample['question_id']
        print(f"Question ID: {question_id}")
        answer = answers_by_id.get(question_id)
        if answer is None:
            print(f"Warning: No answer found for question id {question_id}")
            continue
        print(f"Answer: {answer}")

        sample_set = {'id': question_id, 'question': question, 'answer': answer}
        print(f"Sample set: {sample_set}")

        # Load the video file
        for fmt in tqdm(video_formats):  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = get_model_output(model, processor['video'], tokenizer, video_path, question, args)
                sample_set['pred'] = output
                print(f"Sample set with prediction: {sample_set}")
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                break

    ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
