#!/usr/bin/env python3
"""
Multi-benchmark evaluation for Qwen2.5-VL DUET
Supports: POPE, GQA, ScienceQA, MME
"""

import os
import sys
import json
import time
import argparse
import re
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modeling_qwen2_5vl_duet import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor


# ============================================================================
# Dataset Classes
# ============================================================================

class POPEDataset(Dataset):
    """POPE (Polling-based Object Probing Evaluation) dataset"""
    def __init__(self, question_file, image_folder):
        self.image_folder = image_folder
        self.questions = []
        with open(question_file, 'r') as f:
            for line in f:
                self.questions.append(json.loads(line))
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        image_path = os.path.join(self.image_folder, q['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None
        return {
            'question_id': q['question_id'],
            'image': image,
            'text': q['text'],
            'label': q.get('label', None),
            'category': q.get('category', 'unknown'),
        }


class GQADataset(Dataset):
    """GQA dataset"""
    def __init__(self, question_file, image_folder):
        self.image_folder = image_folder
        self.questions = []
        with open(question_file, 'r') as f:
            for line in f:
                self.questions.append(json.loads(line))
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        image_path = os.path.join(self.image_folder, q['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None
        return {
            'question_id': q['question_id'],
            'image': image,
            'text': q['text'],
        }


class SQADataset(Dataset):
    """ScienceQA dataset - LLaVA conversation format"""
    
    # Prompt suffix to force direct answer (reduces parse failures from 43% to ~5%)
    ANSWER_SUFFIX = "\nAnswer with the option's letter from the given choices directly."
    
    def __init__(self, question_file, image_folder, add_answer_suffix=True):
        self.image_folder = image_folder
        self.add_answer_suffix = add_answer_suffix
        with open(question_file, 'r') as f:
            data = json.load(f)
        
        # LLaVA format: items with 'conversations' and optional 'image'
        self.questions = []
        for item in data:
            if item.get('image'):  # Only include questions with images
                self.questions.append(item)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        image_path = os.path.join(self.image_folder, q['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None
        
        # Extract question from conversations (remove <image> tag)
        human_msg = q['conversations'][0]['value']
        text = human_msg.replace('<image>\n', '').replace('<image>', '').strip()
        
        # Add prompt suffix to force direct answer
        if self.add_answer_suffix:
            text = text + self.ANSWER_SUFFIX
        
        # Get ground truth answer
        gt_answer = q['conversations'][1]['value'] if len(q['conversations']) > 1 else None
        
        return {
            'question_id': q.get('id', idx),
            'image': image,
            'text': text,
            'answer': gt_answer,
        }


class MMEDataset(Dataset):
    """MME dataset"""
    def __init__(self, question_file, image_folder):
        self.image_folder = image_folder
        self.questions = []
        with open(question_file, 'r') as f:
            for line in f:
                self.questions.append(json.loads(line))
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        image_path = os.path.join(self.image_folder, q['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None
        return {
            'question_id': q['question_id'],
            'image': image,
            'text': q['text'],
            'category': q.get('category', 'unknown'),
        }


class TextVQADataset(Dataset):
    """TextVQA dataset with OCR tokens"""
    def __init__(self, question_file, image_folder):
        self.image_folder = image_folder
        self.questions = []
        with open(question_file, 'r') as f:
            for line in f:
                self.questions.append(json.loads(line))
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        q = self.questions[idx]
        image_path = os.path.join(self.image_folder, q['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return None
        return {
            'question_id': q['question_id'],
            'image': image,
            'text': q['text'],
            'prompt': q['text'],  # Keep original for evaluation
        }


def collate_fn(batch):
    return [b for b in batch if b is not None]


# ============================================================================
# Evaluators
# ============================================================================

def evaluate_pope(results, questions, annotation_dir=None):
    """Evaluate POPE accuracy (yes/no classification)
    
    Args:
        results: List of prediction results
        questions: List of questions (used for category mapping)
        annotation_dir: Directory containing coco_pope_*.json files with labels
    """
    import os
    
    # Build question lookup for category
    question_lookup = {q['question_id']: q for q in questions}
    
    # Build result lookup
    result_lookup = {r['question_id']: r['text'] for r in results}
    
    # If annotation_dir provided, load labels from there
    if annotation_dir and os.path.isdir(annotation_dir):
        all_results = {'overall': {'correct': 0, 'total': 0}}
        
        for file in os.listdir(annotation_dir):
            if file.startswith('coco_pope_') and file.endswith('.json'):
                category = file[10:-5]  # Extract category name
                label_file = os.path.join(annotation_dir, file)
                labels = [json.loads(line) for line in open(label_file, 'r')]
                label_lookup = {l['question_id']: l['label'].lower() for l in labels}
                
                correct = 0
                total = 0
                
                for qid, label in label_lookup.items():
                    if qid not in result_lookup:
                        continue
                    
                    pred_text = result_lookup[qid]
                    # Extract yes/no from prediction (following VisionZip_HK logic)
                    text = pred_text
                    if '.' in text:
                        text = text.split('.')[0]
                    text = text.replace(',', '')
                    words = text.split(' ')
                    
                    if 'No' in words or 'not' in words or 'no' in words:
                        pred_ans = 'no'
                    else:
                        pred_ans = 'yes'
                    
                    if pred_ans == label:
                        correct += 1
                    total += 1
                
                if total > 0:
                    acc = correct / total * 100
                    print(f"  {category}: {acc:.2f}% ({correct}/{total})")
                    all_results[category] = {'correct': correct, 'total': total, 'accuracy': acc}
                    all_results['overall']['correct'] += correct
                    all_results['overall']['total'] += total
        
        overall = all_results['overall']
        if overall['total'] > 0:
            return overall['correct'] / overall['total'] * 100
        return 0
    
    # Fallback: try to get labels from questions directly
    correct = 0
    total = 0
    
    for q in questions:
        qid = q['question_id']
        if qid not in result_lookup:
            continue
        
        label = q.get('label', '').lower()
        if not label:
            continue
        
        pred_text = result_lookup[qid]
        # Extract yes/no from prediction
        text = pred_text
        if '.' in text:
            text = text.split('.')[0]
        text = text.replace(',', '')
        words = text.split(' ')
        
        if 'No' in words or 'not' in words or 'no' in words:
            pred_ans = 'no'
        else:
            pred_ans = 'yes'
        
        if pred_ans == label:
            correct += 1
        total += 1
    
    return correct / total * 100 if total > 0 else 0


def evaluate_gqa(results):
    """GQA evaluation - return results for external eval script"""
    # GQA uses official eval script, just return formatted results
    return {r['question_id']: r['text'] for r in results}


def extract_sqa_answer(pred_text, options=["A", "B", "C", "D", "E", "F", "G", "H"]):
    """Extract answer letter from SQA prediction using multiple patterns.
    
    Following VisionZip_HK/llava/eval/eval_science_qa.py logic:
    1. Check if first character is a valid option (handles "C\n\ngarbage..." case)
    2. Check if pred_text is directly an option (A, B, C, D, E)
    3. Check if format is "A. " (answer followed by ". ")
    4. Use regex to match "The answer is ([A-Z])."
    5. Look for "correct answer is X" pattern
    """
    import re
    
    pred_text = pred_text.strip()
    
    # 0. Check if first character is a valid option (handles "C\n\ngarbage..." case)
    if len(pred_text) >= 1 and pred_text[0].upper() in options:
        # Check if followed by non-letter (newline, space, period, etc.)
        if len(pred_text) == 1 or not pred_text[1].isalpha():
            return pred_text[0].upper()
    
    # 1. Direct option match
    if pred_text.upper() in options:
        return pred_text.upper()
    
    # 2. Format "A. " at the start
    if len(pred_text) >= 3 and pred_text[0].upper() in options and pred_text[1:3] == ". ":
        return pred_text[0].upper()
    
    # 3. "The answer is X." pattern
    pattern1 = re.compile(r'[Tt]he answer is ([A-H])')
    res = pattern1.findall(pred_text)
    if len(res) >= 1:
        return res[-1].upper()  # Take last match
    
    # 4. "correct answer is X" or "correct answer is: X" pattern
    pattern2 = re.compile(r'correct answer is:?\s*([A-H])', re.IGNORECASE)
    res = pattern2.findall(pred_text)
    if len(res) >= 1:
        return res[-1].upper()
    
    # 5. Look for standalone letter followed by period/parenthesis at end
    pattern3 = re.compile(r'\b([A-H])[.\)]\s*$')
    res = pattern3.findall(pred_text)
    if len(res) >= 1:
        return res[-1].upper()
    
    # 6. Look for "X." or "(X)" pattern anywhere
    pattern4 = re.compile(r'(?:^|\s)([A-H])[\.\)]')
    res = pattern4.findall(pred_text)
    if len(res) >= 1:
        return res[-1].upper()
    
    return "FAILED"


def evaluate_sqa(results, questions):
    """Evaluate ScienceQA accuracy with improved answer extraction"""
    correct = 0
    total = 0
    failed = 0
    
    result_lookup = {str(r['question_id']): r['text'] for r in results}
    question_lookup = {str(q.get('id', i)): q for i, q in enumerate(questions)}
    
    for qid, pred_text in result_lookup.items():
        if qid not in question_lookup:
            continue
        
        q = question_lookup[qid]
        # Get GT from conversations
        gt_answer = q['conversations'][1]['value'] if len(q.get('conversations', [])) > 1 else None
        if gt_answer is None:
            continue
        
        # Extract answer using improved logic
        pred_letter = extract_sqa_answer(pred_text)
        
        if pred_letter == "FAILED":
            failed += 1
        
        if pred_letter == gt_answer.strip().upper():
            correct += 1
        total += 1
    
    if failed > 0:
        print(f"  Warning: {failed}/{total} predictions failed to parse")
    
    return correct / total * 100 if total > 0 else 0


def evaluate_mme(results):
    """MME evaluation - return results for external eval script"""
    return results


def evaluate_textvqa(results, annotation_file):
    """TextVQA evaluation using VQA accuracy metric.
    
    Uses the same evaluation method as LLaVA-1.5 (m4c_evaluator logic).
    """
    import re
    
    # Load annotations
    with open(annotation_file) as f:
        annotations_data = json.load(f)['data']
    
    # Build annotation lookup by (image_id, question)
    def normalize_question(text):
        """Extract and normalize question from prompt."""
        if text.startswith('OCR tokens: '):
            pattern = r"Question: (.*?) Short answer:"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).lower().strip()
        elif 'Reference OCR token: ' in text:
            lines = text.split('\n')
            if text.startswith('Reference OCR token:'):
                return lines[1].lower().strip() if len(lines) > 1 else text.lower()
            return lines[0].lower().strip()
        elif len(text.split('\n')) == 2:
            return text.split('\n')[0].lower().strip()
        return text.lower().strip()
    
    annotations = {}
    for ann in annotations_data:
        key = (ann['image_id'], ann['question'].lower().strip())
        annotations[key] = ann['answers']
    
    # TextVQA accuracy: prediction matches any of the GT answers
    def vqa_accuracy(pred, gt_answers):
        """Compute VQA accuracy for a single prediction."""
        pred = pred.lower().strip()
        # Remove punctuation for comparison
        pred_clean = re.sub(r'[^\w\s]', '', pred)
        
        match_count = 0
        for gt in gt_answers:
            gt_clean = re.sub(r'[^\w\s]', '', gt.lower().strip())
            if pred_clean == gt_clean:
                match_count += 1
        
        # VQA accuracy formula: min(matches/3, 1)
        return min(match_count / 3.0, 1.0)
    
    total_acc = 0
    count = 0
    missing = 0
    
    for r in results:
        qid = r['question_id']
        pred = r['text']
        prompt = r.get('prompt', '')
        
        # Try to match with annotations
        question = normalize_question(prompt)
        key = (qid, question)
        
        if key not in annotations:
            # Try without normalization
            for (img_id, q), answers in annotations.items():
                if img_id == qid:
                    key = (img_id, q)
                    break
        
        if key in annotations:
            acc = vqa_accuracy(pred, annotations[key])
            total_acc += acc
            count += 1
        else:
            missing += 1
    
    if missing > 0:
        print(f"  Warning: {missing} questions not found in annotations")
    
    return (total_acc / count * 100) if count > 0 else 0


# ============================================================================
# Main Evaluation
# ============================================================================

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model_path}")
    
    processor_kwargs = {}
    if args.min_pixels is not None:
        processor_kwargs['min_pixels'] = args.min_pixels * 28 * 28
    if args.max_pixels is not None:
        processor_kwargs['max_pixels'] = args.max_pixels * 28 * 28
    
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    
    # Configure DUET
    if args.mode == "baseline":
        model.configure_duet(visionzip_enabled=False, pdrop_enabled=False)
        config_str = "BASELINE"
    elif args.mode == "ori_visionzip":
        model.configure_duet(
            visionzip_enabled=True,
            use_ori_visionzip=True,
            dominant_tokens=args.dominant,
            contextual_tokens=args.contextual,
            pdrop_enabled=False,
        )
        config_str = f"VisionZip_orig (d{args.dominant}c{args.contextual})"
    else:  # duet
        model.configure_duet(
            visionzip_enabled=True,
            use_ori_visionzip=False,
            dominant_tokens=args.dominant,
            contextual_tokens=args.contextual,
            cluster_width=args.cluster_width,
            pdrop_enabled=True,
            layer_list=args.layer_list,
            image_token_ratio_list=args.ratio_list,
        )
        config_str = f"DUET (d{args.dominant}c{args.contextual}, pdrop={args.ratio_list})"
    
    print(f"Configuration: {config_str}")
    
    # Create dataset
    if args.benchmark == "pope":
        dataset = POPEDataset(args.question_file, args.image_folder)
    elif args.benchmark == "gqa":
        dataset = GQADataset(args.question_file, args.image_folder)
    elif args.benchmark == "sqa":
        dataset = SQADataset(args.question_file, args.image_folder)
    elif args.benchmark == "mme":
        dataset = MMEDataset(args.question_file, args.image_folder)
    elif args.benchmark == "textvqa":
        dataset = TextVQADataset(args.question_file, args.image_folder)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    
    if args.num_samples:
        dataset.questions = dataset.questions[:args.num_samples]
    
    # Apply chunking for multi-GPU parallel evaluation
    if args.num_chunks > 1:
        total = len(dataset.questions)
        chunk_size = (total + args.num_chunks - 1) // args.num_chunks  # ceiling division
        start_idx = args.chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        dataset.questions = dataset.questions[start_idx:end_idx]
        print(f"Chunk {args.chunk_idx}/{args.num_chunks}: samples {start_idx}-{end_idx} of {total}")
    
    print(f"Evaluating {len(dataset)} samples on {args.benchmark}")
    
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Warmup
    if args.warmup > 0:
        print(f"Warmup...")
        warmup_iter = iter(dataloader)
        for _ in range(min(args.warmup, len(dataset))):
            try:
                batch = next(warmup_iter)
                if not batch:
                    continue
                item = batch[0]
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': item['image']},
                    {'type': 'text', 'text': item['text']}
                ]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=[item['image']], return_tensors='pt').to(device)
                with torch.inference_mode():
                    _ = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            except StopIteration:
                break
        print("Warmup done")
    
    # Run inference
    results = []
    total_time = 0
    
    for batch in tqdm(dataloader, desc=f"Evaluating {args.benchmark}"):
        if not batch:
            continue
        item = batch[0]
        
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': item['image']},
            {'type': 'text', 'text': item['text']}
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[item['image']], return_tensors='pt').to(device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        
        torch.cuda.synchronize()
        total_time += time.time() - start
        
        response = processor.decode(output_ids[0], skip_special_tokens=True)
        if 'assistant' in response:
            response = response.split('assistant')[-1].strip()
        
        results.append({
            'question_id': item['question_id'],
            'text': response,
            'prompt': item['text'],
        })
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"\nSaved {len(results)} results to {args.output_file}")
    print(f"Total time: {total_time:.1f}s, Avg: {total_time/len(results)*1000:.0f}ms/sample")
    
    # Evaluate
    if args.benchmark == "pope":
        # Try to find annotation dir automatically if not provided
        annotation_dir = args.annotation_dir
        if not annotation_dir:
            # Default: look for coco/ subdirectory next to question file
            question_dir = os.path.dirname(args.question_file)
            default_annot = os.path.join(question_dir, "coco")
            if os.path.isdir(default_annot):
                annotation_dir = default_annot
                print(f"Using annotation directory: {annotation_dir}")
        
        acc = evaluate_pope(results, dataset.questions, annotation_dir)
        print(f"\n{args.benchmark.upper()} Accuracy: {acc:.2f}%")
    elif args.benchmark == "sqa":
        acc = evaluate_sqa(results, dataset.questions)
        print(f"\n{args.benchmark.upper()} Accuracy: {acc:.2f}%")
    elif args.benchmark == "textvqa":
        # TextVQA evaluation requires annotation file
        annotation_file = args.annotation_dir  # Reuse annotation_dir for annotation file path
        if annotation_file and os.path.isfile(annotation_file):
            acc = evaluate_textvqa(results, annotation_file)
            print(f"\n{args.benchmark.upper()} Accuracy: {acc:.2f}%")
        else:
            print(f"\nTextVQA: Provide --annotation-dir pointing to TextVQA_0.5.1_val.json")
    elif args.benchmark in ["gqa", "mme"]:
        print(f"\n{args.benchmark.upper()}: Use official eval script")
    
    return args.output_file


def main():
    parser = argparse.ArgumentParser(description="Multi-benchmark evaluation for Qwen2.5-VL DUET")
    
    # Model
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--benchmark", type=str, required=True, 
                        choices=["pope", "gqa", "sqa", "mme", "textvqa"])
    
    # Data
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--annotation-dir", type=str, default=None,
                        help="Directory containing annotation files (e.g., coco_pope_*.json for POPE)")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    
    # Image processing
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=2880)
    
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=32)
    
    # Mode
    parser.add_argument("--mode", type=str, choices=["baseline", "ori_visionzip", "duet"], 
                        default="baseline")
    
    # VisionZip config
    parser.add_argument("--dominant", type=int, default=540)
    parser.add_argument("--contextual", type=int, default=100)
    parser.add_argument("--cluster-width", type=int, default=4)
    
    # PyramidDrop config
    parser.add_argument("--layer-list", type=int, nargs='+', default=[14, 21])
    parser.add_argument("--ratio-list", type=float, nargs='+', default=[0.5, 0.25])
    
    # Chunking for multi-GPU parallel evaluation
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks to split data into")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Which chunk to process (0-indexed)")
    
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
