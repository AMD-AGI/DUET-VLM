import os
import json
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        pred = json.loads(pred)
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['text']

    cur_df.to_excel(os.path.join(args.upload_dir, f"{args.experiment}.xlsx"), index=False, engine='openpyxl')
    # ============= MODS =============
    predictions = cur_df['prediction'].tolist()
    answers = cur_df['answer'].tolist()
    total_score = 0
    correct_score = 0
    for prediction, answer in zip(predictions, answers):
        if prediction == answer:
            correct_score += 1
        total_score += 1
    print(f"Score: {correct_score}/{total_score} = {correct_score/total_score*100:.2f}%")
    # ============= MODS =============