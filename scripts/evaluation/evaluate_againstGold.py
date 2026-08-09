import argparse
# import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

'''
Compares the candidate predictions against the gold standard labels.
'''

def compute_stats(gold_labels, pred_labels):
    cm = confusion_matrix(gold_labels, pred_labels)
    label_counts = np.unique(pred_labels, return_counts=True)
    return cm, label_counts

def evaluate_predictions(gold_df, pred_df, pred_column, gold_column):
    
    # Merge dataframes on 'idx' to align rows
    merged_df = gold_df.merge(
        pred_df[['idx', pred_column]], 
        on='idx', 
        how='inner'
    )

    # Extract gold and predicted labels
    gold_labels = merged_df[gold_column].tolist()
    pred_labels = merged_df[pred_column].tolist()

    gold_labels = [label.lower().strip() for label in gold_labels]
    pred_labels = [label.lower().strip() for label in pred_labels]

    # Compute metrics
    accuracy = accuracy_score(gold_labels, pred_labels)
    f1_weighted = f1_score(gold_labels, pred_labels, average='weighted', zero_division=0)
    f1_macro = f1_score(gold_labels, pred_labels, average='macro', zero_division=0)
    f1_micro = f1_score(gold_labels, pred_labels, average='micro', zero_division=0)
    precision = f1_score(gold_labels, pred_labels, average='weighted', zero_division=0)  # Placeholder for precision
    recall = f1_score(gold_labels, pred_labels, average='weighted', zero_division=0)  # Placeholder for recall
    
    return {
        'accuracy': round(accuracy,2),
        'f1_weighted': round(f1_weighted,2),
        'f1_macro': round(f1_macro,2),
        'f1_micro': round(f1_micro,2),
        'precision': round(precision,2),
        'recall': round(recall,2),
        'matched_samples': len(merged_df)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Emotion-Context-dependency classification"
    )
    parser.add_argument("--gold_csv", required=True)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--pred_column", required=True)
    parser.add_argument("--gold_column", required=True)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    gold_df = pd.read_csv(args.gold_csv)
    candidate_df = pd.read_csv(args.candidate_csv)

    print(f"Using prediction column: {pred_column}")
    print(f"Using gold column: {gold_column}")

    if not args.gold_csv == args.candidate_csv:
        # Evaluate
        metrics = evaluate_predictions(gold_df, candidate_df, pred_column, gold_column)
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
    
    # Compute confusion matrix and label counts
    cm, label_counts = compute_stats(
        gold_df[gold_column].str.lower().str.strip().tolist(),
        candidate_df[pred_column].str.lower().str.strip().tolist())

    print("\nConfusion Matrix:")
    print(cm)
    print("\nLabel Counts:")
    print(label_counts)

if __name__ == "__main__":
    main()
