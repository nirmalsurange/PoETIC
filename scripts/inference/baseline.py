import argparse
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import json
from typing import List, Tuple
import pandas as pd
from datetime import datetime


def load_data(filepath):
    """Load CSV file and prepare dataset"""
    df = pd.read_csv(filepath)
    return df

def prepare_dataset(df, tokenizer, poss_to_id):
    """Convert dataframe to HuggingFace Dataset"""
    df['label'] = df['emotion_possiblity'].map(poss_to_id)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    dataset = Dataset.from_pandas(df[['sentence', 'label']])
    dataset = dataset.map(tokenize_function, batched=True)
    return dataset

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train(args, train_df, dev_df, poss_to_id):
    """Train RoBERTa model"""
    print(f"Training mode: Loading data ") 
    
    # Disable NCCL P2P and IB for RTX 4000 series
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    
    train_dataset = prepare_dataset(train_df, tokenizer, poss_to_id)
    dev_dataset = prepare_dataset(dev_df, tokenizer, poss_to_id)
    
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(poss_to_id)
    )
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # eval_strategy='epoch',
        save_strategy='no',  #'epoch',
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    model.save_pretrained(f'./models/{args.model_name.split("/")[-1]}_emotion')
    
    with open('./models/emotion_possibilities_to_id.json', 'w') as f:
        json.dump(poss_to_id, f)
    
    # print("Model training completed and saved!")


def evaluate(args, test_df, id_to_poss, run_id):
    """Evaluate RoBERTa model"""
    # print(f"Evaluation mode: Loading model and test data")
    
    with open('./models/emotion_possibilities_to_id.json', 'r') as f:
        poss_to_id = json.load(f)
    
    # id_to_emotion = {int(idx): emotion for emotion, idx in emotion_to_id.items()}
    
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        f'./models/{args.model_name.split("/")[-1]}_emotion'
    )
    
    test_dataset = prepare_dataset(test_df, tokenizer, poss_to_id)
    
    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    
    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(axis=1)
    true_labels = predictions.label_ids
    
    fout = open(f'./results/test_evaluation_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt', 'a')
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    fout.write(f"\n\nRun {run_id} Evaluation Metrics:\n")
    fout.write(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
    fout.write(f"Precision: {precision:.4f}")
    fout.write(f"Recall: {recall:.4f}")
    fout.write(f"F1-Score: {f1:.4f}")
    
    fout.write(f"\nClassification Report:")
    fout.write(classification_report(true_labels, pred_labels, target_names=[id_to_poss[i] for i in range(len(id_to_poss))]))
    fout.close()


def prepare_format(df):
    """Prepare dataframe for submission format"""
    df['id'] = df.index + 1
    df = df[['id', 'idx', 'sentence', 'emotion', 'Corrected']].rename(columns={'Corrected': 'emotion_possiblity'})
    df['sent_id'] = df.groupby('sentence', sort=False).ngroup() + 1
    return df


def split_input(df):
    """Split input dataframe into 10 sets"""
    sentence_groups = df.groupby('sentence', sort=False).groups
    sentences = list(sentence_groups.keys())
    n_groups = 10
    group_size = len(sentences) // n_groups

    sent_lists = [sentences[i:i + group_size] for i in range(0, len(sentences), group_size)]
    # print(f"Total sentences: {len(sentences)}, Group size: {group_size}, Number of groups: {len(sent_lists)}")
    # print('sentence group size in list:', [len(g) for g in sent_lists])
    
    df_parts = []
    for i, sent_list in enumerate(sent_lists):
        part_df = df[df['sentence'].isin(sent_list)].copy()
        part_df['part'] = i + 1
        df_parts.append(part_df)
        # print(len(part_df))
    return df_parts


def circular_splits(dfs: List[pd.DataFrame], n_runs: int = 7):
    """
    Generate circular train/dev/test splits.

    Args:
        dfs (List[pd.DataFrame]): List of 10 DataFrames.
        n_runs (int): Number of runs (default = 7).

    Yields:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            train_df (8 combined), dev_df (1), test_df (1)
    """
    n = len(dfs)
    assert n == 10, "This function expects exactly 10 DataFrames"
    assert n_runs <= n, "n_runs cannot exceed number of unique test sets"

    for i in range(n_runs):
        # Test index rotates and is never repeated
        test_idx = i % n

        # Dev index is next one in circular order
        dev_idx = (i + 1) % n

        # Remaining indices for training
        train_indices = [
            idx for idx in range(n)
            if idx not in (test_idx, dev_idx)
        ]

        # Combine training dataframes
        train_df = pd.concat([dfs[idx] for idx in train_indices], ignore_index=True)
        dev_df = dfs[dev_idx]
        test_df = dfs[test_idx]

        yield train_df, dev_df, test_df


def main():
    parser = argparse.ArgumentParser(description="RoBERTa Emotion Classification Baseline")
    parser.add_argument('--model_name', type=str, default='roberta-base', help='Model name from HuggingFace')
    parser.add_argument('--N', required=True, type=int, default=5, help='Number of times to perform cross-validation')
    parser.add_argument('--input', required=True, type=str, help='Path to input CSV file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')

    args = parser.parse_args()
    
    input_df = load_data(args.input)
    input_df = prepare_format(input_df)
    # print(input_df.head(20))

    # Get unique emotion possibilities classes and create mapping
    emotion_possibilities = input_df['emotion_possiblity'].unique()
    poss_to_id = {poss: idx for idx, poss in enumerate(emotion_possibilities)}
    id_to_poss = {idx: poss for poss, idx in poss_to_id.items()}
    
    # print(f"Emotion Possibilities: {emotion_possibilities}")

    # # perform cross-validation
    # print('Splitting data')
    df_parts = split_input(input_df)

    for run_id, (train_df, dev_df, test_df) in enumerate(circular_splits(df_parts, n_runs=args.N)):
        print(f"Run {run_id+1}")
        # print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}, Test size: {len(test_df)}")

        train(args, train_df, dev_df, poss_to_id)


        evaluate(args, test_df, id_to_poss, run_id+1)

if __name__ == '__main__':
    main()