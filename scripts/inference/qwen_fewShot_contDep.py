import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score

LABELS = [
    "Self-sufficient",
    "Context-dependent",
    "Emotion-impossible"
]

# -------------------------
# Prompt builders
# -------------------------
def build_single_example(sentence, emotion, label=None):
    prompt = (
        f"Sentence: {sentence}\n"
        f"Emotion: {emotion}\n"
        f"Label:"
    )
    if label is not None:
        prompt += f" {label}\n\n"
    return prompt

def build_fewshot_prefix(fewshot_df):
    """
    Build a fixed few-shot prefix with 2 examples per class.
    """
    shots = []
    for label in LABELS:
        subset = fewshot_df[fewshot_df["majority_vote"].str.lower() == label.lower()].head(2)
        for _, row in subset.iterrows():
            shots.append(
                build_single_example(
                    row["sentence"],
                    row["emotion"],
                    label
                )
            )
    return "".join(shots)

def build_prompt(sentence, emotion, fewshot_prefix):
    return (
        fewshot_prefix +
        build_single_example(sentence, emotion)
    )

# -------------------------
# Prediction helpers
# -------------------------
def extract_label(text):
    for sep in ["\n", "  "]:
        if sep in text:
            text = text[:text.index(sep)].strip()
    text = text.strip(".").strip()
    if text.lower() in [label.lower() for label in LABELS]:
        return text
    return "Unknown"

def generate_label(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            top_p=1.0,
            top_k=0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    return decoded

# -------------------------
# Evaluation
# -------------------------
def evaluate_predictions(gold_df, pred_df):
    
    # Merge dataframes on 'idx' to align rows
    merged_df = gold_df.merge(
        pred_df[['idx', 'predicted_context_dependency']], 
        on='idx', 
        how='inner'
    )

    # Extract gold and predicted labels
    if 'majority_vote' in merged_df.columns:
        gold_labels = merged_df['majority_vote'].tolist()
    else:
        gold_labels = merged_df['context_dependency'].tolist()
    pred_labels = merged_df['predicted_context_dependency'].tolist()

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

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Few-shot context-dependency emotion classification"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")  # meta-llama/Meta-Llama-3-8B
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--fewshot_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_rows", type=int, default=None)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True
    )
    model.eval()

    # Load few-shot examples
    fewshot_df = pd.read_csv(args.fewshot_csv)    
    fewshot_df = fewshot_df[fewshot_df["majority_vote"].str.lower().isin([label.lower() for label in LABELS])]

    fewshot_prefix = build_fewshot_prefix(fewshot_df)

    # Load inference data
    df = pd.read_csv(args.input_csv)
    if args.max_rows:
        df = df.head(args.max_rows)

    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = build_prompt(
            row["sentence"],
            row["emotion"],
            fewshot_prefix
        )
        raw_output = generate_label(
            model, tokenizer, prompt, args.device
        )
        label = extract_label(raw_output)

        predictions.append({
            "idx": row["idx"],
            "sentence": row["sentence"],
            "emotion": row["emotion"],
            "tagged": row["tagged"],
            "predicted_context_dependency": label
        })

    out_df = pd.DataFrame(predictions)
    out_df.to_csv(args.output_csv, index=False)

    print(f"Saved predictions to {args.output_csv}")

    metrics = evaluate_predictions(df, out_df)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
