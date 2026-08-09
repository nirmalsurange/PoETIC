import os
import json
import time
import math
import argparse
import pandas as pd
from tqdm import tqdm
from google import genai
from google.genai import types
import logging



# =========================
PROMPT = """
    You are an expert NLP evaluator. You must classify how dependent the text is on surrounding context to express the given emotion.

    Allowed labels:
    Self-Sufficient
    Context-Dependent
    Emotion-Impossible

    Label meanings:
    Self-Sufficient: The emotion is clearly expressed without needing additional context.
    Context-Dependent: The emotion is ambiguous or barely present, and typically needs outside context.
    Emotion-Impossible: Even with added context, this text cannot plausibly express the target emotion.

    Special rule for NEUTRAL:
    - Since "NEUTRAL" means complete absence of the 6 target emotions {{ANGER, SADNESS, JOY, FEAR, DISGUST, SURPRISE}},
    a sentence cannot be "Self-Sufficient" for both NEUTRAL and another emotion.

    STRICT OUTPUT REQUIREMENTS:
    - You must output EXACTLY ONE label
    - Do NOT output any text before or after the label.
    - Do NOT explain your choice.
    - If unsure, pick the best label; do NOT invent new labels.

    Now evaluate:

    Text: "{sentence}"
    Emotion: "{emotion}"
""".strip()


# =========================
# HELPERS
# =========================
def normalize_label(text):
    text = text.strip()
    for label in VALID_LABELS:
        if label.lower() in text.lower():
            return label
    return "INVALID"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# =========================
# CREATE JSONL INPUT
# =========================
def create_batch_jsonl(df, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            prompt = PROMPT.format(
                sentence=row["sentence"],
                emotion=row["emotion"]
            )

            record = {
                "key": f"req-{row['idx']}",
                "request": {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt}
                            ]
                        }
                    ],
                    "generation_config": GENERATION_CONFIG
                }
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# RUN ONE BATCH
# =========================

def download_output_file(file_name, local_path):
    content = client.files.download(file=file_name)
    with open(local_path, "wb") as f:
        f.write(content)
    logger.info(f"Downloaded output file to: {local_path}")

def run_batch(batch_df, batch_id, output_dir):
    input_dir = os.path.join(output_dir, "input_jsonl")
    batch_dir = os.path.join(output_dir, f"batch_outputs")
    ensure_dir(input_dir)
    ensure_dir(batch_dir)

    input_jsonl = os.path.join(input_dir, f"batch_{batch_id:04d}_input.jsonl")
    output_jsonl = os.path.join(batch_dir, f"batch_{batch_id:04d}_output.jsonl")
    output_csv = os.path.join(batch_dir, f"batch_{batch_id:04d}_output.csv")

    if os.path.exists(output_csv):
        logger.info(f"Batch {batch_id} already completed. Skipping.")
        return str(output_csv)

    create_batch_jsonl(batch_df, input_jsonl)

    # Upload the file to the File API
    uploaded_file = client.files.upload(
        file=input_jsonl,
        config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
    )
    logger.info(f"Uploaded file: {uploaded_file.name}")

    batch_job = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            batch_job = client.batches.create(
                model=MODEL_NAME,
                src=uploaded_file.name,
            )
            logger.info(f"Started batch job: {batch_job.name}, attempt {attempt + 1}")
            logger.info(f"Batch {batch_id} | state={batch_job.state}")

            pending_start = time.time()
            submit_count = 1
            while True:
                batch_job = client.batches.get(name=batch_job.name)

                if batch_job.state != "JOB_STATE_PENDING":
                    break

                if time.time() - pending_start > MAX_PENDING:
                    print(f"Batch {batch_id} stuck in PENDING, cancelling.")
                    client.batches.cancel(name=batch_job.name)
                    # return "PENDING_TIMEOUT"
                    if submit_count <= 2:
                        time.sleep(10*60) # wait before retrying
                    
                        batch_job = client.batches.create(
                            model=MODEL_NAME,
                            src=uploaded_file.name,
                        )
                        logger.info(f"Restarted batch job: {batch_job.name}, attempt {attempt + 1}")
                        pending_start = time.time()
                        submit_count += 1
                    else:
                        raise Exception("Max PENDING retries reached.")
                time.sleep(25)

            while batch_job.state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED"):
                time.sleep(25)
                logger.info(f"Batch {batch_id} | state={batch_job.state}")
                batch_job = client.batches.get(name=batch_job.name)
                
            if batch_job.state == "JOB_STATE_SUCCEEDED":
                break
        except Exception:
            if attempt == MAX_RETRIES:
                batch_job = None
    
    # =========================
    # PARSE OUTPUT
    # =========================
    predictions = {}

    if batch_job and batch_job.dest.file_name:
        logger.info(f"✔ Batch {batch_id} completed successfully.")
        # download output file
        download_output_file(batch_job.dest.file_name, output_jsonl)
        # Read output JSONL
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj["key"].replace("req-", "")
                if 'candidates' not in obj["response"]:
                    continue
                elif 'content' not in obj["response"]["candidates"][0]:
                    continue
                text = obj["response"]["candidates"][0]["content"]["parts"][0]["text"]
                predictions[cid] = normalize_label(text)
    else:
        logger.info(f"✘ Batch {batch_id} failed after {MAX_RETRIES} retries.")
        exit()

    # Build final CSV    
    final_rows = []
    for _, row in batch_df.iterrows():
        cid = str(row["idx"])
        label = predictions.get(cid, "ERROR")
        final_rows.append({
            **row.to_dict(),
            "context_dependency": label
        })

    pd.DataFrame(final_rows).to_csv(output_csv, index=False)
    return str(output_csv)

# =========================
# MAIN
# =========================
def main(args):
    ensure_dir(args.output_dir)
    df = pd.read_csv(args.input_csv)

    total_batches = math.ceil(len(df) / args.batch_size)
    files = []

    for i in tqdm(range(total_batches), desc="Processing batches"):
        start = i * args.batch_size
        end = start + args.batch_size
        batch_df = df.iloc[start:end]
        filename = run_batch(batch_df, i + 1, args.output_dir)
        files.append(filename)

    logger.info("✔ All batches completed.")

    # Combine all batch CSVs
    df = pd.concat(pd.read_csv(f) for f in files)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"✔ Final output saved to: {args.output_csv}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Batch context-dependency classification using Gemini Batch API"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store batch-wise inputs and outputs (default: batches)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of rows per batch (default: 1000)"
    )

    args = parser.parse_args()

    # =========================
    # DEFAULT CONFIG
    # =========================
    MODEL_NAME = args.model_name
    MAX_RETRIES = 2
    MAX_PENDING = 15 * 60  # 15 minutes

    VALID_LABELS = {
        "Self-sufficient",
        "Context-dependent",
        "Emotion-Impossible"
    }

    GENERATION_CONFIG = {
        "temperature": 0.0,
        "top_p": 1.0,
    }
    # =========================
    # CLIENT
    # =========================
    GEMINI_API_KEY = 'YOUR_API_KEY'  # New key as of 30th Dec 2025
    client = genai.Client(api_key=GEMINI_API_KEY)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    main(args)
