#!/usr/bin/env python3
"""
Robust Batch API pipeline for any dataset size.

Features:
- Streaming JSONL creation in chunks (no huge in-memory lists).
- One batch per JSONL chunk (safe for millions of tasks).
- Stable global custom_id: item_0, item_1, ...
- Retry & auto-resubmit failed items up to max_retries.
- Final CSV built in the same order as tasks were created.
- Keeps your prompts intact (no prompt changes).
"""

import os
import json
import time
import math
import re
import argparse
from datetime import datetime
from pprint import pprint
import base64
import tiktoken
from pathlib import Path
import pandas as pd
from openai import OpenAI
from itertools import islice

# ---------------------------
# Config / Globals
# ---------------------------
openai_client = OpenAI(
    api_key="YOUR_API_KEY"
    )

VALID_LABELS = {
    "A": "Self-Sufficient",
    "B": "Context-Dependent",
    "C": "Emotion-Impossible",
}

VALID_EMOTIONS = [
    "ANGER", "DISGUST", "FEAR", "SURPRISE", "NEUTRAL", "SADNESS", "JOY"
]

# How many tasks per JSONL file. Tune this based on batch API size limits and memory.
# For small datasets you can leave this large; for huge datasets it's safer to keep it
# under ~50k tasks per JSONL (but this number can be adjusted).
DEFAULT_MAX_TASKS_PER_JSONL = 1000  #10000

# How many times to retry failed items (resubmitted as a new batch).
DEFAULT_MAX_RETRIES = 1

# Polling parameters for batch completion (in seconds)
POLL_INTERVAL = 60
POLL_LONG_INTERVAL = 180  # for later rounds backoff


# ---------------------------
# Utility helpers
# ---------------------------
def timed_print(*args):
    ts = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    print(f"[{ts}]", *args)

# ---------------------------------------------------------
# Utility: Encode and Decode metadata in custom_id
# ---------------------------------------------------------

def encode_meta(text):
    return base64.urlsafe_b64encode(text.encode()).decode()

def decode_meta(cid):
    # format: item_<id>|<emotion>|<b64>|<tagged>
    parts = cid.split("|")
    if len(parts) != 4:
        return "", "", "", ""
    idx = parts[0]
    emotion = parts[1]
    text = base64.urlsafe_b64decode(parts[2]).decode()
    tagged = parts[3].lower() == "true"
    return idx, text, emotion, tagged

# ---------------------------
# Create JSONL job entry
# ---------------------------
def make_jsonl_job(global_idx, text, emotion, tagged):
    """
    Create a single JSONL job dict for the batch API.
    Encodes metadata in custom_id for stable retrieval.
    Uses global_idx to create stable custom_id: item_{global_idx}.
    """

    b64_text = encode_meta(text)
    cid = f"item_{global_idx}|{emotion}|{b64_text}|{tagged}"

    # Build prompt (kept identical to your original)
    prompt = f"""
        You are an expert NLP evaluator. You must classify how dependent the text is on surrounding context to express the given emotion.

        Allowed labels (SHORT FORM):
        A = Self-Sufficient
        B = Context-Dependent
        C = Emotion-Impossible

        Label meanings:
        A: The emotion is clearly expressed without needing additional context.
        B: The emotion is ambiguous or barely present, and typically needs outside context.
        C: Even with added context, this text cannot plausibly express the target emotion.

        Special rule for NEUTRAL:
        - Since "NEUTRAL" means complete absence of the 6 target emotions {{ANGER, SADNESS, JOY, FEAR, DISGUST, SURPRISE}},
        a sentence cannot be "A" (Self-Sufficient) for both NEUTRAL and another emotion.

        STRICT OUTPUT REQUIREMENTS:
        - You must output ONLY a JSON object of the form:
        {{"label":"A"}} or {{"label":"B"}} or {{"label":"C"}}
        - Do NOT output any text before or after the JSON.
        - Do NOT explain your choice.
        - Do NOT add any additional keys.
        - If unsure, pick the best label; do NOT invent new labels.

        Now evaluate:

        Text: {text}
        Emotion: {emotion}
    """.strip()


    job = {
        "custom_id": cid,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model":"gpt-5-mini",  # "gpt-4o",  #"gpt-4.1", # "gpt-5.1",  # "gpt-5",  # "gpt-5-mini", "gpt-4o-mini",
            "messages":[
                {
                    "role": "system",
                    "content": (
                        "You are an expert NLP evaluator.\n"
                        "You must output a JSON object with exactly one key: 'label'.\n"
                        "The value MUST be exactly one of these single codes:\n"
                        "A = Self-Sufficient\n"
                        "B = Context-Dependent\n"
                        "C = Emotion-Impossible\n"
                        "No other output is allowed."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            # "temperature": 0,
            "max_completion_tokens": 1800, # 1028,
        }
    }

    return job


# ---------------------------
# 1) load_items_from_csv: streaming generator
# ---------------------------
def load_items_from_csv(input_csv):
    """
    Yields (text, emotion) pairs for every row in CSV x VALID_EMOTIONS.
    This avoids loading the whole expanded list into memory.
    """
    # Use pandas to read in chunks if the CSV is enormous
    # We'll read row-by-row using pandas iterator to keep memory low.
    if input_csv.endswith(".tsv"):
        df_iter = pd.read_csv(input_csv, sep="\t", chunksize=10000, iterator=True, dtype=str)
    else:
        df_iter = pd.read_csv(input_csv, chunksize=10000, iterator=True, dtype=str)

    for df in df_iter:
        # Ensure 'text' column exists
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a 'text' column.")
        for _, row in df.iterrows():
            text = str(row["text"]) if not pd.isna(row["text"]) else ""
            tagged_emotions = str(row["ekman_emos"]) if "ekman_emos" in df.columns and not pd.isna(row["ekman_emos"]) else ""
            for emotion in VALID_EMOTIONS:
                if emotion in tagged_emotions.split(","):
                    tagged = True
                else:
                    tagged = False
                yield text, emotion, tagged


# ---------------------------
# 2) create_jsonl_chunks: write JSONL files in streaming chunks
# ---------------------------
def create_jsonl_chunks(input_csv, jsonl_path, base_fname, max_tasks_per_file=DEFAULT_MAX_TASKS_PER_JSONL):
    """
    Read items from load_items_from_csv() and write JSONL files each containing up to
    max_tasks_per_file tasks. Returns list of jsonl file paths and total_items count.
    Each JSONL line contains "meta" inside body so we can reliably reload later.
    custom_id uses global incremental counter: item_{global_idx}.
    """
    os.makedirs(jsonl_path, exist_ok=True)
    jsonl_files = []
    global_idx = 0
    file_idx = 0
    current_f = None

    def open_new_file(fi):
        fname = os.path.join(jsonl_path, f"{base_fname}_{fi:05d}.jsonl")
        f = open(fname, "w", encoding="utf-8")
        return f, fname

    current_f, fname = open_new_file(file_idx)
    jsonl_files.append(fname)
    tasks_written_in_file = 0

    for text, emotion, tagged in load_items_from_csv(input_csv):
        # generate jsonl job
        job = make_jsonl_job(global_idx, text, emotion, tagged)

        current_f.write(json.dumps(job, ensure_ascii=False) + "\n")
        tasks_written_in_file += 1
        global_idx += 1

        if tasks_written_in_file >= max_tasks_per_file:
            current_f.close()
            file_idx += 1
            current_f, fname = open_new_file(file_idx)
            jsonl_files.append(fname)
            tasks_written_in_file = 0

    # close final file
    if current_f:
        current_f.close()

    total_items = global_idx
    timed_print(f"Created {len(jsonl_files)} jsonl file(s) with total {total_items} tasks.")
    return jsonl_files, total_items

# ---------------------------
# 3) verify_batch: check JSONL for size/token issues
# ---------------------------

# Limits (set slightly below API hard limits)
MAX_BYTES = 4.5 * 1024 * 1024         # ~4.5 MB per request line
MAX_TOKENS = 120_000                  # safety margin below ~128k
MAX_UTF8_DENSITY = 3.5     # average bytes per character threshold
MODEL_OVERHEAD = 600        # internal system wrapper tokens
ENC = tiktoken.encoding_for_model("gpt-5-mini")

CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0B-\x1F\x7F]"
)

def _count_tokens(messages):
    """Approximate total token count across all message contents."""
    text = ""
    for msg in messages:
        text += msg.get("content", "")
    return len(ENC.encode(text))

def _utf8_density(s):
    if not s:
        return 1.0
    return len(s.encode("utf-8")) / max(1, len(s))

def verify_batch(path):
    """
    Validates an OpenAI batch JSONL file.
    Returns:
      - True if the file passes all checks
      - Otherwise, a dictionary summarizing all issues
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    oversized_bytes = []
    oversized_tokens = []
    invalid_json = []
    missing_fields = []
    control_chars = []
    utf8_explosive = []
    risky_output = []
    endpoint_warnings = []

    with path.open("r", encoding="utf-8") as f:
        for i, raw_line in enumerate(f):
            line_num = i + 1
            line = raw_line.strip()

            if not line:
                invalid_json.append((line_num, "empty line"))
                continue

            # Byte size
            size_bytes = len(line.encode("utf-8"))
            if size_bytes > MAX_BYTES:
                oversized_bytes.append((line_num, size_bytes))

            # Strict JSON
            try:
                obj = json.loads(line)
            except Exception as e:
                invalid_json.append((line_num, f"json error: {e}"))
                continue

            # Required fields
            required = ["custom_id", "method", "url", "body"]
            missing = [k for k in required if k not in obj]
            if missing:
                missing_fields.append((line_num, missing))
                continue

            method = obj["method"]
            url = obj["url"]
            body = obj["body"]
            model = body.get("model")

            # # Endpoint compatibility warning (non-fatal)
            # if model == "gpt-5-mini" and url == "/v1/chat/completions":
            #     endpoint_warnings.append(line_num)

            # Messages / input
            messages = body.get("messages") or body.get("input")
            if not isinstance(messages, list):
                missing_fields.append((line_num, "messages/input missing or not list"))
                continue

            # Token count (+ overhead)
            try:
                tok = _count_tokens(messages)
                tok_with_overhead = tok + MODEL_OVERHEAD
                if tok_with_overhead > MAX_TOKENS:
                    oversized_tokens.append((line_num, tok_with_overhead))
            except Exception as e:
                invalid_json.append((line_num, f"tokenizer error: {e}"))
                continue

            # Control chars
            joined = "".join(msg.get("content", "") for msg in messages)
            if CONTROL_CHAR_RE.search(joined):
                control_chars.append(line_num)

            # UTF-8 density
            if _utf8_density(joined) > MAX_UTF8_DENSITY:
                utf8_explosive.append(line_num)

            # Risky output length
            if len(joined) > 30_000:
                risky_output.append((line_num, len(joined)))

    summary = {
        "oversized_bytes": oversized_bytes,
        "oversized_tokens": oversized_tokens,
        "invalid_json": invalid_json,
        "missing_fields": missing_fields,
        "control_chars": control_chars,
        "utf8_explosive": utf8_explosive,
        "risky_output": risky_output,
        "endpoint_warnings": endpoint_warnings,
        "total_requests": i + 1
    }

    # ---- Return logic ----
    # If all problem lists are empty (except endpoint warnings), return TRUE
    problem_keys = [
        "oversized_bytes",
        "oversized_tokens",
        "invalid_json",
        "missing_fields",
        "control_chars",
        "utf8_explosive",
        "risky_output",
    ]
    any_problems = any(len(summary[k]) > 0 for k in problem_keys)

    if not any_problems:
        return True, {}
    else:
        return False, summary

# ---------------------------
# 4) submit_batch: upload + create batch job
# ---------------------------
def submit_batch(jsonl_file):
    """
    Uploads jsonl_file as a batch input and creates a batch job.
    Returns: batch_id, and the file id (input_file_id) if needed.
    """
    timed_print(f"Uploading {jsonl_file} ...")
    # create file object
    with open(jsonl_file, "rb") as f:
        batch_file = openai_client.files.create(file=f, purpose="batch")

    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    timed_print(f"Submitted batch for {os.path.basename(jsonl_file)} -> batch_id: {batch_job.id}")
    return batch_job.id


# ---------------------------
# 5) wait_for_batch: poll until job completes/failed
# ---------------------------
def wait_for_batch(batch_id, initial_interval=POLL_INTERVAL, max_wait_minutes=24*60):
    """
    Polls the batch status periodically until it is completed/failed/canceled/expired.
    Returns the final batch object (as returned by openai_client.batches.retrieve()).
    """
    timed_print(f"Waiting for batch {batch_id} to finish...")
    start = time.time()
    interval = initial_interval
    max_wait_seconds = max_wait_minutes * 60
    prev_status = None
    while True:
        batch = openai_client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None) or batch.status
        if status != prev_status:
            timed_print(f"Batch {batch_id} status: {status}")
        if status in ("completed", "failed", "expired", "canceled"):
            return batch

        if time.time() - start > max_wait_seconds:
            timed_print(f"Timeout waiting for batch {batch_id}. Returning current state.")
            return batch

        time.sleep(interval)
        # gentle backoff
        interval = min(interval * 1.5, POLL_LONG_INTERVAL)
        prev_status = status


# ---------------------------
# 6) read_batch_output with retry-auto-resubmit for failed items
# ---------------------------
def read_batch_output_with_retries(jsonl_input_file, batch, out_jsonl_file, max_retries=DEFAULT_MAX_RETRIES):
    """
    Given a completed batch object and the input JSONL filename (so we can map meta data),
    download the output, parse results into a dict {cid: label_or_error}, and if some items
    failed, auto-resubmit them up to max_retries times.
    Returns final results dict for all items in that input file.
    Notes:
      - jsonl_input_file: original JSONL file used to create this batch (so we can reconstruct meta)
      - batch: returned batch object (from wait_for_batch)
      - out_jsonl_file: path where to save the batch output bytes
      - This function will create temporary JSONL files and batch jobs for failed items as needed.
    """
    # # Helper: read meta map from original input JSONL
    # Revised meta_map loader using encoded metadata in custom_id
    def load_meta_map(input_jsonl_path):
        meta_map = {}  # cid_str -> meta dict
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj["custom_id"]  #.replace("item_", "")
                cid, text, emotion, tagged = decode_meta(cid)
                meta_map[cid] = {"text": text, "emotion": emotion, "tagged": tagged}
        return meta_map

    # Download initial batch output and store
    def download_and_parse(curr_batch, save_path, input_jsonl):
        # load meta map
        meta_map = load_meta_map(input_jsonl)

        if curr_batch == "" and os.path.exists(save_path):
            timed_print("Batch object is empty, but output file exists. Loading from file:", save_path)
            with open(save_path, "rb") as f:
                raw = f.read()
            timed_print("Loaded output from existing file:", save_path)
            
        elif not getattr(curr_batch, "output_file_id", None):
            timed_print("Batch has no output file. Checking error file...")
            if getattr(curr_batch, "error_file_id", None):
                err = openai_client.files.content(curr_batch.error_file_id)
                timed_print("Batch error file content:", err.text)
            else:
                timed_print("Batch has no error file either. Cannot proceed.")
                return {cid: "FAILED_BATCH" for cid in meta_map.keys()}, []
                
            # mark everything as ERROR
            failed_cids = list(meta_map.keys())
            # create a dict with all cids set to "ERROR"
            return {cid: "ERROR" for cid in meta_map.keys()}, failed_cids

        else:
            timed_print("Downloading batch output...")
            raw = openai_client.files.content(curr_batch.output_file_id).content
            with open(save_path, "wb") as f:
                f.write(raw)
            timed_print(f"Saved batch output to {save_path}.")

        results_local = {}
        failed_cids = []

        lines = raw.decode("utf-8").splitlines()
        for line in lines:
            obj = json.loads(line)
            cid = obj["custom_id"].split("|")[0]  # get only item_{id} part

            # Some batch entries may contain an 'error' key under response
            response_block = obj.get("response", {})
            if "error" in response_block:
                results_local[cid] = "ERROR"
                failed_cids.append(cid)
                continue

            try:
                output = response_block["body"]["choices"][0]["message"]["content"].strip()
            except Exception:
                results_local[cid] = "ERROR"
                failed_cids.append(cid)
                continue

            # parse JSON output from model
            try:
                parsed = json.loads(output)
                label = parsed.get("label", "PARSING_ERROR")
            except Exception:
                label = "PARSING_ERROR"

            if label not in VALID_LABELS:
                results_local[cid] = "PARSING_ERROR"
                failed_cids.append(cid)
            else:
                results_local[cid] = VALID_LABELS[label]

        # Any cids in meta_map but missing from results_local are treated as ERROR
        for cid in meta_map.keys():
            if cid not in results_local:
                results_local[cid] = "MISSING"
                failed_cids.append(cid)

        # de-duplicate and sort failed array
        failed_cids = sorted(set(failed_cids), key=lambda x: int(x.replace("item_", "")))
        return results_local, failed_cids

    # first parse
    results, failed = download_and_parse(batch, out_jsonl_file, jsonl_input_file)
    timed_print(f"Initial parse: {len(results)} items; {len(failed)} failed/missing/parsing errors.")
    # pprint(dict(islice(results.items(), 5)))

    attempt = 0
    meta_map = load_meta_map(jsonl_input_file)
    while failed and attempt < max_retries:
        attempt += 1
        timed_print(f"Retry attempt {attempt} for {len(failed)} failed items...")

        # Build a temporary JSONL with the failed items (use their meta)
        tmp_jsonl = jsonl_input_file.replace(".jsonl", f".retry{attempt}.jsonl")
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for cid in failed:
                meta = meta_map.get(cid, {})
                text = meta.get("text", "")
                emotion = meta.get("emotion", "")
                tagged = meta.get("tagged", False)
                cidx = int(cid.replace("item_", ""))

                # generate jsonl job
                job = make_jsonl_job(cidx, text, emotion, tagged)
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
        
        # prepare output file path for retry
        retry_out_file = tmp_jsonl.replace(".jsonl", "_out.jsonl")
        if not os.path.exists(retry_out_file):
            # submit retry batch
            valid, summary = verify_batch(tmp_jsonl)
            if not valid:
                timed_print(f"Batch verification failed for retry JSONL file: {tmp_jsonl}\nSummary:\n", summary)
                exit()
            else:
                retry_batch_id = submit_batch(tmp_jsonl)
                retry_batch = wait_for_batch(retry_batch_id)
        else:
            timed_print("Retry output file already exists. Loading from file:", retry_out_file)
            retry_batch = ""  # dummy, will load from file
        
        # parse retry results
        retry_results, retry_failed = download_and_parse(retry_batch, retry_out_file, tmp_jsonl)

        # merge retry_results into main results
        for cid, val in retry_results.items():
            results[cid] = val
       
        # prepare next round failed list
        failed = retry_failed
        timed_print(f"After attempt {attempt}: remaining failed = {len(failed)}")

        # if still failed and attempts remain, loop continues

    if failed:
        timed_print(f"Final unresolved failed items for {jsonl_input_file}: {len(failed)} (will be marked as 'ERROR'/'MISSING').")
        for cid in failed:
            results[cid] = "ERROR"

    return results


# ---------------------------
# 7) process_all_jsonl_files: orchestrate submission + output parsing & retries
# ---------------------------
def process_all_jsonl_files(jsonl_files, max_retries=DEFAULT_MAX_RETRIES):
    """
    For each jsonl file in jsonl_files:
      - submit batch
      - wait for completion
      - download + parse output
      - handle retries via read_batch_output_with_retries
    Returns a results dict mapping global cid (string) -> label
    """
    all_results = {}

    for jsonl_file in jsonl_files:
        timed_print("=== Processing", jsonl_file)
        out_jsonl_file = jsonl_file.replace(".jsonl", "_out.jsonl")
         # If the batch output file already exists, skip submitting
        if not os.path.exists(out_jsonl_file):
            # # Submit the batch job      
            valid, summary = verify_batch(jsonl_file)
            if not valid:
                timed_print(f"Batch verification failed for file: {jsonl_file}\nSummary:\n", summary)
                exit()
            else:
                timed_print("Batch verification passed for file: ", jsonl_file)
                batch_id = submit_batch(jsonl_file)
                batch_obj = wait_for_batch(batch_id)
        else:
            batch_obj = ""  # dummy, will load from file
        print("Batch object ready for processing. With out_jsonl file: ", out_jsonl_file)
        results = read_batch_output_with_retries(jsonl_file, batch_obj, out_jsonl_file, max_retries=max_retries)

        # Merge
        all_results.update(results)
        timed_print(f"Completed {jsonl_file}. Results now contain {len(all_results)} items total.")

    return all_results


# ---------------------------
# 8) build_output_csv: stream final CSV in original order
# ---------------------------
def build_output_csv_from_jsonls(jsonl_files, results_dict, out_csv):
    """
    Reads each input JSONL in order and writes rows to CSV in the same order.
    Avoids building an enormous in-memory DataFrame for very large datasets.
    """
    timed_print("Writing output CSV to", out_csv)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cols = ["idx", "sentence", "emotion", "tagged", "context_dependency"]
    first_write = True

    # We'll stream and append
    for jsonl in jsonl_files:
        rows = []
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj["custom_id"]
                idx, sentence, emotion, tagged = decode_meta(cid)

                label = results_dict.get(idx, "MISSING")
                rows.append({"idx": idx, "sentence": sentence, "emotion": emotion, "tagged": tagged, "context_dependency": label})

        df_chunk = pd.DataFrame(rows, columns=cols)
        if first_write:
            df_chunk.to_csv(out_csv, index=False, mode="w")
            first_write = False
        else:
            df_chunk.to_csv(out_csv, index=False, header=False, mode="a")

    timed_print("CSV writing completed:", out_csv)


# ---------------------------
# 9) main: full pipeline
# ---------------------------
def main(input_csv, jsonl_path, jsonl_base_name, output_csv, max_tasks_per_jsonl=DEFAULT_MAX_TASKS_PER_JSONL, max_retries=DEFAULT_MAX_RETRIES):
    timed_print("Starting pipeline.")
    # # # 1. Create JSONL chunk files (streaming)
    jsonl_files, total_items = create_jsonl_chunks(input_csv, jsonl_path, jsonl_base_name, max_tasks_per_file=max_tasks_per_jsonl)
    timed_print(f"Created {len(jsonl_files)} JSONL files for {total_items} total tasks.")

    print("jsonl_files to process:", jsonl_files)   
    
    # 2. Process each jsonl file: submit, wait, download & parse with retries
    results = process_all_jsonl_files(jsonl_files, max_retries=max_retries)

    timed_print(f"All batches processed. Total results collected: {len(results)}")

    # 3. Build the final CSV in original order
    build_output_csv_from_jsonls(jsonl_files, results, output_csv)

    timed_print("Pipeline finished successfully.")


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust batch pipeline for context-dependency classification.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV (must contain 'text' column).")
    parser.add_argument("--jsonl_path", default="./batchAPI_contDep/", help="Directory to write JSONL chunk files.")
    parser.add_argument("--jsonl_base", required=True, help="Base name for JSONL files (e.g. 'myjob' -> myjob_00001.jsonl).")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--max_tasks_per_jsonl", type=int, default=DEFAULT_MAX_TASKS_PER_JSONL, help="Max tasks per JSONL chunk (tune per API limits).")
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retry attempts for failed items.")
    args = parser.parse_args()

    main(args.input_csv, args.jsonl_path, args.jsonl_base, args.output_csv, max_tasks_per_jsonl=args.max_tasks_per_jsonl, max_retries=args.max_retries)
