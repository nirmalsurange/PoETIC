import pandas as pd
import os
import argparse
from groq import Groq
from tqdm import tqdm
import time
from groq import RateLimitError


# Groq GoEmo API key setup
GROQ_API_KEY = 'Your_API_KEY'  # Sir's API key with higher rate limits
client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.3-70b-versatile"  # Official Groq ID as per docs[12]

def get_label(sentence, emotion):
    prompt = (
        "Classify the dependency of the following '{sentence}' on its prior context to express the emotion '{emotion}' "
        "Allowed classes with definition:\n"
        "'SS': The emotion is clearly expressed without needing additional context.\n"
        "'CD': The emotion is ambiguous or barely present, and typically needs outside context.\n"
        "'EI': Even with added context, this text cannot plausibly express the target emotion.\n"
        " Reply ONLY with the class label, do NOT include any explanation or text.\n"
        "Example output: SS\n"

        "-----------------\n"
        f"Sentence: {sentence}\n"
        f"Emotion: {emotion}\n"
        "Class-label ?"
        
    )
    # print(f"Sentence: {sentence}\nEmotion: {emotion}")
    for i in range(1,10):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                        {"role": "system", "content": "You are a linguist and NLP expert."},
                        {"role": "user", "content": prompt}
                    ],
                max_tokens=3,
                temperature=0.1
            )
            
            text = response.choices[0].message.content
            
            try:
                label = text.split("Class-label ?")[-1].strip().split()
            except Exception:
                label = "Error"  # Default label if parsing fails
            
            return label[0]
        
        except RateLimitError:
            j = 240*i
            print(f"Rate limit exceeded. Retrying after {j} seconds...")
            time.sleep(j)  # Wait for 2 minutes before retrying
    return "Failed"  # Return "Failed" if all retries fail


def main(input_csv, output_csv_prefix):
    """
    This function reads a CSV, processes it in batches, and saves the output.
    The input DataFrame is split into batches of 50. Each batch is
    paired with emotions and processed to generate a context score.
    The results for each batch are saved in a separate CSV file
    with the batch index appended to the filename.

    Args:
        input_csv (str): The path to the input CSV file.
        output_csv_prefix (str): The prefix for the output CSV filenames.
                                 The batch index will be appended to this.
    """
    # --- CSV I/O ---
    df = pd.read_csv(input_csv)

    # Define batch size
    batch_size = 500
    num_batches = (len(df) + batch_size - 1) // batch_size 

    class_labels = {'SS': 'Self-sufficient', 'CD': 'Context-Dependent', 'EI': 'Emotion-Impossible'}
    total_df = pd.DataFrame()

    for batch_index in tqdm(range(142), desc="Processing batches"):
        start_idx = batch_index * batch_size
        end_idx = min((batch_index + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        batch_rows = []
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {batch_index}"):
            sentence = row['sentence']
            emotion = row['emotion']
            row_id = row['idx']
            tagged = row['tagged']

            # for idx, emotion in enumerate(emotions):
            label = get_label(sentence, emotion)
            batch_rows.append({
                'idx': row_id,
                'sentence': sentence,
                'emotion': emotion,
                'tagged': tagged,
                'context_dependency': class_labels.get(label, f"Unknown_{label}")
            })

        # Construct the output filename for the current batch
        output_filename = f"{output_csv_prefix}_batch_{batch_index}.csv"

        # Create a DataFrame for the current batch's output
        batch_output_df = pd.DataFrame(batch_rows)
        # if exists # # batch_output_df = pd.read_csv(output_filename)
        total_df = pd.concat([total_df, batch_output_df], ignore_index=True)

        # Save the batch's output DataFrame to a CSV
        batch_output_df.to_csv(output_filename, index=False)
        print(f"Batch {batch_index} scored and saved to {output_filename}")

    total_df.to_csv(f"{output_csv_prefix}_total.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score context-dependency for sentences based on emotion using an LLM.")
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file.")
    parser.add_argument("--output_csv", type=str, help="Path to the output CSV file.")
    args = parser.parse_args()

    main(args.input_csv, args.output_csv)

