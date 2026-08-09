import pandas as pd
# from scipy.stats import pearsonr, spearmanr
import sys
import argparse
from sklearn.metrics import cohen_kappa_score

def compute_correlations(human_ann_path, human_col, majority_path=None):
    """
    Compute correlations between human annotations and model predictions.
    
    Args:
        human_ann_path: Path to human annotated CSV
        majority_path: Path to majority voting CSV
    """
    # Load CSVs
    human_df = pd.read_csv(human_ann_path)
    majority_df = pd.read_csv(majority_path) if majority_path else None
    
    if majority_df:
        # Merge on 'idx' column
        merged_df = pd.merge(human_df, majority_df, on='idx', how='inner')
    else:
        merged_df = human_df
    
    # Models to compare
    models = {
        'contDep_gpt5mini': 'GPT-5-Mini',
        'contDep_groqLlama': 'Groq-Llama',
        'contDep_gemini25flash': 'Gemini-2.5-Flash'
    }
    
    # Check if human column exists
    if human_col not in merged_df.columns:
        print(f"Error: '{human_col}' not found in human annotations CSV")
        return
    
    print(f"Total samples for correlation: {len(merged_df)}\n")
    print("=" * 70)
    
    # Compute correlations for each model
    for model_col, model_name in models.items():
        if model_col not in merged_df.columns:
            print(f"Warning: '{model_col}' not found in majority CSV")
            continue
        
        # Remove rows with NaN values for this comparison
        valid_data = merged_df[[human_col, model_col]].dropna()
        
        if len(valid_data) == 0:
            print(f"{model_name}: No valid data for correlation")
            continue
        
        print(f"\nComputing correlations for {model_name}...")
        # print(f"  Valid samples: {len(valid_data)}")
        print(f"columns compared: {human_col} vs {model_col}")

        kappa = cohen_kappa_score(valid_data[human_col], valid_data[model_col])
        print(kappa)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlations between human annotations and model predictions.")
    parser.add_argument('--human_ann_csv', help='Path to human annotated CSV')
    parser.add_argument('--human_col', default='Corrected_ContDep', help='Column name in human annotated CSV')
    parser.add_argument('--majority_csv', default=None, help='Path to majority voting CSV')
    args = parser.parse_args()
    
    compute_correlations(args.human_ann_csv, args.human_col, args.majority_csv)
