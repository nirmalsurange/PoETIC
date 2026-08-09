import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Confusion matrix data
labels = ["emotion-plausible", "emotion-implausible", "emotion-obvious"]


## Affect - 1K Test =======================================
def confusion_plot(cm_list, cm_names, labels):
    for i, cm in enumerate(cm_list):
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            df_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        # plt.title(f"{cm_names[i]} vs TestGold")

        plt.tight_layout()
        plt.savefig(f"../cmplots/cmat_{cm_names[i].replace(' ', '_')}.pdf", dpi=300)
        plt.close()

        # Row-normalized confusion matrix
        df_cm_norm = df_cm.div(df_cm.sum(axis=1), axis=0)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            df_cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True
        )

        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        # plt.title(f"{cm_names[i]} vs TestGold")

        plt.tight_layout()
        plt.savefig(f"../cmplots/cmat_norm_{cm_names[i].replace(' ', '_')}.pdf", dpi=300)  # .png
        plt.close()


def compute_stats(gold_labels, pred_labels):
    cm = confusion_matrix(gold_labels, pred_labels)
    label_counts = np.unique(pred_labels, return_counts=True)
    return cm, label_counts

def print_confusion_matrix(csv_file, col1, col2):
    df = pd.read_csv(csv_file)
    true_labels = df[col2].str.lower()
    pred_labels = df[col1].str.lower()
    # print(f"True labels from column '{col2}': {true_labels}")
    # print(f"Predicted labels from column '{col1}': {pred_labels}")
    # exit()
    cm, label_counts = compute_stats(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(cm)
    print("Label Counts:")
    print(label_counts)

# Example usage
# print_confusion_matrix("../data/TestGold.csv", "predicted_context_dependency", "majority_vote")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Print confusion matrix from CSV")
    argparser.add_argument("--csv_file", required=True, help="Path to the CSV file")
    argparser.add_argument("--pred_col", required=True, help="Column name for predicted labels")
    argparser.add_argument("--true_col", required=True, help="Column name for true labels")
    args = argparser.parse_args()
    
    # # compute and print confusion matrix
    # print_confusion_matrix(args.csv_file, args.pred_col, args.true_col)

    ## 1K Test =======================================
    qwen_zero_cm = np.array([
        [3449, 24, 1],
        [2387, 59, 1],
        [1074, 2, 2]
    ])

    merallama_zero_cm = np.array([
        [3030, 0, 443],
        [2224, 0, 221],
        [825, 0, 252]
    ])

    gpt5mini_cm = np.array([
        [2929, 235, 310],
        [826, 1579, 43],
        [168, 8, 902]
    ])

    gemini_cm = np.array([
        [2106, 856, 512],
        [623, 1725, 100],
        [134, 14, 930]
    ])

    groqLlama_cm = np.array([
        [2919, 155, 400],
        [1780, 460, 208],
        [323, 0, 755]
    ])

    qwen_fewshot_cm = np.array([
        [2229, 1020, 222],
        [1347, 1032, 67],
        [649, 80, 348]
    ])

    groqQwen_cm = np.array([
        [2226, 1117, 107],
        [811, 1527, 83],
        [613, 31, 422]
     ])

    llama_fewshot_cm = np.array([
        [11, 3456, 7],
        [5, 2443, 0],
        [3, 1067, 8]
    ])


    ## ============== Average ===================
    cm_avg = np.array([
    [2454, 740, 433],
    [529,  1584, 62],
    [170,  13, 951]
    ])


    cm_list = [qwen_zero_cm, merallama_zero_cm, gpt5mini_cm, gemini_cm,  qwen_fewshot_cm, groqQwen_cm, llama_fewshot_cm, groqLlama_cm,cm_avg]
    cm_names = ["Qwen3-8B Zero-Shot", "MeraLlama3-8B Zero-Shot", "GPT-5 Mini", "Gemini 2.5-flash", "Qwen3-8B Few-Shot", "GroqQwen3-32B", "Meta-Llama3-8B Few-Shot", "GroqLlama3-70B","Average confusion"]

    confusion_plot(cm_list, cm_names, labels=['EP', 'EI', 'EO'])
