import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tagged_cm = np.array([
    [866, 223],
    [3686, 2225]
])

labels = ["EP+EO", "EI"]
idx = ["Emotion-True", "Emotion-False"]
cm_list = [tagged_cm]
# cm_names = ["Qwen3-8B Zero-Shot", "MeraLlama3-8B Zero-Shot", "GPT-5 Mini", "Gemini 2.5-flash", "Qwen3-8B Few-Shot", "GroqQwen3-32B", "Meta-Llama3-8B Few-Shot", "GroqLlama3-70B"]

for i, cm in enumerate(cm_list):
    df_cm = pd.DataFrame(cm, index=idx, columns=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        df_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True
    )

    plt.xlabel("PoETIC")
    plt.ylabel("GoEmotions")

    plt.tight_layout()
    plt.savefig(f"../cmplots/cmat_tagged.pdf", dpi=300)
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

    plt.xlabel("PoETIC")
    plt.ylabel("GoEmotions")

    plt.tight_layout()
    plt.savefig(f"../cmplots/cmat_norm_tagged.pdf", dpi=300)  # .png
    plt.close()
