import argparse
import pandas as pd
import numpy as np

LABEL_MAP = {
    "emotion-obvious": 1,
    "emotion-plausible": 2,
    "emotion-implausible": 3
}

VALID_LABELS = set(LABEL_MAP.keys())


def normalize_label(x):
    if isinstance(x, str):
        return x.lower()
    return x


def compute_fleiss_kappa(ratings, n_categories=3):
    """
    ratings: numpy array of shape (n_items, n_raters)
             containing integer category labels starting from 1
    """
    n_items, n_raters = ratings.shape

    # Build category count matrix
    M = np.zeros((n_items, n_categories), dtype=int)
    for i in range(n_items):
        for r in ratings[i]:
            M[i, int(r) - 1] += 1

    # Proportion of all assignments to each category
    p = M.sum(axis=0) / (n_items * n_raters)

    # Agreement per item
    P_i = (np.sum(M * M, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = np.mean(P_i)
    P_e = np.sum(p * p)

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def compute_krippendorff_alpha_nominal(ratings):
    """
    ratings: numpy array of shape (n_items, n_raters)
             containing integer category labels starting from 1
    """
    ratings = np.asarray(ratings)
    n_items, n_raters = ratings.shape

    # Flatten ratings
    values = ratings.flatten()

    # Observed disagreement
    Do = 0.0
    for i in range(n_items):
        row = ratings[i]
        for j in range(len(row)):
            for k in range(j + 1, len(row)):
                Do += 0 if row[j] == row[k] else 1

    Do /= (n_items * n_raters * (n_raters - 1) / 2)

    # Expected disagreement
    categories, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()

    De = 1 - np.sum(probs ** 2)

    alpha = 1 - (Do / De)
    return alpha


def compute_icc_3(data):
    """
    Computes ICC(3,1) and ICC(3,k) following
    Shrout & Fleiss (1979), two-way fixed effects, absolute agreement.
    """
    data = np.asarray(data, dtype=float)

    n, k = data.shape

    mean_per_target = data.mean(axis=1)
    mean_per_rater = data.mean(axis=0)
    grand_mean = data.mean()

    # Sum of squares
    ss_targets = k * np.sum((mean_per_target - grand_mean) ** 2)
    ss_error = np.sum(
        (data - mean_per_target[:, None] - mean_per_rater + grand_mean) ** 2
    )

    ms_targets = ss_targets / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    # ICC(3,1): single rater
    icc_3_1 = (ms_targets - ms_error) / (ms_targets + (k - 1) * ms_error)

    # ICC(3,k): mean of k raters
    icc_3_k = (ms_targets - ms_error) / ms_targets

    return icc_3_1, icc_3_k


def main():
    parser = argparse.ArgumentParser(
        description="Compute ICC(3,1) and ICC(3,k) for three annotators."
    )
    parser.add_argument("--csv_file", required=True)
    parser.add_argument("--column1", required=True)
    parser.add_argument("--column2", required=True)
    # parser.add_argument("--column3", required=True)
    parser.add_argument('--count', type=int, required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    # Normalize labels
    for col in [args.column1, args.column2]: #, args.column3]:
        df[col] = df[col].apply(normalize_label)

    # Listwise deletion + take first 5k
    df = df[
        df[args.column1].isin(VALID_LABELS) &
        df[args.column2].isin(VALID_LABELS) 
        # df[args.column3].isin(VALID_LABELS)
    ].head(args.count)

    print(f"Using {len(df)} valid rows for ICC calculation.")

    if len(df) < 2:
        raise ValueError("Not enough valid rows to compute ICC.")

    # ratings = df[[args.column1, args.column2, args.column3]].replace(LABEL_MAP)
    ratings = df[[args.column1, args.column2]].replace(LABEL_MAP)

    icc_3_1, icc_3_k = compute_icc_3(ratings.values)

    print(f"ICC(3,1) — single annotator reliability: {icc_3_1:.4f}")
    print(f"ICC(3,3) — mean annotator reliability:   {icc_3_k:.4f}")

    # Compute Fleiss' Kappa
    fleiss_kappa = compute_fleiss_kappa(ratings.values, n_categories=len(LABEL_MAP))
    print(f"Fleiss' Kappa: {fleiss_kappa:.4f}")

    # Compute Krippendorff's Alpha
    krippendorff_alpha = compute_krippendorff_alpha_nominal(ratings.values)
    print(f"Krippendorff's Alpha: {krippendorff_alpha:.4f}")


if __name__ == "__main__":
    main()
