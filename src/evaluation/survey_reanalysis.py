"""
Re-analysis of listening test survey data.
Computes MTS vs PHC correlations per fragment, variability analysis,
and identifies fragments where expert/non-expert disagreement is highest.
"""
import os
import sys
import csv
import pandas as pd
import numpy as np
from scipy import stats


def load_survey_data():
    """Load and clean survey data."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, "data", "survey", "raw", "1.csv")
    df = pd.read_csv(csv_path, encoding="utf-8")
    return df


def parse_phc_response(response):
    """Convert Spanish survey response to numeric PHC value [-2, 2]."""
    mapping = {
        "A mucho más compleja que B": 2,
        "A más compleja que B": 1,
        "Igual de complejas": 0,
        "B más compleja que A": -1,
        "B mucho más compleja que A": -2,
    }
    response = str(response).strip()
    for key, val in mapping.items():
        if key in response or response == key:
            return val
    return np.nan


def categorize_mts(score_norm):
    """Categorize MTS into Low, Medium, High."""
    try:
        s = float(str(score_norm).replace(",", "."))
    except (ValueError, TypeError):
        return "Unknown"
    if s < 0.25:
        return "Low"
    elif s < 0.6:
        return "Medium"
    else:
        return "High"


def analyze_survey():
    """Main analysis function."""
    df = load_survey_data()
    print(f"Loaded {len(df)} participants")

    # Parse MTS
    df["MTS"] = df["SCORE_NORM"].apply(
        lambda x: float(str(x).replace(",", ".")) if pd.notna(x) else np.nan
    )
    df["MTS_group"] = df["MTS"].apply(categorize_mts)

    # Filter participants who passed control questions
    # C1 and C2: if they answered correctly (we assume both controls were passed
    # if the pattern matches - C1 should be opposite extremes, C2 check)
    # For now, include all but flag those who answered "Igual de complejas" to C1
    valid = df[df["C1"].notna() & (df["C1"] != "Igual de complejas")].copy()
    n_filtered = len(df) - len(valid)
    if n_filtered > 0:
        print(f"Filtered out {n_filtered} participants (failed C1 control)")

    # Block 1: SCL vs RVAE (L1-L6)
    block1_cols = [f"L{i}" for i in range(1, 7)]
    # Block 2: SCL vs Human (H1-H6)
    block2_cols = [f"H{i}" for i in range(1, 7)]

    # Convert responses to numeric PHC
    for col in block1_cols + block2_cols:
        valid[f"{col}_phc"] = valid[col].apply(parse_phc_response)

    # --- Per-fragment PHC statistics ---
    print("\n=== Block 1: SCL vs RVAE ===")
    print(f"{'Fragment':<8} {'Mean PHC':>10} {'Std':>8} {'Median':>8} {'N':>6} {'%Positive':>10}")
    print("-" * 55)
    block1_stats = []
    for col in block1_cols:
        phc_col = f"{col}_phc"
        values = valid[phc_col].dropna()
        mean_phc = values.mean()
        std_phc = values.std()
        med_phc = values.median()
        n = len(values)
        pct_pos = (values > 0).mean() * 100
        block1_stats.append({"fragment": col, "mean_phc": mean_phc, "std": std_phc, "n": n})
        print(f"{col:<8} {mean_phc:>10.3f} {std_phc:>8.3f} {med_phc:>8.3f} {n:>6} {pct_pos:>9.1f}%")

    print(f"\n  Overall Block 1 mean PHC: {np.mean([s['mean_phc'] for s in block1_stats]):.3f}")

    print("\n=== Block 2: SCL vs Human ===")
    print(f"{'Fragment':<8} {'Mean PHC':>10} {'Std':>8} {'Median':>8} {'N':>6} {'%Positive':>10}")
    print("-" * 55)
    block2_stats = []
    for col in block2_cols:
        phc_col = f"{col}_phc"
        values = valid[phc_col].dropna()
        mean_phc = values.mean()
        std_phc = values.std()
        med_phc = values.median()
        n = len(values)
        pct_pos = (values > 0).mean() * 100
        block2_stats.append({"fragment": col, "mean_phc": mean_phc, "std": std_phc, "n": n})
        print(f"{col:<8} {mean_phc:>10.3f} {std_phc:>8.3f} {med_phc:>8.3f} {n:>6} {pct_pos:>9.1f}%")

    print(f"\n  Overall Block 2 mean PHC: {np.mean([s['mean_phc'] for s in block2_stats]):.3f}")

    # --- PHC by MTS group ---
    print("\n=== PHC by MTS Group ===")
    for block_name, cols in [("Block 1 (SCL vs RVAE)", block1_cols), ("Block 2 (SCL vs Human)", block2_cols)]:
        print(f"\n  {block_name}:")
        print(f"  {'MTS Group':<12} {'Mean PHC':>10} {'Std':>8} {'N':>6} {'%Positive':>10}")
        print("  " + "-" * 50)
        for group in ["Low", "Medium", "High"]:
            group_df = valid[valid["MTS_group"] == group]
            if len(group_df) == 0:
                continue
            all_values = []
            for col in cols:
                all_values.extend(group_df[f"{col}_phc"].dropna().tolist())
            all_values = pd.Series(all_values)
            mean_phc = all_values.mean()
            std_phc = all_values.std()
            n = len(all_values)
            pct_pos = (all_values > 0).mean() * 100
            print(f"  {group:<12} {mean_phc:>10.3f} {std_phc:>8.3f} {n:>6} {pct_pos:>9.1f}%")

    # --- Correlation: MTS vs PHC per fragment ---
    print("\n=== Correlation: MTS vs PHC per Fragment ===")
    print(f"  {'Fragment':<8} {'Pearson r':>10} {'p-value':>10} {'Interpretation':>30}")
    print("  " + "-" * 65)
    correlations = []
    for block_name, cols in [("Block 1", block1_cols), ("Block 2", block2_cols)]:
        for col in cols:
            phc_col = f"{col}_phc"
            pair = valid[[phc_col, "MTS"]].dropna()
            if len(pair) < 5:
                continue
            r, p = stats.pearsonr(pair["MTS"], pair[phc_col])
            interp = ""
            if p < 0.05:
                interp = "SIGNIFICANT"
                if r > 0:
                    interp += " (experts rate SCL higher)"
                else:
                    interp += " (experts rate SCL LOWER!)"
            else:
                interp = "not significant"
            correlations.append({"fragment": col, "r": r, "p": p, "block": block_name})
            print(f"  [{block_name}] {col:<5} {r:>10.3f} {p:>10.4f} {interp:>30}")

    # --- Identify fragments with highest expert/novice disagreement ---
    print("\n=== Expert vs Novice Disagreement Analysis ===")
    for block_name, cols in [("Block 1", block1_cols), ("Block 2", block2_cols)]:
        print(f"\n  {block_name}:")
        for col in cols:
            phc_col = f"{col}_phc"
            low = valid[valid["MTS_group"] == "Low"][phc_col].dropna()
            high = valid[valid["MTS_group"] == "High"][phc_col].dropna()
            if len(low) >= 3 and len(high) >= 3:
                diff = high.mean() - low.mean()
                t_stat, t_p = stats.ttest_ind(low, high) if len(low) > 1 and len(high) > 1 else (np.nan, np.nan)
                flag = " *** HIGH DISAGREEMENT ***" if abs(diff) > 1.0 else ""
                print(f"    {col}: Low MTS={low.mean():.3f}, High MTS={high.mean():.3f}, "
                      f"Diff={diff:.3f} (p={t_p:.3f}){flag}")

    # Save results
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "survey", "processed"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save per-fragment stats
    stats_df = pd.DataFrame(block1_stats + block2_stats)
    stats_df.to_csv(os.path.join(output_dir, "fragment_stats.csv"), index=False)

    # Save correlations
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(os.path.join(output_dir, "mts_phc_correlations.csv"), index=False)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    analyze_survey()
