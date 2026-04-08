#!/usr/bin/env python3
"""
Part 3: Responder vs Non-Responder Analysis

Compare cell population frequencies between melanoma patients receiving miraclib
who respond (yes) versus those who do not (no). Uses PBMC samples only.
Includes statistical tests to identify significant differences.
"""

import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

DB_PATH = Path(__file__).parent / "cell_data.db"


def load_responder_data() -> pd.DataFrame:
    """Load melanoma/miraclib/PBMC data with response information."""
    query = """
    SELECT
        cps.sample_id,
        ct.cell_type_name AS population,
        cps.percentage,
        s.condition,
        s.treatment,
        s.sample_type,
        subj.response
    FROM cell_population_summaries cps
    JOIN cell_types ct ON cps.cell_type_id = ct.cell_type_id
    JOIN samples s ON cps.sample_id = s.sample_id
    JOIN subjects subj ON s.subject_id = subj.subject_id
    WHERE s.condition = 'melanoma'
      AND s.treatment = 'miraclib'
      AND s.sample_type = 'PBMC'
      AND subj.response IN ('yes', 'no')
    ORDER BY cps.sample_id, ct.cell_type_name
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def statistical_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical tests comparing responders vs non-responders."""
    results = []

    for population in df["population"].unique():
        pop_data = df[df["population"] == population]

        responders = pop_data[pop_data["response"] == "yes"]["percentage"].values
        non_responders = pop_data[pop_data["response"] == "no"]["percentage"].values

        if len(responders) < 3 or len(non_responders) < 3:
            continue  # Skip populations with very small sample sizes

        # T-test (Welch's, doesn't assume equal variance) and Mann-Whitney U test
        t_stat, t_pvalue = stats.ttest_ind(responders, non_responders, equal_var=False)
        u_stat, u_pvalue = stats.mannwhitneyu(responders, non_responders, alternative="two-sided")

        # Effect sizes
        # Cohen's d (for parametric context)
        pooled_std = np.sqrt(((len(responders) - 1) * np.std(responders, ddof=1) ** 2 +
                              (len(non_responders) - 1) * np.std(non_responders, ddof=1) ** 2) /
                             (len(responders) + len(non_responders) - 2))
        cohens_d = (np.mean(responders) - np.mean(non_responders)) / pooled_std if pooled_std > 0 else 0
        
        # Rank-biserial correlation (effect size for Mann-Whitney U test)
        n1, n2 = len(responders), len(non_responders)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        results.append({
            "Population": population,
            "Responders_n": len(responders),
            "Non-Responders_n": len(non_responders),
            "Responders_mean": np.mean(responders),
            "Responders_std": np.std(responders, ddof=1),
            "Non-Responders_mean": np.mean(non_responders),
            "Non-Responders_std": np.std(non_responders, ddof=1),
            "T_statistic": t_stat,
            "T-test_pvalue": t_pvalue,
            "Mann-Whitney_pvalue": u_pvalue,
            "Cohens_d": cohens_d,
            "Rank_biserial": rank_biserial,
        })

    results_df = pd.DataFrame(results)
    
    # Apply FDR correction (Benjamini-Hochberg) to Mann-Whitney p-values (primary test)
    if len(results_df) > 0:
        # FDR correction on Mann-Whitney (more appropriate for non-normal data)
        reject_mw, fdr_pvalues_mw, alphacSid_mw, alphacBon_mw = multipletests(
            results_df["Mann-Whitney_pvalue"].values,
            alpha=0.05,
            method='fdr_bh'
        )
        results_df["Mann-Whitney_pvalue_FDR"] = fdr_pvalues_mw
        
        # Create significance columns for both raw and FDR-corrected (at 0.05 and 0.01)
        # Raw Mann-Whitney p-values
        results_df["Significant_MW_raw_0.05"] = (results_df["Mann-Whitney_pvalue"] < 0.05).apply(lambda x: "Yes" if x else "No")
        results_df["Significant_MW_raw_0.01"] = (results_df["Mann-Whitney_pvalue"] < 0.01).apply(lambda x: "Yes" if x else "No")
        
        # FDR-corrected Mann-Whitney p-values
        results_df["Significant_MW_FDR_0.05"] = ["Yes" if r else "No" for r in reject_mw]
        
        # Also create columns for t-test for reference
        results_df["Significant_t_raw_0.05"] = (results_df["T-test_pvalue"] < 0.05).apply(lambda x: "Yes" if x else "No")
        results_df["Significant_t_raw_0.01"] = (results_df["T-test_pvalue"] < 0.01).apply(lambda x: "Yes" if x else "No")
    
    return results_df


def create_boxplots(df: pd.DataFrame, output_path: Path = None):
    """Create boxplots with significance annotations comparing responders vs non-responders for each population."""
    populations = sorted(df["population"].unique())

    # Dynamic grid sizing to handle any number of populations
    n = len(populations)
    cols = min(4, n)  # Max 4 columns per row
    rows = int(np.ceil(n / cols))
    figsize = (5 * cols, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Flatten axes array for consistent indexing
    if n == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, population in enumerate(populations):
        pop_data = df[df["population"] == population]

        responders = pop_data[pop_data["response"] == "yes"]["percentage"].values
        non_responders = pop_data[pop_data["response"] == "no"]["percentage"].values

        ax = axes[idx]
        bp = ax.boxplot([responders, non_responders], labels=["Responders", "Non-Responders"], patch_artist=True)

        # Color boxes
        colors = ["#2ecc71", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Significance annotation (Mann-Whitney U test)
        if len(responders) >= 3 and len(non_responders) >= 3:
            _, p_value = stats.mannwhitneyu(responders, non_responders, alternative="two-sided")
            if p_value < 0.001:
                sig_label = f"p={p_value:.2e} ***"
            elif p_value < 0.01:
                sig_label = f"p={p_value:.3f} **"
            elif p_value < 0.05:
                sig_label = f"p={p_value:.3f} *"
            else:
                sig_label = f"p={p_value:.3f} ns"

            all_vals = np.concatenate([responders, non_responders])
            y_max = np.max(all_vals)
            y_span = np.max(all_vals) - np.min(all_vals) if np.max(all_vals) != np.min(all_vals) else 1
            bar_y = y_max + y_span * 0.08
            tip_y = bar_y - y_span * 0.02
            ax.plot([1, 1, 2, 2], [tip_y, bar_y, bar_y, tip_y], "k-", linewidth=1)
            ax.text(1.5, bar_y + y_span * 0.01, sig_label, ha="center", va="bottom", fontsize=9)
            ax.set_ylim(top=bar_y + y_span * 0.18)

        ax.set_ylabel("Percentage (%)")
        ax.set_title(population.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(len(populations), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Cell Population Frequencies: Responders vs Non-Responders (PBMC, Melanoma + Miraclib)", fontsize=14, y=1.00)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Boxplot saved to {output_path}")

    return fig


def generate_report(df: pd.DataFrame, stats_df: pd.DataFrame):
    """Generate a text report of findings."""
    print("\n" + "="*100)
    print("PART 3: RESPONDER VS NON-RESPONDER ANALYSIS")
    print("="*100)
    print("\nStudy Parameters:")
    print("- Condition: Melanoma")
    print("- Treatment: Miraclib")
    print("- Sample Type: PBMC")
    print("- Comparison: Responders (response='yes') vs Non-Responders (response='no')\n")

    total_samples = df["sample_id"].nunique()
    responder_samples = df[df["response"] == "yes"]["sample_id"].nunique()
    non_responder_samples = df[df["response"] == "no"]["sample_id"].nunique()

    print(f"Total samples: {total_samples}")
    print(f"  - Responders: {responder_samples}")
    print(f"  - Non-Responders: {non_responder_samples}\n")

    print("="*100)
    print("STATISTICAL TEST RESULTS")
    print("="*100 + "\n")

    # Sort by Mann-Whitney p-value (primary test)
    stats_sorted = stats_df.sort_values("Mann-Whitney_pvalue")
    
    for _, row in stats_sorted.iterrows():
        print(f"\n{row['Population'].upper().replace('_', ' ')}")
        print("-" * 80)
        print(f"Sample sizes: Responders n={int(row['Responders_n'])}, Non-Responders n={int(row['Non-Responders_n'])}")
        print(f"Responders:     mean = {row['Responders_mean']:.2f}% (SD = {row['Responders_std']:.2f})")
        print(f"Non-Responders: mean = {row['Non-Responders_mean']:.2f}% (SD = {row['Non-Responders_std']:.2f})")
        print(f"Difference:     {row['Responders_mean'] - row['Non-Responders_mean']:.2f}%")
        print(f"\nStatistical Tests (Mann-Whitney primary):")
        print(f"  - Mann-Whitney p-value (raw):     {row['Mann-Whitney_pvalue']:.4f} {'*' if row['Mann-Whitney_pvalue'] < 0.05 else ''}")
        print(f"  - Mann-Whitney p-value (FDR):     {row['Mann-Whitney_pvalue_FDR']:.4f} {'*' if row['Mann-Whitney_pvalue_FDR'] < 0.05 else ''}")
        print(f"  - Welch t-test p-value:  {row['T-test_pvalue']:.4f} (supplementary)")
        print(f"  - Effect sizes: rank-biserial r = {row['Rank_biserial']:.3f} (Mann-Whitney), Cohen's d = {row['Cohens_d']:.3f} (parametric reference)")
        print(f"  - Significant (raw, p<0.05): {row['Significant_MW_raw_0.05']}")
        print(f"  - Significant (FDR, p<0.05): {row['Significant_MW_FDR_0.05']}")

    print("\n" + "="*100)
    print("SUMMARY: MANN-WHITNEY U TEST RESULTS")
    print("="*100)

    sig_populations_raw = stats_df[stats_df["Mann-Whitney_pvalue"] < 0.05]
    if len(sig_populations_raw) > 0:
        print(f"\nSignificant findings (raw p < 0.05): {len(sig_populations_raw)} population(s)")
        for _, row in sig_populations_raw.iterrows():
            direction = "higher" if row['Responders_mean'] > row['Non-Responders_mean'] else "lower"
            print(f"  - {row['Population']}: {direction} in responders ({row['Responders_mean']:.1f}% vs {row['Non-Responders_mean']:.1f}%, p={row['Mann-Whitney_pvalue']:.4f})")
    else:
        print("\nNo populations showed significant differences at raw p < 0.05")

    sig_populations_fdr = stats_df[stats_df["Significant_MW_FDR_0.05"] == "Yes"]
    if len(sig_populations_fdr) > 0:
        print(f"\nRobust findings (FDR-corrected p < 0.05): {len(sig_populations_fdr)} population(s)")
        for _, row in sig_populations_fdr.iterrows():
            direction = "higher" if row['Responders_mean'] > row['Non-Responders_mean'] else "lower"
            print(f"  - {row['Population']}: {direction} in responders ({row['Responders_mean']:.1f}% vs {row['Non-Responders_mean']:.1f}%, adjusted p={row['Mann-Whitney_pvalue_FDR']:.4f})")

    print("\n" + "="*100)
    print("MULTIPLE TESTING CORRECTION (Benjamini-Hochberg FDR)")
    print("="*100)

    sig_raw = stats_df[stats_df["Significant_MW_raw_0.05"] == "Yes"]
    sig_fdr = stats_df[stats_df["Significant_MW_FDR_0.05"] == "Yes"]
    
    print(f"\nRaw Mann-Whitney p-values (uncorrected):")
    print(f"  - Significant populations: {len(sig_raw)}")
    if len(sig_raw) > 0:
        for _, row in sig_raw.iterrows():
            print(f"    • {row['Population']}: p={row['Mann-Whitney_pvalue']:.4f}")
    
    print(f"\nAfter FDR Correction (Benjamini-Hochberg):")
    print(f"  - Significant populations: {len(sig_fdr)}")
    if len(sig_fdr) > 0:
        for _, row in sig_fdr.iterrows():
            print(f"    • {row['Population']}: adjusted p={row['Mann-Whitney_pvalue_FDR']:.4f}")
    else:
        print("  - No populations remain significant after multiple testing correction")
    
    print(f"\nConclusion:")
    if len(sig_raw) > len(sig_fdr):
        print(f"  ⚠️  FDR correction reduced the number of potentially false positive findings by {len(sig_raw) - len(sig_fdr)}")
        print(f"  Of the {len(sig_raw)} initially significant populations, {len(sig_fdr)} remain significant after correction")
    elif len(sig_fdr) == len(sig_raw):
        print(f"  ✓ All {len(sig_fdr)} significant findings remain robust after correction")
    else:
        print(f"  ✓ {len(sig_fdr)} populations show significant differences accounting for multiple testing")
    
    print("\n" + "="*100 + "\n")


def main():
    """Main analysis function."""
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found. Run load_data.py first.")
        return

    print("Loading melanoma/miraclib/PBMC data...")
    df = load_responder_data()

    if df.empty:
        print("No data found matching criteria (melanoma, miraclib, PBMC, with response info)")
        return

    print(f"Loaded {len(df)} records from {df['sample_id'].nunique()} samples\n")

    # Perform statistical analysis
    print("Running statistical tests...")
    stats_df = statistical_analysis(df)

    # Generate report
    generate_report(df, stats_df)

    output_dir = Path(__file__).parent / "outputs" / "part3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save statistics to CSV
    stats_output = output_dir / "responder_analysis_statistics.csv"
    stats_df.to_csv(stats_output, index=False)
    print(f"✓ Statistics saved to {stats_output}")

    # Create boxplots with significance annotations
    print("\nGenerating boxplots...")
    plot_output = output_dir / "responder_analysis_boxplots.png"
    create_boxplots(df, output_path=plot_output)
    print(f"✓ Boxplots saved to {plot_output}")

    # Save full data to CSV
    export_output = output_dir / "responder_analysis_data.csv"
    df.to_csv(export_output, index=False)
    print(f"✓ Full data saved to {export_output}\n")

    print("✓ Analysis complete!")


if __name__ == "__main__":
    main()
