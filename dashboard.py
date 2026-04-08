#!/usr/bin/env python3
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests

DB_PATH = Path(__file__).parent / "cell_data.db"


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    query = """
    SELECT
        cps.sample_id AS sample,
        cps.total_count,
        ct.cell_type_name AS population,
        cps.count,
        cps.percentage,
        s.project_id,
        s.condition,
        s.sample_type,
        s.treatment,
        s.time_from_treatment,
        subj.subject_id,
        subj.age,
        subj.sex,
        subj.response
    FROM cell_population_summaries cps
    JOIN cell_types ct ON cps.cell_type_id = ct.cell_type_id
    JOIN samples s ON cps.sample_id = s.sample_id
    JOIN subjects subj ON s.subject_id = subj.subject_id
    ORDER BY cps.sample_id, ct.cell_type_name
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df


@st.cache_data
def load_responder_data() -> pd.DataFrame:
    """Load melanoma/miraclib/PBMC data for responder analysis."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    query = """
    SELECT
        cps.sample_id,
        ct.cell_type_name AS population,
        cps.percentage,
        s.condition,
        s.treatment,
        s.sample_type,
        subj.response,
        subj.subject_id
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


@st.cache_data
def load_baseline_data() -> pd.DataFrame:
    """Load baseline melanoma/miraclib/PBMC samples (time_from_treatment = 0)."""
    if not DB_PATH.exists():
        return pd.DataFrame()

    query = """
    SELECT
        s.sample_id,
        s.project_id,
        s.condition,
        s.treatment,
        s.sample_type,
        s.time_from_treatment,
        subj.subject_id,
        subj.response,
        subj.sex,
        subj.age
    FROM samples s
    JOIN subjects subj ON s.subject_id = subj.subject_id
    WHERE s.condition = 'melanoma'
      AND s.treatment = 'miraclib'
      AND s.sample_type = 'PBMC'
      AND s.time_from_treatment = 0
    ORDER BY s.project_id, s.sample_id
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters for Frequency Analysis")

    project_ids = df["project_id"].dropna().unique().tolist()
    selected_projects = st.sidebar.multiselect("Project", sorted(project_ids), default=sorted(project_ids))

    conditions = df["condition"].dropna().unique().tolist()
    selected_conditions = st.sidebar.multiselect("Condition", sorted(conditions), default=sorted(conditions))

    treatments = df["treatment"].fillna("None").unique().tolist()
    selected_treatments = st.sidebar.multiselect("Treatment", sorted(treatments), default=sorted(treatments))

    responses = df["response"].fillna("Unknown").unique().tolist()
    selected_responses = st.sidebar.multiselect("Response", sorted(responses), default=sorted(responses))

    sample_type = st.sidebar.multiselect("Sample Type", sorted(df["sample_type"].dropna().unique().tolist()), default=sorted(df["sample_type"].dropna().unique().tolist()))

    sexes = df["sex"].dropna().unique().tolist()
    selected_sexes = st.sidebar.multiselect("Sex", sorted(sexes), default=sorted(sexes))

    timepoints = sorted(df["time_from_treatment"].dropna().unique().tolist())
    selected_timepoints = st.sidebar.multiselect("Time from Treatment (days)", timepoints, default=timepoints)

    population_query = st.sidebar.multiselect("Population", sorted(df["population"].unique().tolist()), default=sorted(df["population"].unique().tolist()))

    filtered = df[
        df["project_id"].isin(selected_projects)
        & df["condition"].isin(selected_conditions)
        & df["treatment"].fillna("None").isin(selected_treatments)
        & df["response"].fillna("Unknown").isin(selected_responses)
        & df["sample_type"].isin(sample_type)
        & df["sex"].isin(selected_sexes)
        & df["time_from_treatment"].isin(selected_timepoints)
        & df["population"].isin(population_query)
    ]

    return filtered


def main():
    st.set_page_config(page_title="Immune Cell Population Analysis", layout="wide")
    st.title("Immune Cell Population Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Frequency Analysis",
        "Responder Analysis",
        "Baseline Cohort Analysis",
        "Time-Course & Subject Explorer",
    ])

    # ------------------------------------------------------------------
    # TAB 1: Frequency Analysis
    # ------------------------------------------------------------------
    with tab1:
        st.markdown(
            """
            This interactive dashboard displays the relative frequency of each immune cell type in every sample.
            Use the filters in the sidebar to narrow the data by project, condition, treatment, response, and sample metadata.
            """
        )

        df = load_data()
        if df.empty:
            st.error("Database file `cell_data.db` not found. Run `python load_data.py` first.")
            return

        filtered_df = filter_dataframe(df)

        total_samples = df["sample"].nunique()
        filtered_samples = filtered_df["sample"].nunique()
        total_rows = len(df)
        filtered_rows = len(filtered_df)

        st.metric(f"Samples in view out of {total_samples}", filtered_samples)
        st.metric(f"Rows in view out of {total_rows}", filtered_rows)

        st.subheader("Summary table")
        if len(filtered_df) > 200:
            st.warning(
                f"Showing 200 of {len(filtered_df):,} rows. Download the CSV below to access the full dataset."
            )
        st.dataframe(filtered_df.head(200), use_container_width=True)

        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered data as CSV", csv, "filtered_frequency_summary.csv", "text/csv")

        st.subheader("Population mean count and percentage")
        population_agg = (
            filtered_df.groupby("population")
            .agg(mean_count=("count", "mean"), mean_percentage=("percentage", "mean"))
            .round(2)
            .sort_values("mean_percentage", ascending=False)
            .reset_index()
        )
        st.dataframe(population_agg, use_container_width=True)

        fig_pop, ax_pop = plt.subplots(figsize=(8, 4))
        ax_pop.bar(population_agg["population"], population_agg["mean_percentage"], color="#3498db")
        ax_pop.set_xlabel("Cell Population")
        ax_pop.set_ylabel("Mean Percentage (%)")
        ax_pop.set_title("Mean Cell Type Percentage Across Filtered Samples")
        ax_pop.tick_params(axis="x", rotation=15)
        st.pyplot(fig_pop, use_container_width=True)
        plt.close(fig_pop)

        if filtered_df["sample"].nunique() == 1:
            st.subheader("Sample frequency breakdown")
            single_sample = filtered_df.iloc[0]["sample"]
            st.write(f"Detailed frequency for sample **{single_sample}**")
            st.bar_chart(filtered_df.set_index("population")["percentage"])

        st.subheader("Aggregated heatmap: Mean percentage by condition/treatment")
        heatmap_data = (
            filtered_df.groupby(["condition", "treatment", "population"])["percentage"]
            .mean()
            .reset_index()
        )
        heatmap_data["group"] = heatmap_data["condition"] + " (" + heatmap_data["treatment"].fillna("none") + ")"
        heatmap_pivot = heatmap_data.pivot_table(
            index="group", columns="population", values="percentage", aggfunc="mean"
        )

        if not heatmap_pivot.empty:
            fig_hm, ax_hm = plt.subplots(figsize=(10, max(4, len(heatmap_pivot) * 0.6)))
            sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Mean %"}, ax=ax_hm)
            ax_hm.set_title("Mean Cell Type Percentage by Condition/Treatment")
            ax_hm.set_xlabel("Cell Type")
            ax_hm.set_ylabel("Condition (Treatment)")
            st.pyplot(fig_hm, use_container_width=True)
            plt.close(fig_hm)
        else:
            st.info("No data available for heatmap with current filters.")

    # ------------------------------------------------------------------
    # TAB 2: Responder Analysis
    # ------------------------------------------------------------------
    with tab2:
        st.markdown(
            """
            ## Responder vs Non-Responder Analysis

            Compare cell population frequencies in melanoma patients receiving miraclib who respond versus
            those who do not. Includes statistical tests to identify significant biomarkers predictive of
            treatment response.

            **Note:** This tab uses a fixed dataset (Melanoma + Miraclib treatment, PBMC samples only).
            The sidebar filters from Tab 1 do not apply here — all eligible samples are included.
            """
        )

        responder_df = load_responder_data()

        if responder_df.empty:
            st.error("No data available for responder analysis. Need melanoma PBMC samples from miraclib treatment.")
            return

        total_resp_samples = responder_df["sample_id"].nunique()
        resp_yes = responder_df[responder_df["response"] == "yes"]["sample_id"].nunique()
        resp_no = responder_df[responder_df["response"] == "no"]["sample_id"].nunique()
        response_rate = (resp_yes / total_resp_samples * 100) if total_resp_samples > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", total_resp_samples)
        with col2:
            st.metric("Responders (yes)", resp_yes)
        with col3:
            st.metric("Non-Responders (no)", resp_no)
        with col4:
            st.metric("Response Rate", f"{response_rate:.1f}%")

        populations = sorted(responder_df["population"].unique())

        # Precompute all statistics including FDR before drawing boxplots,
        # so boxplot annotations can use FDR-corrected p-values.
        results = []
        for population in populations:
            pop_data = responder_df[responder_df["population"] == population]
            resp_vals = pop_data[pop_data["response"] == "yes"]["percentage"].values
            non_resp_vals = pop_data[pop_data["response"] == "no"]["percentage"].values

            if len(resp_vals) < 3 or len(non_resp_vals) < 3:
                continue

            t_stat, t_pvalue = stats.ttest_ind(resp_vals, non_resp_vals, equal_var=False)
            u_stat, u_pvalue = stats.mannwhitneyu(resp_vals, non_resp_vals, alternative="two-sided")

            pooled_std = np.sqrt(
                ((len(resp_vals) - 1) * np.std(resp_vals, ddof=1) ** 2 +
                 (len(non_resp_vals) - 1) * np.std(non_resp_vals, ddof=1) ** 2) /
                (len(resp_vals) + len(non_resp_vals) - 2)
            )
            cohens_d = (np.mean(resp_vals) - np.mean(non_resp_vals)) / pooled_std if pooled_std > 0 else 0
            n1, n2 = len(resp_vals), len(non_resp_vals)
            rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

            results.append({
                "Population": population,
                "Responders (n)": n1,
                "Non-Responders (n)": n2,
                "Responders Mean %": np.mean(resp_vals),
                "Non-Responders Mean %": np.mean(non_resp_vals),
                "Difference %": np.mean(resp_vals) - np.mean(non_resp_vals),
                "Mann-Whitney p-value (raw)": u_pvalue,
                "Welch t-test p-value": t_pvalue,
                "Rank-biserial r": rank_biserial,
                "Cohen's d": cohens_d,
            })

        # Apply FDR correction and build lookup for boxplot annotations
        fdr_lookup = {}
        if results:
            mw_pvalues = [r["Mann-Whitney p-value (raw)"] for r in results]
            reject_mw, fdr_pvalues_mw, _, _ = multipletests(mw_pvalues, alpha=0.05, method="fdr_bh")
            for i, result in enumerate(results):
                result["Mann-Whitney p-value (FDR)"] = fdr_pvalues_mw[i]
                result["Significant Raw (p<0.05)"] = "✓ Yes" if result["Mann-Whitney p-value (raw)"] < 0.05 else "No"
                result["Significant FDR (p<0.05)"] = "✓ Yes" if fdr_pvalues_mw[i] < 0.05 else "No"
                fdr_lookup[result["Population"]] = fdr_pvalues_mw[i]

        # Boxplots annotated with FDR-corrected p-values
        st.subheader("Cell Population Comparison: Responders vs Non-Responders")
        st.caption("Significance brackets show FDR-corrected Mann-Whitney p-values.")

        n = len(populations)
        n_cols = min(4, n)
        n_rows = int(np.ceil(n / n_cols))
        fig_bp, axes_bp = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        if n == 1:
            axes_bp = np.array([axes_bp])
        else:
            axes_bp = axes_bp.flatten()

        for idx, population in enumerate(populations):
            pop_data = responder_df[responder_df["population"] == population]
            resp_vals = pop_data[pop_data["response"] == "yes"]["percentage"].values
            non_resp_vals = pop_data[pop_data["response"] == "no"]["percentage"].values

            ax = axes_bp[idx]
            bp = ax.boxplot(
                [resp_vals, non_resp_vals],
                labels=["Responders", "Non-Responders"],
                patch_artist=True,
            )
            colors = ["#2ecc71", "#e74c3c"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            fdr_p = fdr_lookup.get(population)
            if fdr_p is not None and len(resp_vals) >= 3 and len(non_resp_vals) >= 3:
                if fdr_p < 0.001:
                    sig_label = f"p={fdr_p:.2e} ***"
                elif fdr_p < 0.01:
                    sig_label = f"p={fdr_p:.3f} **"
                elif fdr_p < 0.05:
                    sig_label = f"p={fdr_p:.3f} *"
                else:
                    sig_label = f"p={fdr_p:.3f} ns"

                all_vals = np.concatenate([resp_vals, non_resp_vals])
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

        for idx in range(n, len(axes_bp)):
            axes_bp[idx].set_visible(False)

        fig_bp.suptitle(
            "Cell Population Frequencies: Responders vs Non-Responders (PBMC, Melanoma + Miraclib)",
            fontsize=14, y=1.00,
        )
        plt.tight_layout()
        st.pyplot(fig_bp, use_container_width=True)
        plt.close(fig_bp)

        # Statistical results table
        st.subheader("Statistical Test Results: Before and After FDR Correction")

        stats_df = pd.DataFrame(results).sort_values("Mann-Whitney p-value (raw)")

        sig_raw = sum(1 for r in results if r["Mann-Whitney p-value (raw)"] < 0.05)
        sig_fdr = sum(1 for r in results if r.get("Mann-Whitney p-value (FDR)", 1) < 0.05)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Raw Significant (p<0.05)", sig_raw)
        with col2:
            st.metric("FDR Significant (p<0.05)", sig_fdr)
        with col3:
            if sig_raw > sig_fdr:
                st.warning(f"FDR correction removed {sig_raw - sig_fdr} potentially false positive finding(s)")
            elif sig_fdr > 0:
                st.success(f"All {sig_fdr} finding(s) robust after FDR correction")
            else:
                st.info("No significant findings after FDR correction")

        display_df = stats_df.copy()
        for col in ["Responders Mean %", "Non-Responders Mean %", "Difference %"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
        for col in ["Mann-Whitney p-value (raw)", "Mann-Whitney p-value (FDR)", "Welch t-test p-value"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        display_df["Rank-biserial r"] = display_df["Rank-biserial r"].apply(lambda x: f"{x:.3f}")
        display_df["Cohen's d"] = display_df["Cohen's d"].apply(lambda x: f"{x:.3f}")
        st.dataframe(display_df, use_container_width=True)

        # Key findings
        st.subheader("Key Findings: After Multiple Testing Correction")
        sig_populations = [r for r in results if r.get("Mann-Whitney p-value (FDR)", 1) < 0.05]

        if sig_populations:
            st.success(f"✓ Found {len(sig_populations)} population(s) with statistically significant differences after FDR correction")
            for result in sig_populations:
                direction = "higher" if result["Difference %"] > 0 else "lower"
                st.write(
                    f"• **{result['Population'].replace('_', ' ').title()}**: {direction} in responders "
                    f"({result['Responders Mean %']:.1f}% vs {result['Non-Responders Mean %']:.1f}%, "
                    f"FDR-adjusted p={result['Mann-Whitney p-value (FDR)']:.4f})"
                )
        else:
            st.info("After multiple testing correction (FDR), no cell populations showed statistically significant differences at p < 0.05")

        # Downloads
        st.subheader("Downloads")
        col1, col2 = st.columns(2)
        with col1:
            csv_stats = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download statistics as CSV", csv_stats, "responder_statistics.csv", "text/csv")
        with col2:
            csv_raw = responder_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download raw analysis data as CSV", csv_raw, "responder_raw_data.csv", "text/csv")

    # ------------------------------------------------------------------
    # TAB 3: Baseline Cohort Analysis
    # ------------------------------------------------------------------
    with tab3:
        st.markdown(
            """
            ## Baseline Cohort Analysis - Early Treatment Effects

            Explore the composition of melanoma patients receiving miraclib at baseline (time_from_treatment_start = 0).
            This analysis helps understand the treatment cohort in terms of project distribution,
            responder composition, and demographic characteristics.

            **Dataset:** Melanoma + Miraclib treatment, PBMC samples at baseline only
            """
        )

        baseline_df = load_baseline_data()

        if baseline_df.empty:
            st.error("No baseline samples found for melanoma + miraclib + PBMC.")
            return

        st.subheader("Cohort Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Baseline Samples", baseline_df["sample_id"].nunique())
        with col2:
            st.metric("Unique Subjects", baseline_df["subject_id"].nunique())
        with col3:
            st.metric("Treatment", "Miraclib")

        # Samples by Project
        st.subheader("Sample Distribution by Project")
        project_counts = baseline_df.groupby("project_id").agg(
            {"sample_id": "count", "subject_id": "nunique"}
        ).reset_index()
        project_counts.columns = ["Project", "Sample Count", "Unique Subjects"]
        project_counts = project_counts.sort_values("Sample Count", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(project_counts, use_container_width=True)
        with col2:
            st.bar_chart(project_counts.set_index("Project")["Sample Count"])

        # Subjects by Response Status
        st.subheader("Subject Distribution by Response Status")
        unique_subj_resp = baseline_df[["subject_id", "response"]].drop_duplicates()
        response_counts = unique_subj_resp["response"].value_counts(dropna=False).reset_index()
        response_counts.columns = ["Response", "Subject Count"]
        response_counts["Percentage"] = (
            response_counts["Subject Count"] / response_counts["Subject Count"].sum() * 100
        ).round(1)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(response_counts, use_container_width=True)
        with col2:
            st.bar_chart(response_counts.set_index("Response")["Subject Count"])

        # Subjects by Sex
        st.subheader("Subject Distribution by Sex")
        unique_subj_sex = baseline_df[["subject_id", "sex"]].drop_duplicates()
        sex_counts = unique_subj_sex["sex"].value_counts(dropna=False).reset_index()
        sex_counts.columns = ["Sex", "Subject Count"]
        sex_counts["Percentage"] = (
            sex_counts["Subject Count"] / sex_counts["Subject Count"].sum() * 100
        ).round(1)

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(sex_counts, use_container_width=True)
        with col2:
            st.bar_chart(sex_counts.set_index("Sex")["Subject Count"])

        # Age Distribution
        st.subheader("Age Distribution")
        unique_subj_age = baseline_df[["subject_id", "age", "response"]].drop_duplicates()

        col1, col2 = st.columns(2)
        with col1:
            age_stats = (
                unique_subj_age.groupby("response")["age"]
                .agg(N="count", Min="min", Median="median", Mean="mean", Max="max")
                .round(1)
            )
            st.caption("Age summary by response group")
            st.dataframe(age_stats, use_container_width=True)
        with col2:
            fig_age, ax_age = plt.subplots(figsize=(6, 4))
            for response_val, group in unique_subj_age.groupby("response"):
                ax_age.hist(group["age"], bins=10, alpha=0.6, label=str(response_val))
            ax_age.set_xlabel("Age")
            ax_age.set_ylabel("Number of Subjects")
            ax_age.set_title("Age Distribution by Response")
            ax_age.legend(title="Response")
            st.pyplot(fig_age, use_container_width=True)
            plt.close(fig_age)

        # Cross-tabulations
        st.subheader("Sex vs Response Status")
        unique_subj_cross = baseline_df[["subject_id", "response", "sex"]].drop_duplicates()
        sex_response_crosstab = pd.crosstab(
            unique_subj_cross["sex"],
            unique_subj_cross["response"],
            margins=True,
            margins_name="Total",
            dropna=False,
        )
        st.dataframe(sex_response_crosstab, use_container_width=True)

        st.subheader("Project vs Response Status")
        unique_subj_proj = baseline_df[["subject_id", "project_id", "response"]].drop_duplicates()
        project_response_crosstab = pd.crosstab(
            unique_subj_proj["project_id"],
            unique_subj_proj["response"],
            margins=True,
            margins_name="Total",
            dropna=False,
        )
        st.dataframe(project_response_crosstab, use_container_width=True)

        # Detailed table
        st.subheader("Detailed Sample Information")
        detail_df = (
            baseline_df[["sample_id", "subject_id", "project_id", "age", "sex", "response"]]
            .drop_duplicates()
            .sort_values(["project_id", "subject_id"])
        )
        st.dataframe(detail_df, use_container_width=True)

        csv_baseline = detail_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download baseline cohort details as CSV", csv_baseline, "baseline_cohort_details.csv", "text/csv")

    # ------------------------------------------------------------------
    # TAB 4: Time-Course & Subject Explorer
    # ------------------------------------------------------------------
    with tab4:
        st.markdown(
            """
            ## Time-Course & Subject Explorer

            Visualize how immune cell populations change over time in response to treatment,
            and drill down into individual subjects' trajectories.
            """
        )

        full_df = load_data()
        if full_df.empty:
            st.error("Database file `cell_data.db` not found. Run `python load_data.py` first.")
            return

        # --- Section 1: Population time-course ---
        st.subheader("Population Time-Course")
        st.caption("Mean cell type percentage at each timepoint, stratified by response group.")

        col1, col2, col3 = st.columns(3)
        with col1:
            tc_conditions = sorted(full_df["condition"].dropna().unique().tolist())
            selected_tc_condition = st.selectbox("Condition", tc_conditions, key="tc_condition")
        with col2:
            tc_treatments = sorted(
                full_df[full_df["condition"] == selected_tc_condition]["treatment"]
                .fillna("None").unique().tolist()
            )
            selected_tc_treatment = st.selectbox("Treatment", tc_treatments, key="tc_treatment")
        with col3:
            tc_sample_types = sorted(full_df["sample_type"].dropna().unique().tolist())
            selected_tc_sample_type = st.selectbox("Sample Type", tc_sample_types, key="tc_sample_type")

        tc_df = full_df[
            (full_df["condition"] == selected_tc_condition)
            & (full_df["treatment"].fillna("None") == selected_tc_treatment)
            & (full_df["sample_type"] == selected_tc_sample_type)
        ]

        if tc_df.empty:
            st.info("No data for the selected combination.")
        else:
            tc_grouped = (
                tc_df.groupby(["time_from_treatment", "population", "response"])["percentage"]
                .mean()
                .reset_index()
            )

            populations_tc = sorted(tc_df["population"].unique())
            n = len(populations_tc)
            n_cols = min(3, n)
            n_rows = int(np.ceil(n / n_cols))
            fig_tc, axes_tc = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
            axes_tc = axes_tc.flatten()

            response_colors = {"yes": "#2ecc71", "no": "#e74c3c"}

            for idx, population in enumerate(populations_tc):
                ax = axes_tc[idx]
                pop_tc = tc_grouped[tc_grouped["population"] == population]

                for response_val, group in pop_tc.groupby("response"):
                    color = response_colors.get(str(response_val), "#95a5a6")
                    label = f"Responder ({response_val})" if response_val in ("yes", "no") else str(response_val)
                    ax.plot(
                        group["time_from_treatment"], group["percentage"],
                        marker="o", label=label, color=color,
                    )

                ax.set_xlabel("Days from Treatment Start")
                ax.set_ylabel("Mean Percentage (%)")
                ax.set_title(population.replace("_", " ").title())
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

            for idx in range(n, len(axes_tc)):
                axes_tc[idx].set_visible(False)

            fig_tc.suptitle(
                f"Cell Population Time-Course: {selected_tc_condition.title()} / {selected_tc_treatment}",
                fontsize=13, y=1.01,
            )
            plt.tight_layout()
            st.pyplot(fig_tc, use_container_width=True)
            plt.close(fig_tc)

        st.divider()

        # --- Section 2: Subject explorer ---
        st.subheader("Subject Explorer")
        st.caption("Select a subject to view their cell population percentages across all collected timepoints.")

        subject_ids = sorted(full_df["subject_id"].unique().tolist())
        selected_subject = st.selectbox("Subject ID", subject_ids, key="subject_select")

        subj_df = full_df[full_df["subject_id"] == selected_subject].copy()

        if not subj_df.empty:
            meta = subj_df.iloc[0]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", int(meta["age"]))
            with col2:
                st.metric("Sex", meta["sex"])
            with col3:
                st.metric("Response", meta["response"] if pd.notna(meta["response"]) else "N/A")
            with col4:
                st.metric("Condition", meta["condition"])

            populations_subj = sorted(subj_df["population"].unique())
            n = len(populations_subj)
            n_cols = min(3, n)
            n_rows = int(np.ceil(n / n_cols))
            fig_subj, axes_subj = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
            axes_subj = axes_subj.flatten()

            for idx, population in enumerate(populations_subj):
                ax = axes_subj[idx]
                pop_subj = subj_df[subj_df["population"] == population].sort_values("time_from_treatment")

                for stype, group in pop_subj.groupby("sample_type"):
                    ax.plot(group["time_from_treatment"], group["percentage"], marker="o", label=stype)

                ax.set_xlabel("Days from Treatment Start")
                ax.set_ylabel("Percentage (%)")
                ax.set_title(population.replace("_", " ").title())
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

            for idx in range(n, len(axes_subj)):
                axes_subj[idx].set_visible(False)

            response_label = (
                "Responder" if meta["response"] == "yes"
                else "Non-Responder" if meta["response"] == "no"
                else "N/A"
            )
            fig_subj.suptitle(
                f"Subject {selected_subject}: Cell Populations Over Time ({response_label})",
                fontsize=13, y=1.01,
            )
            plt.tight_layout()
            st.pyplot(fig_subj, use_container_width=True)
            plt.close(fig_subj)

            with st.expander("View raw data for this subject"):
                subj_table = (
                    subj_df[["sample", "time_from_treatment", "population", "count", "percentage", "sample_type", "project_id"]]
                    .sort_values(["time_from_treatment", "population"])
                )
                st.dataframe(subj_table, use_container_width=True)


if __name__ == "__main__":
    main()
