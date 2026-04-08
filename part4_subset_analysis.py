#!/usr/bin/env python3
"""
Part 4: Data Subset Analysis - Early Treatment Effects

Explore baseline samples (time_from_treatment_start = 0) from melanoma patients 
receiving miraclib treatment. Analyze sample distribution by project, 
responder status, and sex to understand treatment cohort composition.
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent / "cell_data.db"


def load_baseline_data() -> pd.DataFrame:
    """Load baseline melanoma/miraclib/PBMC samples."""
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


def analyze_by_project(df: pd.DataFrame) -> pd.DataFrame:
    """Count samples by project."""
    project_counts = df.groupby("project_id").agg({
        "sample_id": "count",
        "subject_id": "nunique"
    }).reset_index()
    
    project_counts.columns = ["Project", "Sample Count", "Unique Subjects"]
    return project_counts.sort_values("Sample Count", ascending=False)


def analyze_by_response(df: pd.DataFrame) -> pd.DataFrame:
    """Count subjects by response status."""
    # Get unique subjects only (remove duplicates from multiple samples per subject)
    unique_subjects = df[["subject_id", "response"]].drop_duplicates()
    
    response_counts = unique_subjects["response"].value_counts(dropna=False).reset_index()
    response_counts.columns = ["Response", "Subject Count"]
    
    # Calculate percentage
    total = response_counts["Subject Count"].sum()
    response_counts["Percentage"] = (response_counts["Subject Count"] / total * 100).round(1)
    
    return response_counts


def analyze_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Count subjects by sex."""
    # Get unique subjects only
    unique_subjects = df[["subject_id", "sex"]].drop_duplicates()
    
    sex_counts = unique_subjects["sex"].value_counts(dropna=False).reset_index()
    sex_counts.columns = ["Sex", "Subject Count"]
    
    # Calculate percentage
    total = sex_counts["Subject Count"].sum()
    sex_counts["Percentage"] = (sex_counts["Subject Count"] / total * 100).round(1)
    
    return sex_counts


def analyze_response_by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of response status by sex."""
    # Get unique subjects only
    unique_subjects = df[["subject_id", "response", "sex"]].drop_duplicates()
    
    crosstab = pd.crosstab(
        unique_subjects["sex"],
        unique_subjects["response"],
        margins=True,
        margins_name="Total",
        dropna=False
    )
    
    return crosstab


def analyze_by_project_and_response(df: pd.DataFrame) -> pd.DataFrame:
    """Count subjects by project and response status."""
    # Get unique subjects only
    unique_subjects = df[["subject_id", "project_id", "response"]].drop_duplicates()
    
    project_response = pd.crosstab(
        unique_subjects["project_id"],
        unique_subjects["response"],
        margins=True,
        margins_name="Total",
        dropna=False
    )
    
    return project_response


def generate_report(df: pd.DataFrame):
    """Generate a comprehensive report."""
    print("\n" + "="*100)
    print("PART 4: DATA SUBSET ANALYSIS - EARLY TREATMENT EFFECTS")
    print("="*100)
    print("\nStudy Parameters:")
    print("- Condition: Melanoma")
    print("- Treatment: Miraclib")
    print("- Sample Type: PBMC")
    print("- Time Point: Baseline (time_from_treatment_start = 0)")
    print("\n" + "="*100)
    
    # Overview
    total_samples = df["sample_id"].nunique()
    total_subjects = df["subject_id"].nunique()
    
    print(f"\nOVERVIEW:")
    print(f"  Total samples: {total_samples}")
    print(f"  Unique subjects: {total_subjects}")
    
    # By Project
    print("\n" + "="*100)
    print("SAMPLE DISTRIBUTION BY PROJECT:")
    print("="*100)
    project_analysis = analyze_by_project(df)
    print(project_analysis.to_string(index=False))
    
    # By Response
    print("\n" + "="*100)
    print("SUBJECT DISTRIBUTION BY RESPONSE STATUS:")
    print("="*100)
    response_analysis = analyze_by_response(df)
    print(response_analysis.to_string(index=False))
    
    # By Sex
    print("\n" + "="*100)
    print("SUBJECT DISTRIBUTION BY SEX:")
    print("="*100)
    sex_analysis = analyze_by_sex(df)
    print(sex_analysis.to_string(index=False))
    
    # Cross-tabulation: Sex vs Response
    print("\n" + "="*100)
    print("SEX vs RESPONSE STATUS CROSS-TABULATION:")
    print("="*100)
    sex_response = analyze_response_by_sex(df)
    print(sex_response)
    
    # By Project and Response
    print("\n" + "="*100)
    print("SUBJECTS BY PROJECT AND RESPONSE STATUS:")
    print("="*100)
    project_response = analyze_by_project_and_response(df)
    print(project_response)
    
    return project_analysis, response_analysis, sex_analysis, sex_response, project_response


def main():
    """Main analysis function."""
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found. Run load_data.py first.")
        return

    print("Loading baseline melanoma/miraclib/PBMC samples...")
    df = load_baseline_data()

    if df.empty:
        print("No data found matching criteria (melanoma, miraclib, PBMC, time_from_treatment=0)")
        return

    print(f"Loaded {len(df)} records from {df['sample_id'].nunique()} samples\n")

    # Generate report
    project_analysis, response_analysis, sex_analysis, sex_response, project_response = generate_report(df)

    # Save analyses to CSV
    output_dir = Path(__file__).parent / "outputs" / "part4"
    output_dir.mkdir(parents=True, exist_ok=True)

    project_analysis.to_csv(output_dir / "samples_by_project.csv", index=False)
    print(f"\n✓ Samples by project saved to {output_dir / 'samples_by_project.csv'}")

    response_analysis.to_csv(output_dir / "subjects_by_response.csv", index=False)
    print(f"✓ Subjects by response saved to {output_dir / 'subjects_by_response.csv'}")

    sex_analysis.to_csv(output_dir / "subjects_by_sex.csv", index=False)
    print(f"✓ Subjects by sex saved to {output_dir / 'subjects_by_sex.csv'}")

    sex_response.to_csv(output_dir / "sex_vs_response_crosstab.csv")
    print(f"✓ Sex vs response cross-tabulation saved to {output_dir / 'sex_vs_response_crosstab.csv'}")

    project_response.to_csv(output_dir / "project_vs_response_crosstab.csv")
    print(f"✓ Project vs response cross-tabulation saved to {output_dir / 'project_vs_response_crosstab.csv'}")

    df.to_csv(output_dir / "baseline_samples_full.csv", index=False)
    print(f"✓ Full baseline dataset saved to {output_dir / 'baseline_samples_full.csv'}\n")

    print("✓ Analysis complete!")


if __name__ == "__main__":
    main()
