#!/usr/bin/env python3
"""
Part 2: Cell Type Frequency Analysis

Displays the relative frequency of each cell population in each sample.
For each sample, calculates total cell count and the percentage of each cell type.
"""

import sqlite3
from pathlib import Path
import pandas as pd

DB_PATH = Path(__file__).parent / "cell_data.db"


def get_frequency_summary():
    """Query the database and return frequency summary as a DataFrame."""
    query = """
    SELECT
        cps.sample_id as sample,
        cps.total_count,
        ct.cell_type_name as population,
        cps.count,
        cps.percentage
    FROM cell_population_summaries cps
    JOIN cell_types ct ON cps.cell_type_id = ct.cell_type_id
    ORDER BY cps.sample_id, ct.cell_type_name
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    return df


def display_summary(df, limit=10):
    """Display a preview of the summary table."""
    print("\n" + "="*100)
    print("CELL TYPE FREQUENCY SUMMARY PREVIEW")
    print("="*100 + "\n")
    
    print(f"Showing the first {limit} rows out of {len(df)} total rows.")
    print(df.head(limit).to_string(index=False))
    
    print("\n" + "="*100)
    print(f"Total rows: {len(df)}")
    print(f"Unique samples: {df['sample'].nunique()}")
    print(f"Unique populations: {df['population'].nunique()}")
    print("="*100 + "\n")


def save_to_csv(df, filename="frequency_summary.csv"):
    """Save the summary table to a CSV file."""
    output_path = Path(__file__).parent / "outputs" / "part2" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Summary saved to {output_path}\n")


def sample_statistics(df):
    """Display statistics by sample."""
    print("\n" + "="*100)
    print("STATISTICS BY SAMPLE")
    print("="*100 + "\n")
    
    sample_counts = df.groupby('sample').agg({
        'total_count': 'first',
        'population': 'count'
    }).rename(columns={'population': 'populations_tracked'})
    
    unique_counts = sorted(sample_counts['populations_tracked'].unique())
    print(f"Total unique samples: {len(sample_counts)}")
    print(f"Populations tracked per sample: {unique_counts}")
    if len(unique_counts) == 1:
        print(f"All samples have {unique_counts[0]} tracked populations.")
    
    # Sum of per-sample totals = every cell recorded across the entire dataset
    total_cells = sample_counts['total_count'].sum()
    print(f"Total cells across all samples: {total_cells}")


def population_statistics(df):
    """Display statistics by population."""
    print("\n" + "="*100)
    print("STATISTICS BY POPULATION")
    print("="*100 + "\n")
    
    pop_stats = df.groupby('population').agg({
        'count': ['min', 'mean', 'max'],
        'percentage': ['min', 'mean', 'max']
    }).round(2)
    
    pop_stats.columns = ['_'.join(col).strip() for col in pop_stats.columns.values]
    
    print(pop_stats.to_string())


def get_sample_details(sample_id, df):
    """Get details for a specific sample."""
    sample_df = df[df['sample'] == sample_id]
    
    if sample_df.empty:
        print(f"Sample {sample_id} not found in database.\n")
        return
    
    print(f"\n{'='*100}")
    print(f"DETAILS FOR SAMPLE: {sample_id}")
    print(f"{'='*100}\n")
    
    print(sample_df.to_string(index=False))
    print(f"\nTotal cell count: {sample_df['total_count'].iloc[0]}")
    print(f"Cell populations tracked: {len(sample_df)}")


def main():
    """Main function."""
    if not DB_PATH.exists():
        print(f"Error: {DB_PATH} not found. Run load_data.py first.")
        return
    
    print("Loading frequency summary from database...")
    df = get_frequency_summary()
    
    # Display a small preview and summary stats (CSV contains the full data)
    display_summary(df)
    
    # Save to CSV
    save_to_csv(df)
    
    # Display statistics
    sample_statistics(df)
    population_statistics(df)
    
    # Example: Show details for first 3 samples
    print("\n" + "="*100)
    print("EXAMPLE: FIRST 3 SAMPLES DETAILED VIEW")
    print("="*100)
    
    first_3_samples = df['sample'].unique()[:3]
    for sample_id in first_3_samples:
        get_sample_details(sample_id, df)
    
    print("\n✓ Analysis complete!")
    print(f"Use get_sample_details(sample_id, df) to inspect specific samples.\n")


if __name__ == "__main__":
    main()
