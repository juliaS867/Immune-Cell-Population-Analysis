#!/usr/bin/env python3
"""
Load cell-count.csv data into a normalized SQLite database.

This script creates the database schema and populates it with data from cell-count.csv.
Usage: python load_data.py
"""

import sqlite3
import csv
from pathlib import Path

# Database file path
DB_PATH = Path(__file__).parent / "cell_data.db"

def create_schema(conn):
    """Create the database schema."""
    cursor = conn.cursor()
    
    # Projects table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id TEXT PRIMARY KEY,
        project_name TEXT
    )
    """)
    
    # Subjects table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subjects (
        subject_id TEXT PRIMARY KEY,
        age INTEGER CHECK (age > 0 AND age < 150),
        sex TEXT CHECK (sex IN ('M', 'F')),
        response TEXT CHECK (response IN ('yes', 'no') OR response IS NULL)
    )
    """)
    
    # Samples table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS samples (
        sample_id TEXT PRIMARY KEY,
        subject_id TEXT NOT NULL,
        project_id TEXT NOT NULL,
        condition TEXT,
        sample_type TEXT,
        treatment TEXT,
        time_from_treatment INTEGER,
        FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
        FOREIGN KEY (project_id) REFERENCES projects(project_id)
    )
    """)
    
    # Cell types lookup table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cell_types (
        cell_type_id INTEGER PRIMARY KEY,
        cell_type_name TEXT UNIQUE NOT NULL
    )
    """)
    
    # Cell counts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cell_counts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id TEXT NOT NULL,
        cell_type_id INTEGER NOT NULL,
        count INTEGER NOT NULL,
        FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
        FOREIGN KEY (cell_type_id) REFERENCES cell_types(cell_type_id),
        UNIQUE (sample_id, cell_type_id)
    )
    """)
    
    # Cell population summaries (materialized view)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cell_population_summaries (
        summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id TEXT NOT NULL,
        cell_type_id INTEGER NOT NULL,
        total_count INTEGER NOT NULL,
        count INTEGER NOT NULL,
        percentage REAL NOT NULL,
        FOREIGN KEY (sample_id) REFERENCES samples(sample_id),
        FOREIGN KEY (cell_type_id) REFERENCES cell_types(cell_type_id),
        UNIQUE (sample_id, cell_type_id)
    )
    """)
    
    # Create indexes for analytical queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_subject_id ON samples(subject_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_project_id ON samples(project_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_treatment ON samples(treatment)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_condition ON samples(condition)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_sample_type ON samples(sample_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_timepoint ON samples(time_from_treatment)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cell_counts_sample_id ON cell_counts(sample_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cell_counts_cell_type ON cell_counts(cell_type_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_sample_id ON cell_population_summaries(sample_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_cell_type ON cell_population_summaries(cell_type_id)")
    
    conn.commit()


def seed_cell_types(conn):
    """Seed the cell_types table with predefined cell types."""
    cursor = conn.cursor()
    
    cell_types = [
        'b_cell',
        'cd8_t_cell',
        'cd4_t_cell',
        'nk_cell',
        'monocyte'
    ]
    
    for cell_type in cell_types:
        cursor.execute("INSERT OR IGNORE INTO cell_types (cell_type_name) VALUES (?)", (cell_type,))
    
    conn.commit()


def load_data(conn, csv_path):
    """Load data from CSV file into the database."""
    cursor = conn.cursor()
    
    # Track unique subjects to avoid duplicates
    subjects = {}
    projects = set()
    
    # Read CSV file
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # First pass: extract unique subjects and projects
    for row in rows:
        project = row['project']
        subject = row['subject']
        
        projects.add(project)
        
        if subject not in subjects:
            subjects[subject] = {
                'age': int(row['age']),
                'sex': row['sex'],
                'response': row['response'] if row['response'] in ('yes', 'no') else None
            }
    
    # Insert projects
    for project_id in projects:
        cursor.execute(
            "INSERT OR IGNORE INTO projects (project_id, project_name) VALUES (?, ?)",
            (project_id, project_id)  # Use project_id as name if not provided
        )
    
    # Insert subjects
    for subject_id, data in subjects.items():
        cursor.execute(
            "INSERT OR IGNORE INTO subjects (subject_id, age, sex, response) VALUES (?, ?, ?, ?)",
            (subject_id, data['age'], data['sex'], data['response'])
        )
    
    # Get cell_type_id mapping
    cursor.execute("SELECT cell_type_id, cell_type_name FROM cell_types")
    cell_type_map = {name: cell_id for cell_id, name in cursor.fetchall()}
    
    # Second pass: insert samples and cell counts
    for row in rows:
        sample_id = row['sample']
        subject_id = row['subject']
        project_id = row['project']
        condition = row['condition']
        sample_type = row['sample_type']
        treatment = row['treatment'] if row['treatment'] else None
        time_from_treatment = int(row['time_from_treatment_start']) if row['time_from_treatment_start'] else 0
        
        # Insert sample
        cursor.execute(
            """INSERT OR IGNORE INTO samples 
               (sample_id, subject_id, project_id, condition, sample_type, treatment, time_from_treatment)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sample_id, subject_id, project_id, condition, sample_type, treatment, time_from_treatment)
        )
        
        # Insert cell counts for this sample
        for cell_type_name in cell_type_map.keys():
            count = int(row[cell_type_name])
            cell_type_id = cell_type_map[cell_type_name]
            
            cursor.execute(
                "INSERT OR IGNORE INTO cell_counts (sample_id, cell_type_id, count) VALUES (?, ?, ?)",
                (sample_id, cell_type_id, count)
            )
    
    conn.commit()


def calculate_summaries(conn):
    """Calculate cell population summaries with total counts and percentages."""
    cursor = conn.cursor()
    
    # Clear existing summaries
    cursor.execute("DELETE FROM cell_population_summaries")
    
    # Calculate total counts per sample
    cursor.execute("""
    SELECT 
        cc.sample_id,
        cc.cell_type_id,
        cc.count,
        SUM(cc.count) OVER (PARTITION BY cc.sample_id) as total_count,
        ROUND(CAST(cc.count AS FLOAT) / SUM(cc.count) OVER (PARTITION BY cc.sample_id) * 100, 2) as percentage
    FROM cell_counts cc
    """)
    
    summaries = cursor.fetchall()
    
    # Insert summaries
    for sample_id, cell_type_id, count, total_count, percentage in summaries:
        cursor.execute(
            """INSERT INTO cell_population_summaries 
               (sample_id, cell_type_id, total_count, count, percentage)
               VALUES (?, ?, ?, ?, ?)""",
            (sample_id, cell_type_id, total_count, count, percentage)
        )
    
    conn.commit()


def main():
    """Main function to initialize and populate the database."""
    csv_path = Path(__file__).parent / "cell-count.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
    
    # Remove existing database if present
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing database: {DB_PATH}")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        print("Creating database schema...")
        create_schema(conn)
        
        print("Seeding cell types...")
        seed_cell_types(conn)
        
        print(f"Loading data from {csv_path}...")
        load_data(conn, csv_path)
        
        print("Calculating cell population summaries...")
        calculate_summaries(conn)
        
        print(f"✓ Database successfully created: {DB_PATH}")
        
        # Print summary statistics
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM projects")
        print(f"  - Projects: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM subjects")
        print(f"  - Subjects: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM samples")
        print(f"  - Samples: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM cell_types")
        print(f"  - Cell types: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM cell_counts")
        print(f"  - Cell count records: {cursor.fetchone()[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
