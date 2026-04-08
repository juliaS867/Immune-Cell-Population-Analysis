# Immune Cell Population Analysis

## Running the Project

### Requirements
- Python 3.7+
- `make`

### Quick Start

```bash
make setup      # Install dependencies
make pipeline   # Run the full data pipeline
make dashboard  # Launch the interactive dashboard
```

When running in GitHub Codespaces, `make dashboard` will start the Streamlit server on port 8501. Codespaces will automatically prompt you to open the forwarded port in your browser.

### What each command does

| Command | What it runs |
|---|---|
| `make setup` | `pip install -r requirements.txt` |
| `make pipeline` | `load_data.py` → `part2_frequency_analysis.py` → `part3_responder_analysis.py` → `part4_subset_analysis.py` |
| `make dashboard` | `streamlit run dashboard.py` |

### Output files

Running `make pipeline` generates the following outputs:

```
outputs/
├── part2/
│   └── frequency_summary.csv
├── part3/
│   ├── responder_analysis_statistics.csv
│   ├── responder_analysis_data.csv
│   └── responder_analysis_boxplots.png
└── part4/
    ├── samples_by_project.csv
    ├── subjects_by_response.csv
    ├── subjects_by_sex.csv
    ├── sex_vs_response_crosstab.csv
    ├── project_vs_response_crosstab.csv
    └── baseline_samples_full.csv
```

---

## Dashboard

**Live link:** https://immune-cell-population-analysis-ewujkv6wo6clzrgwynaq2b.streamlit.app/

To run locally: `make dashboard` then open `http://localhost:8501`.

The dashboard has four tabs:
- **Frequency Analysis** — filterable summary table and aggregations across all samples
- **Responder Analysis** — statistical comparison (Mann-Whitney + FDR correction) of cell populations between responders and non-responders
- **Baseline Cohort Analysis** — cohort characterisation at time=0 (project, response, sex, age breakdowns)
- **Time-Course & Subject Explorer** — time-course plots by condition/treatment and per-subject trajectory viewer

---

## Database Schema

### Tables

```
projects        (project_id PK, project_name)
subjects        (subject_id PK, age, sex, response)
samples         (sample_id PK, subject_id FK, project_id FK, condition,
                 sample_type, treatment, time_from_treatment)
cell_types      (cell_type_id PK, cell_type_name UNIQUE)
cell_counts     (id PK, sample_id FK, cell_type_id FK, count,
                 UNIQUE(sample_id, cell_type_id))
cell_population_summaries  (summary_id PK, sample_id FK, cell_type_id FK,
                             total_count, count, percentage,
                             UNIQUE(sample_id, cell_type_id))
```

### Rationale

**Normalisation.** The schema is in third normal form. Subject demographics (age, sex, response) are stored once per subject rather than repeated across every sample row, and cell type names are stored once in a lookup table rather than as string columns. This eliminates update anomalies and redundancy.

**Separation of entities.** The four core entities (projects, subjects, samples, and cell types) each have their own table with a clear primary key. Samples link to both subjects and projects via foreign keys, reflecting the real-world relationship that one subject can have multiple samples across multiple projects.

**Precomputed summaries.** `cell_population_summaries` stores the total count and percentage for each (sample, cell type) pair. These values are derived from `cell_counts` but are expensive to recompute on every query (requiring a window function over all rows for a sample). Precomputing them once at load time makes all downstream analytical queries simple and fast.

**Indexes.** Indexes are created on the most commonly filtered columns: `subject_id`, `project_id`, `treatment`, `condition`, `sample_type`, and `time_from_treatment` on the samples table, and `sample_id` and `cell_type_id` on the counts and summaries tables. This ensures filter-heavy analytical queries (e.g. "melanoma + miraclib + PBMC + time=0") remain fast as data grows.

### Scalability

With hundreds of projects, thousands of samples, and a growing set of analytics needs, this schema scales well in several ways:

- **New cell populations** can be added by inserting a row into `cell_types` and running the loader with no schema changes required. The existing five populations are not hardcoded as columns.
- **New sample metadata** (e.g. batch ID, site, collection date) can be added as columns to `samples` without affecting other tables.
- **New subject attributes** (e.g. prior treatments) can be added to `subjects` in the same way.
- **Analytical queries** across thousands of samples remain performant because they target the pre-aggregated `cell_population_summaries` table with indexed joins rather than scanning raw counts.
- **Migration to a production database.** The normalised structure maps directly to other relational databases. For very large datasets, `cell_counts` and `cell_population_summaries` could be partitioned by `project_id` or stored in a columnar format (e.g. Parquet) to support bulk analytical workloads. The `cell_population_summaries` table could be replaced with a true materialised view with automatic refresh on insert.

---

## Code Structure

```
.
├── load_data.py                  # Part 1: schema creation, data loading, summary precomputation
├── part2_frequency_analysis.py   # Part 2: per-sample cell type frequency table
├── part3_responder_analysis.py   # Part 3: responder vs non-responder statistical analysis
├── part4_subset_analysis.py      # Part 4: baseline cohort subset analysis
├── dashboard.py                  # Interactive Streamlit dashboard (all four parts)
├── verify_db.py                  # Utility: inspect database contents after loading
├── cell-count.csv                # Input data
├── cell_data.db                  # SQLite database (generated by load_data.py)
├── requirements.txt
├── Makefile
└── outputs/
    ├── part2/
    ├── part3/
    └── part4/
```

### Design decisions

**Database as the single source of truth.** All scripts and the dashboard read from `cell_data.db` rather than from the CSV directly. This means the CSV is parsed and validated exactly once (in `load_data.py`), and all downstream code benefits from indexed SQL queries rather than in-memory pandas filtering on the raw file.

**Precomputed percentages at load time.** Rather than computing relative frequencies on every query, `load_data.py` calculates them once and stores them in `cell_population_summaries`. Every analytical script and the dashboard then queries this table directly, keeping query logic simple and consistent across all parts.


**Highly customisable frequency analysis.** The Frequency Analysis tab exposes sidebar filters for every relevant dimension (project, condition, treatment, response, sample type, sex, time from treatment, and cell population) so a researcher can slice the data to any combination of interest without writing a query. The summary table, aggregation table, bar chart, and heatmap all update live with the filters, and a sample-level breakdown appears automatically when the view is narrowed to a single sample.

**Time-course and subject-level explorer.** I added a dedicated Time-Course & Subject Explorer tab that was beyond the core requirements to support the kind of questions that naturally arise in a clinical trial like for example, how does a cell population trend over treatment time for responders versus non-responders, or what does an individual patient's immune profile look like across their collected timepoints? Researchers can select any condition, treatment, and sample type to view population trajectories over time, and drill into any individual subject to see their full series in one chart.

**Researcher-friendly UI.** Throughout the dashboard, results are designed to be immediately usable without needing to rerun code. Every tab includes download buttons so filtered tables and analysis outputs can be exported to CSV directly from the browser. Raw data underlying the responder analysis is downloadable separately from the statistics table, giving researchers flexibility to run their own follow-up analyses. Detailed subject-level data in the explorer is tucked into a collapsible expander to keep the view clean while still being accessible. Metric cards at the top of each tab give an at-a-glance summary before the detailed results load.

**Statistical choices in Part 3.** After researching, I chose to use Mann-Whitney U as the primary test because it makes no assumption about normality, which is appropriate for immune cell percentage data that can be skewed. Benjamini-Hochberg FDR correction is applied across all five populations to control the false discovery rate from multiple comparisons. Cohen's d and rank-biserial correlation are reported alongside p-values to quantify effect size, since statistical significance alone is insufficient to judge biological relevance.

**Dashboard as a complement, not a replacement.** The standalone scripts (`part2`–`part4`) exist independently of the dashboard and produce their own outputs. The dashboard is an interactive layer on top of the same database useful for exploration, but the scripts ensure all required outputs can be reproduced programmatically without a browser.

**One script per part.** Each analytical part is a self-contained script that can be run independently, read, and understood on its own. This makes it straightforward to reproduce any individual result without running the whole pipeline, and keeps the analytical logic separate from the dashboard UI.
