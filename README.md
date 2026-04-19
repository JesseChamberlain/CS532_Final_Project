# CS532 Final Project

Loan application analytics and model benchmarking pipeline built with PySpark.

# --------------------------------------------------
# Quick Start
# --------------------------------------------------

## Prerequisites

- Python 3.10+
- Java 17 or 21

## Setup and Run

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python spark_loan.py

# Generate visualization charts for the report
python generate_visualizations.py
```

## Output Files

After running, you will have:

- `outputs/*.csv` - Benchmark data files
- `figures/*.png` - Visualization charts for your slideshow

# --------------------------------------------------
# 1. Project constants
# --------------------------------------------------

- Input dataset: `Loan_Default.csv`
- Label column: `Status`
- Output folder: `outputs/`

# --------------------------------------------------
# 2. Spark session setup
# --------------------------------------------------

The project creates Spark sessions with adaptive query execution settings enabled for better shuffle/partition behavior.

# --------------------------------------------------
# 3. Load data + lazy execution profiling
# --------------------------------------------------

The pipeline:

- Loads CSV with schema inference
- Prints schema and sample rows
- Shows partition count and repartitions based on default parallelism
- Demonstrates lazy execution and explains physical/logical plan before action

# --------------------------------------------------
# 4. Data cleaning
# --------------------------------------------------

Cleaning includes:

- Dropping rows with missing labels
- Casting label to numeric
- Filling categorical nulls with `Unknown`
- Printing post-cleaning class distribution

# --------------------------------------------------
# 5. Partitioning + shuffle benchmark
# --------------------------------------------------

The script tests multiple repartition counts and `spark.sql.shuffle.partitions` settings, then records runtime for each configuration.

Output CSV:

- `outputs/partition_shuffle_benchmark.csv`

# --------------------------------------------------
# 6. Cache storage level benchmark
# --------------------------------------------------

The script compares cache storage levels (`MEMORY_ONLY`, `MEMORY_AND_DISK`, `DISK_ONLY`) using repeated actions to measure first pass and cache-hit timings.

Output CSV:

- `outputs/cache_storage_benchmark.csv`

# --------------------------------------------------
# 7. Shared preprocessing stages
# --------------------------------------------------

Shared ML preprocessing pipeline:

- `StringIndexer` for categorical columns
- `OneHotEncoder` for indexed categoricals
- `Imputer` (median) for numeric null handling
- `VectorAssembler` and `StandardScaler` for final features

# --------------------------------------------------
# 8. Class imbalance utilities
# --------------------------------------------------

Two imbalance strategies are evaluated:

- Class weighting (`class_weight`)
- Majority downsampling

# --------------------------------------------------
# 9. Model definitions + evaluation
# --------------------------------------------------

Models trained per scenario:

- Logistic Regression
- Decision Tree
- Random Forest

Metrics recorded:

- AUC
- Accuracy
- F1 Score
- Weighted Precision
- Weighted Recall
- Train+Eval runtime

Output CSV:

- `outputs/model_system_comparison.csv`

# --------------------------------------------------
# 10. Fault tolerance demonstration
# --------------------------------------------------

The script demonstrates lineage-based recomputation by:

- Computing and caching a derived DataFrame
- Measuring cache-hit runtime
- Unpersisting cache and measuring recomputation runtime
- Creating a checkpoint and validating checkpointed count

Output CSV:

- `outputs/fault_tolerance_benchmark.csv`

# --------------------------------------------------
# 11. Resource tuning + scalability benchmark
# --------------------------------------------------

The script runs the same workload under different local Spark resource configs:

- Cores: 1, 2, 4
- Driver memory: 1g/2g

It reports throughput, total runtime, and speedup vs 1 core to identify diminishing returns.

Output CSV:

- `outputs/resource_scalability_benchmark.csv`

# --------------------------------------------------
# 12. Main orchestration
# --------------------------------------------------

Execution order in `spark_loan.py`:

1. Build Spark session
2. Load/profile/clean data
3. Tune partition and shuffle settings
4. Split and persist train/test DataFrames
5. Benchmark caching levels
6. Build weighted and resampled training variants
7. Train/evaluate all models across scenarios
8. Run fault tolerance demonstration
9. Export benchmark/model outputs to CSV
10. Run scalability experiment and export results

# --------------------------------------------------
# 13. Visualizations Guide
# --------------------------------------------------

Run `python generate_visualizations.py` to create charts in `figures/`.

## class_imbalance.png
Pie and bar chart of the distribution of loan defaults (class 1) vs non-defaults (class 0)

## partition_benchmark.png
Heatmap and bar chart of runtime for different combinations of repartition count and shuffle partition settings

## cache_benchmark.png
Bar chart comparison of MEMORY_ONLY, MEMORY_AND_DISK, DISK_ONLY first pass vs cache hit

## fault_tolerance.png
Bars comparing initial compute time, cache hit time, and recomputation time after cache is lost

## scalability.png
Throughput (rows/second) and speedup as CPU cores and memory increase

## model_comparison.png
Bar charts comparing model AUC scores and training times across different configs

## model_heatmap.png
Heatmap of AUC, Accuracy, F1, Precision, Recallacross all model/scenario combos

# --------------------------------------------------
# 14. Slideshow Outline
# --------------------------------------------------

## Slide 1: Title
- Project: Loan Default Prediction with Apache Spark
- Team members
  
## Slide 2: Problem Statement
- Goal: Predict loan defaults using distributed computing
- Challenge: Large dataset, class imbalance (3:1 ratio where most borrowers repaid their loan)
- Figure: `class_imbalance.png`

## Slide 3: System Architecture
- Spark driver/worker model
- Data flow: CSV -> DataFrame -> ML Pipeline -> Predictions
- Figure: `Map Spark deployment architecture.jpg`

## Slide 4: Data Pipeline
- Loading with schema inference
- Lazy evaluation and DAG execution
- Preprocessing: cleaning, encoding, scaling

## Slide 5: Partitioning Optimization
- Why partitioning matters for parallel performance
- Benchmark results: best config found
- Figure: `partition_benchmark.png`

## Slide 6: Caching Strategies
- Storage levels: MEMORY_ONLY, MEMORY_AND_DISK, DISK_ONLY
- Cache hit provides 5-15x speedup
- Figure: `cache_benchmark.png`

## Slide 7: Fault Tolerance
- Spark's lineage-based recovery
- Demo: unpersist cache, recompute from DAG
- Figure: `fault_tolerance.png`

## Slide 8: Scalability Analysis
- Testing with 1, 2, and 4 cores
- Diminishing returns observed (Amdahl's Law)
- Figure: `scalability.png`

## Slide 9: ML Model Comparison
- Three models: Logistic Regression, Decision Tree, Random Forest
- Three scenarios: Baseline, Class Weighting, Downsampling
- Figure: `model_comparison.png` or `model_heatmap.png`

## Slide 10: Results Summary
- Found best model was Random Forest w (AUC = 0.98)
- Found best Spark config was 10 partitions, 40 shuffle partitions
- Show that tuning and model go together

## Slide 11: Conclusions
- Demonstrated distributed ML pipeline with Spark
- Showed impact of partitioning, caching, and resource allocation
- Validated fault tolerance via RDD lineage

