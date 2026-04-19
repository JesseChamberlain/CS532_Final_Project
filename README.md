# CS532 Final Project

Loan application analytics and model benchmarking pipeline built with PySpark.

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

