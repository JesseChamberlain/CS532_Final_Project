import time
from pathlib import Path

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# --------------------------------------------------
# 1. Project constants
# --------------------------------------------------
CSV_PATH = "Loan_Default.csv"
LABEL_COL = "Status"
CATEGORICAL_COLS = ["loan_purpose", "Gender", "loan_type", "Region"]
NUMERIC_COLS = [
    "loan_amount",
    "rate_of_interest",
    "property_value",
    "income",
    "Credit_Score",
    "LTV",
    "dtir1",
]
OUTPUT_DIR = Path("outputs")


# --------------------------------------------------
# 2. Spark session setup
# --------------------------------------------------
def create_spark(app_name="LoanDefaultFinalProject"):
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


# --------------------------------------------------
# 3. Load data + lazy execution profiling
# --------------------------------------------------
def load_and_profile_data(spark, csv_path):
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    print("Schema:")
    df.printSchema()

    print(f"Initial partition count: {df.rdd.getNumPartitions()}")
    default_par = spark.sparkContext.defaultParallelism
    df = df.repartition(default_par)
    print(
        f"Repartitioned to {df.rdd.getNumPartitions()} partitions (defaultParallelism={default_par})"
    )

    df_filtered = df.filter(F.col("loan_amount") > 1000)
    print("Applied filter transformation (lazy, no action yet).")

    print("\nExecution plan (lazy DAG before action):")
    df_filtered.explain(extended=True)

    count = df_filtered.count()
    print(f"Count after lazy filter (action triggered execution): {count}\n")

    print("Sample rows:")
    df.show(5, truncate=False)

    return df


# --------------------------------------------------
# 4. Data cleaning
# --------------------------------------------------
def clean_data(df):
    df = df.dropna(subset=[LABEL_COL])
    df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("double"))

    for c in CATEGORICAL_COLS:
        df = df.fillna({c: "Unknown"})

    print("Row count after cleaning:", df.count())
    print("Label counts:")
    df.groupBy(LABEL_COL).count().show()

    return df


# --------------------------------------------------
# 5. Partitioning + shuffle benchmark
# --------------------------------------------------
def benchmark_partition_and_shuffle(df, spark):
    print("\n" + "=" * 80)
    print("PARTITIONING & SHUFFLE OPTIMIZATION BENCHMARK")
    print("=" * 80)

    default_par = max(2, spark.sparkContext.defaultParallelism)
    repartition_options = sorted({default_par // 2, default_par, default_par * 2})
    repartition_options = [x for x in repartition_options if x >= 2]
    shuffle_options = [
        max(8, default_par),
        max(16, default_par * 2),
        max(32, default_par * 4),
    ]

    benchmark_rows = []

    for repart_n in repartition_options:
        for shuffle_n in shuffle_options:
            spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_n))

            t0 = time.time()
            workload_df = (
                df.repartition(repart_n, "Region")
                .groupBy("Region", "loan_purpose")
                .agg(
                    F.count("*").alias("loan_count"),
                    F.avg("loan_amount").alias("avg_loan_amount"),
                    F.avg("income").alias("avg_income"),
                )
                .orderBy(F.col("loan_count").desc())
            )

            _ = workload_df.collect()
            elapsed = time.time() - t0

            benchmark_rows.append(
                {
                    "repartition": repart_n,
                    "shuffle_partitions": shuffle_n,
                    "runtime_seconds": round(elapsed, 3),
                }
            )

    benchmark_pd = pd.DataFrame(benchmark_rows).sort_values("runtime_seconds")
    print(benchmark_pd.to_string(index=False))

    best_row = benchmark_pd.iloc[0]
    print("\nBest partition/shuffle config found:")
    print(
        f"repartition={int(best_row['repartition'])}, "
        f"shuffle_partitions={int(best_row['shuffle_partitions'])}, "
        f"runtime={best_row['runtime_seconds']}s"
    )

    spark.conf.set("spark.sql.shuffle.partitions", str(int(best_row["shuffle_partitions"])))
    return int(best_row["repartition"]), int(best_row["shuffle_partitions"]), benchmark_pd


# --------------------------------------------------
# 6. Cache storage level benchmark
# --------------------------------------------------
def benchmark_cache_levels(train_df):
    print("\n" + "=" * 80)
    print("CACHE STORAGE LEVEL BENCHMARK")
    print("=" * 80)

    levels = [
        ("MEMORY_ONLY", StorageLevel.MEMORY_ONLY),
        ("MEMORY_AND_DISK", StorageLevel.MEMORY_AND_DISK),
        ("DISK_ONLY", StorageLevel.DISK_ONLY),
    ]

    rows = []
    for name, level in levels:
        train_df.unpersist(blocking=True)
        cached = train_df.persist(level)

        t0 = time.time()
        _ = cached.count()
        first_pass = time.time() - t0

        t1 = time.time()
        _ = cached.count()
        second_pass = time.time() - t1

        rows.append(
            {
                "storage_level": name,
                "first_pass_seconds": round(first_pass, 3),
                "second_pass_seconds": round(second_pass, 3),
            }
        )

    train_df.unpersist(blocking=True)

    cache_pd = pd.DataFrame(rows).sort_values("second_pass_seconds")
    print(cache_pd.to_string(index=False))

    return cache_pd


# --------------------------------------------------
# 7. Shared preprocessing stages
# --------------------------------------------------
def build_preprocessing_stages():
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
        for c in CATEGORICAL_COLS
    ]

    encoders = [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec")
        for c in CATEGORICAL_COLS
    ]

    imputer = Imputer(
        inputCols=NUMERIC_COLS,
        outputCols=[f"{c}_imputed" for c in NUMERIC_COLS],
    ).setStrategy("median")

    feature_cols = [f"{c}_vec" for c in CATEGORICAL_COLS] + [
        f"{c}_imputed" for c in NUMERIC_COLS
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="assembled_features",
        handleInvalid="skip",
    )

    scaler = StandardScaler(inputCol="assembled_features", outputCol="features")

    return indexers, encoders, imputer, assembler, scaler


# --------------------------------------------------
# 8. Class imbalance utilities
# --------------------------------------------------
def find_minority_label(df):
    counts = (
        df.groupBy(LABEL_COL)
        .count()
        .orderBy(F.col("count").asc())
        .collect()
    )
    minority_label = counts[0][LABEL_COL]
    majority_label = counts[-1][LABEL_COL]
    minority_count = counts[0]["count"]
    majority_count = counts[-1]["count"]

    print(
        f"Minority label={minority_label} (count={minority_count}), "
        f"Majority label={majority_label} (count={majority_count})"
    )

    return minority_label, majority_label, minority_count, majority_count


def make_weighted_train_df(train_df, minority_label, minority_count, majority_count):
    imbalance_ratio = majority_count / max(1, minority_count)

    weighted_df = train_df.withColumn(
        "class_weight",
        F.when(F.col(LABEL_COL) == F.lit(minority_label), F.lit(float(imbalance_ratio))).otherwise(F.lit(1.0)),
    )

    print(f"Applied class weighting ratio: {imbalance_ratio:.3f}")
    return weighted_df


def make_resampled_train_df(train_df, minority_label, minority_count, majority_count):
    minority_df = train_df.filter(F.col(LABEL_COL) == F.lit(minority_label))
    majority_df = train_df.filter(F.col(LABEL_COL) != F.lit(minority_label))

    downsample_fraction = min(1.0, minority_count / max(1, majority_count))
    majority_downsampled = majority_df.sample(withReplacement=False, fraction=downsample_fraction, seed=42)

    resampled = minority_df.unionByName(majority_downsampled)

    print(
        f"Downsampled majority class with fraction={downsample_fraction:.4f}. "
        f"Resampled train rows={resampled.count()}"
    )

    return resampled


# --------------------------------------------------
# 9. Model definitions + evaluation
# --------------------------------------------------
def get_models(weight_col=None):
    lr_kwargs = {
        "featuresCol": "features",
        "labelCol": LABEL_COL,
        "maxIter": 120,
        "regParam": 0.05,
        "elasticNetParam": 0.2,
    }
    dt_kwargs = {
        "featuresCol": "features",
        "labelCol": LABEL_COL,
        "maxDepth": 8,
        "minInstancesPerNode": 30,
    }
    rf_kwargs = {
        "featuresCol": "features",
        "labelCol": LABEL_COL,
        "numTrees": 80,
        "maxDepth": 10,
        "seed": 42,
    }

    if weight_col:
        lr_kwargs["weightCol"] = weight_col
        dt_kwargs["weightCol"] = weight_col
        rf_kwargs["weightCol"] = weight_col

    return {
        "Logistic Regression": LogisticRegression(**lr_kwargs),
        "Decision Tree": DecisionTreeClassifier(**dt_kwargs),
        "Random Forest": RandomForestClassifier(**rf_kwargs),
    }


def evaluate_models(train_df, test_df, scenario_name, show_plan=False, weight_col=None):
    indexers, encoders, imputer, assembler, scaler = build_preprocessing_stages()
    models = get_models(weight_col=weight_col)

    binary_eval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    multi_eval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
    )

    results = []

    for model_name, classifier in models.items():
        print("\n" + "-" * 70)
        print(f"Scenario={scenario_name} | Model={model_name}")
        print("-" * 70)

        pipeline = Pipeline(
            stages=indexers + encoders + [imputer, assembler, scaler, classifier]
        )

        t0 = time.time()
        fitted_model = pipeline.fit(train_df)
        predictions = fitted_model.transform(test_df)
        train_eval_time = time.time() - t0

        if show_plan and model_name == "Logistic Regression":
            print("Physical execution plan (formatted):")
            predictions.explain(mode="formatted")

        auc = binary_eval.evaluate(predictions)
        accuracy = multi_eval.setMetricName("accuracy").evaluate(predictions)
        f1 = multi_eval.setMetricName("f1").evaluate(predictions)
        weighted_precision = multi_eval.setMetricName("weightedPrecision").evaluate(
            predictions
        )
        weighted_recall = multi_eval.setMetricName("weightedRecall").evaluate(predictions)

        predictions.select(LABEL_COL, "prediction", "probability").show(6, truncate=False)

        results.append(
            {
                "Scenario": scenario_name,
                "Model": model_name,
                "AUC": round(auc, 4),
                "Accuracy": round(accuracy, 4),
                "F1 Score": round(f1, 4),
                "Weighted Precision": round(weighted_precision, 4),
                "Weighted Recall": round(weighted_recall, 4),
                "Train+Eval Time (s)": round(train_eval_time, 3),
            }
        )

    return results


# --------------------------------------------------
# 10. Fault tolerance demonstration
# --------------------------------------------------
def demonstrate_fault_tolerance(spark, df):
    print("\n" + "=" * 80)
    print("FAULT TOLERANCE DEMONSTRATION (LINEAGE RECOMPUTATION)")
    print("=" * 80)

    checkpoint_dir = str(Path("spark_checkpoints").resolve())
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    lineage_df = (
        df.filter(F.col("loan_amount") > 1000)
        .withColumn("ltv_income_ratio", F.col("LTV") / F.when(F.col("income") == 0, 1).otherwise(F.col("income")))
        .groupBy("Region")
        .agg(F.avg("ltv_income_ratio").alias("avg_ltv_income_ratio"), F.count("*").alias("n"))
    )

    cached_df = lineage_df.persist(StorageLevel.MEMORY_ONLY)

    t0 = time.time()
    first_count = cached_df.count()
    first_time = time.time() - t0

    t1 = time.time()
    second_count = cached_df.count()
    second_time = time.time() - t1

    cached_df.unpersist(blocking=True)

    t2 = time.time()
    third_count = lineage_df.count()
    third_time = time.time() - t2

    checkpointed = lineage_df.checkpoint(eager=True)
    cp_count = checkpointed.count()

    print(f"First action (compute + cache): rows={first_count}, time={first_time:.3f}s")
    print(f"Second action (cache hit):     rows={second_count}, time={second_time:.3f}s")
    print(f"After unpersist (recompute):   rows={third_count}, time={third_time:.3f}s")
    print(f"Checkpointed rows: {cp_count}")
    print("Recompute after cache loss demonstrates lineage-based recovery.")

    return pd.DataFrame(
        [
            {
                "first_count": first_count,
                "first_time_seconds": round(first_time, 3),
                "second_count": second_count,
                "second_time_seconds": round(second_time, 3),
                "third_count": third_count,
                "third_time_seconds": round(third_time, 3),
                "checkpoint_count": cp_count,
                "recompute_slower_than_cache_hit": round(max(0.0, third_time - second_time), 3),
            }
        ]
    )


# --------------------------------------------------
# 11. Resource tuning + scalability benchmark
# --------------------------------------------------
def run_scalability_experiment(csv_path):
    print("\n" + "=" * 80)
    print("RESOURCE TUNING & SCALABILITY ANALYSIS")
    print("=" * 80)

    configs = [
        {"cores": 1, "driver_memory": "1g"},
        {"cores": 2, "driver_memory": "2g"},
        {"cores": 4, "driver_memory": "2g"},
    ]

    rows = []

    for cfg in configs:
        cores = cfg["cores"]
        driver_memory = cfg["driver_memory"]
        shuffle_parts = max(8, cores * 4)

        spark = (
            SparkSession.builder.appName(f"LoanDefaultScalability_{cores}c")
            .master(f"local[{cores}]")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", str(shuffle_parts))
            .getOrCreate()
        )

        try:
            t0 = time.time()
            sdf = spark.read.csv(csv_path, header=True, inferSchema=True)
            row_count = sdf.count()

            t1 = time.time()
            agg = (
                sdf.select("Region", "loan_amount", "income", LABEL_COL)
                .dropna(subset=["Region", "loan_amount", "income", LABEL_COL])
                .repartition(max(4, cores * 2), "Region")
                .groupBy("Region")
                .agg(
                    F.count("*").alias("rows"),
                    F.avg("loan_amount").alias("avg_loan_amount"),
                    F.avg("income").alias("avg_income"),
                )
            )
            _ = agg.collect()

            t2 = time.time()
            total_time = t2 - t0
            throughput = row_count / max(total_time, 1e-9)

            rows.append(
                {
                    "cores": cores,
                    "driver_memory": driver_memory,
                    "shuffle_partitions": shuffle_parts,
                    "rows": row_count,
                    "total_runtime_seconds": round(total_time, 3),
                    "throughput_rows_per_sec": round(throughput, 2),
                }
            )

            print(
                f"cores={cores}, memory={driver_memory}, rows={row_count}, "
                f"time={total_time:.3f}s, throughput={throughput:.2f} rows/s"
            )

        finally:
            spark.stop()

    scale_pd = pd.DataFrame(rows).sort_values("cores")
    baseline_time = float(scale_pd.iloc[0]["total_runtime_seconds"])
    scale_pd["speedup_vs_1_core"] = (
        baseline_time / scale_pd["total_runtime_seconds"]
    ).round(3)

    print("\nScalability summary:")
    print(scale_pd.to_string(index=False))

    best = scale_pd.sort_values("throughput_rows_per_sec", ascending=False).iloc[0]
    print(
        f"\nBest throughput config: cores={int(best['cores'])}, "
        f"memory={best['driver_memory']}, throughput={best['throughput_rows_per_sec']} rows/s"
    )

    if len(scale_pd) >= 3:
        s1 = float(scale_pd.iloc[1]["speedup_vs_1_core"])
        s2 = float(scale_pd.iloc[2]["speedup_vs_1_core"])
        print(
            f"Diminishing returns check: speedup at 2 cores={s1:.3f}, "
            f"at 4 cores={s2:.3f}."
        )

    return scale_pd


# --------------------------------------------------
# 12. Main orchestration
# --------------------------------------------------
def main():
    # Ensure output folder exists before writing benchmark CSVs.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    try:
        # Load + clean the full dataset used by all experiments.
        df = load_and_profile_data(spark, CSV_PATH)
        df = clean_data(df)

        # Tune partition/shuffle settings before model training.
        best_repart, best_shuffle, partition_benchmark_pd = benchmark_partition_and_shuffle(df, spark)

        # Split and persist core DataFrames for repeated training/evaluation actions.
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
        train_df = train_df.repartition(best_repart)
        test_df = test_df.repartition(max(2, best_repart // 2))

        train_df.persist(StorageLevel.MEMORY_AND_DISK)
        test_df.persist(StorageLevel.MEMORY_AND_DISK)

        _ = train_df.count()
        _ = test_df.count()

        print(f"\ntrain_df storage level: {train_df.storageLevel}")
        print(f"test_df storage level: {test_df.storageLevel}")
        print(f"Using tuned shuffle partitions={best_shuffle}")

        # Compare cache storage strategies on the training split.
        cache_benchmark_pd = benchmark_cache_levels(train_df)

        minority_label, majority_label, minority_count, majority_count = find_minority_label(
            train_df
        )

        weighted_train_df = make_weighted_train_df(
            train_df, minority_label, minority_count, majority_count
        )
        resampled_train_df = make_resampled_train_df(
            train_df, minority_label, minority_count, majority_count
        )

        # Evaluate baseline, weighted, and resampled training scenarios.
        all_results = []

        all_results.extend(
            evaluate_models(
                train_df,
                test_df,
                scenario_name="Baseline (No Imbalance Handling)",
                show_plan=True,
                weight_col=None,
            )
        )

        all_results.extend(
            evaluate_models(
                weighted_train_df,
                test_df,
                scenario_name="Class Weighting",
                show_plan=False,
                weight_col="class_weight",
            )
        )

        all_results.extend(
            evaluate_models(
                resampled_train_df,
                test_df,
                scenario_name="Majority Downsampling",
                show_plan=False,
                weight_col=None,
            )
        )

        results_pd = pd.DataFrame(all_results)
        results_pd = results_pd.sort_values(
            by=["AUC", "F1 Score", "Weighted Recall"], ascending=False
        ).reset_index(drop=True)

        print("\n" + "#" * 80)
        print("MODEL + SYSTEM COMPARISON TABLE")
        print("#" * 80)
        print(results_pd.to_string(index=False))

        best_row = results_pd.iloc[0]
        print("\n" + "*" * 60)
        print("BEST CONFIGURATION")
        print("*" * 60)
        print(f"Scenario: {best_row['Scenario']}")
        print(f"Model: {best_row['Model']}")
        print(f"AUC: {best_row['AUC']}")
        print(f"Accuracy: {best_row['Accuracy']}")
        print(f"F1 Score: {best_row['F1 Score']}")
        print(f"Weighted Precision: {best_row['Weighted Precision']}")
        print(f"Weighted Recall: {best_row['Weighted Recall']}")
        print(f"Train+Eval Time (s): {best_row['Train+Eval Time (s)']}")

        # Demonstrate Spark lineage recomputation and checkpoint behavior.
        fault_tolerance_pd = demonstrate_fault_tolerance(spark, df)

        # Export all benchmark/model outputs for reporting.
        partition_benchmark_pd.to_csv(OUTPUT_DIR / "partition_shuffle_benchmark.csv", index=False)
        cache_benchmark_pd.to_csv(OUTPUT_DIR / "cache_storage_benchmark.csv", index=False)
        results_pd.to_csv(OUTPUT_DIR / "model_system_comparison.csv", index=False)
        fault_tolerance_pd.to_csv(OUTPUT_DIR / "fault_tolerance_benchmark.csv", index=False)

        print("\nSaved CSV outputs:")
        print(f"- {OUTPUT_DIR / 'partition_shuffle_benchmark.csv'}")
        print(f"- {OUTPUT_DIR / 'cache_storage_benchmark.csv'}")
        print(f"- {OUTPUT_DIR / 'model_system_comparison.csv'}")
        print(f"- {OUTPUT_DIR / 'fault_tolerance_benchmark.csv'}")

        train_df.unpersist(blocking=True)
        test_df.unpersist(blocking=True)

    finally:
        spark.stop()

    # Run and export the separate scalability sweep (new Spark sessions per config).
    scale_pd = run_scalability_experiment(CSV_PATH)
    scale_pd.to_csv(OUTPUT_DIR / "resource_scalability_benchmark.csv", index=False)
    print(f"- {OUTPUT_DIR / 'resource_scalability_benchmark.csv'}")


if __name__ == "__main__":
    main()
