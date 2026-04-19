from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pandas as pd

# --------------------------------------------------
# 1. Create Spark session
# --------------------------------------------------
spark = SparkSession.builder.appName("LoanDefaultModelComparison").getOrCreate()

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
df = spark.read.csv("Loan_Default.csv", header=True, inferSchema=True)

print("Schema:")
df.printSchema()

print("Sample rows:")
df.show(5)

print("Row count before cleaning:", df.count())

# --------------------------------------------------
# 3. Define columns
# --------------------------------------------------
categorical_cols = [
    "loan_purpose",
    "Gender",
    "loan_type",
    "Region"
]

numeric_cols = [
    "loan_amount",
    "rate_of_interest",
    "property_value",
    "income",
    "Credit_Score",
    "LTV",
    "dtir1"
]

label_col = "Status"

# --------------------------------------------------
# 4. Clean data
# --------------------------------------------------
df = df.dropna(subset=[label_col])
df = df.withColumn(label_col, col(label_col).cast("double"))

for c in categorical_cols:
    df = df.fillna({c: "Unknown"})

print("Row count after label cleaning:", df.count())

print("Label counts in full dataset:")
df.groupBy(label_col).count().show()

# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print("Training label counts:")
train_df.groupBy(label_col).count().show()

print("Testing label counts:")
test_df.groupBy(label_col).count().show()

# --------------------------------------------------
# 6. Shared preprocessing stages
# --------------------------------------------------
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec")
    for c in categorical_cols
]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[f"{c}_imputed" for c in numeric_cols]
).setStrategy("median")

feature_cols = [f"{c}_vec" for c in categorical_cols] + [f"{c}_imputed" for c in numeric_cols]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="assembled_features",
    handleInvalid="skip"
)

scaler = StandardScaler(
    inputCol="assembled_features",
    outputCol="features"
)

# --------------------------------------------------
# 7. Define models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        featuresCol="features",
        labelCol=label_col,
        maxIter=100
    ),
    "Decision Tree": DecisionTreeClassifier(
        featuresCol="features",
        labelCol=label_col,
        maxDepth=5
    ),
    "Random Forest": RandomForestClassifier(
        featuresCol="features",
        labelCol=label_col,
        numTrees=50,
        maxDepth=8,
        seed=42
    )
}

# --------------------------------------------------
# 8. Evaluators
# --------------------------------------------------
binary_eval = BinaryClassificationEvaluator(
    labelCol=label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multi_eval = MulticlassClassificationEvaluator(
    labelCol=label_col,
    predictionCol="prediction"
)

# --------------------------------------------------
# 9. Train and evaluate each model
# --------------------------------------------------
results = []

for model_name, classifier in models.items():
    print(f"\n{'=' * 60}")
    print(f"Training {model_name}")
    print(f"{'=' * 60}")

    pipeline = Pipeline(
        stages=indexers + encoders + [imputer, assembler, scaler, classifier]
    )

    fitted_model = pipeline.fit(train_df)
    predictions = fitted_model.transform(test_df)

    print(f"Sample predictions for {model_name}:")
    predictions.select(label_col, "prediction", "probability").show(10, truncate=False)

    auc = binary_eval.evaluate(predictions)
    accuracy = multi_eval.setMetricName("accuracy").evaluate(predictions)
    f1 = multi_eval.setMetricName("f1").evaluate(predictions)
    weighted_precision = multi_eval.setMetricName("weightedPrecision").evaluate(predictions)
    weighted_recall = multi_eval.setMetricName("weightedRecall").evaluate(predictions)

    results.append({
        "Model": model_name,
        "AUC": auc,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Weighted Precision": weighted_precision,
        "Weighted Recall": weighted_recall
    })

# --------------------------------------------------
# 10. Convert results to pandas DataFrame
# --------------------------------------------------
results_pd = pd.DataFrame(results)

results_pd = results_pd.sort_values(by="AUC", ascending=False).reset_index(drop=True)

metric_cols = ["AUC", "Accuracy", "F1 Score", "Weighted Precision", "Weighted Recall"]
results_pd[metric_cols] = results_pd[metric_cols].round(4)

print("\n" + "#" * 80)
print("MODEL COMPARISON TABLE")
print("#" * 80)
print(results_pd.to_string(index=False))

# --------------------------------------------------
# 11. Best model
# --------------------------------------------------
best_row = results_pd.iloc[0]

print("\n" + "*" * 60)
print("BEST MODEL BASED ON AUC")
print("*" * 60)
print(f"Best Model: {best_row['Model']}")
print(f"AUC: {best_row['AUC']}")
print(f"Accuracy: {best_row['Accuracy']}")
print(f"F1 Score: {best_row['F1 Score']}")
print(f"Weighted Precision: {best_row['Weighted Precision']}")
print(f"Weighted Recall: {best_row['Weighted Recall']}")

spark.stop()