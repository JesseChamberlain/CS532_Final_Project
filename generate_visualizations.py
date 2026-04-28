from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path("Loan_Default.csv")
OUTPUT_DIR = Path("outputs")
FIGURES_DIR = Path("figures")


def setup_style():
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )


def load_class_counts():
    status_df = pd.read_csv(DATA_PATH, usecols=["Status"])
    counts = status_df["Status"].dropna().astype(int).value_counts().to_dict()
    non_default_count = int(counts.get(0, 0))
    default_count = int(counts.get(1, 0))
    return non_default_count, default_count


def plot_model_comparison():
    df = pd.read_csv(OUTPUT_DIR / "model_system_comparison.csv")
    # AUC comparison bar
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot_auc = df.pivot(index="Model", columns="Scenario", values="AUC")
    pivot_auc.plot(kind="bar", ax=axes[0], colormap="viridis", edgecolor="black")
    axes[0].set_title("AUC by Model and Scenario")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("AUC Score")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(title="Scenario", loc="lower right", fontsize=8)
    axes[0].tick_params(axis="x", rotation=15)

    # training time comparison bar
    pivot_time = df.pivot(
        index="Model", columns="Scenario", values="Train+Eval Time (s)"
    )
    pivot_time.plot(kind="bar", ax=axes[1], colormap="plasma", edgecolor="black")
    axes[1].set_title("Training + Evaluation Time by Model")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Time (seconds)")
    axes[1].legend(title="Scenario", loc="upper right", fontsize=8)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close()

    # heatmap of combinations
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["AUC", "Accuracy", "F1 Score", "Weighted Precision", "Weighted Recall"]
    df["Config"] = df["Scenario"] + " | " + df["Model"].str[:8]
    heatmap_data = df.set_index("Config")[metrics].T
    im = ax.imshow(
        heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0
    )

    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)

    # add values in heatmap
    for i in range(len(metrics)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            color = "white" if val < 0.75 else "black"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8
            )

    plt.colorbar(im, ax=ax, label="Score")
    ax.set_title("Model Performance Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Generated: model_comparison.png, model_heatmap.png")


# visualize shuffles vs runtime
def plot_partition_benchmark():
    df = pd.read_csv(OUTPUT_DIR / "partition_shuffle_benchmark.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # heatmap of runtime + config
    pivot = df.pivot(
        index="repartition", columns="shuffle_partitions", values="runtime_seconds"
    )

    im = axes[0].imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel("Shuffle Partitions")
    axes[0].set_ylabel("Repartition Count")
    axes[0].set_title("Runtime (s) by Partition Configuration")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            axes[0].text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=axes[0], label="Runtime (s)")

    # bar of runtime vs config
    df_sorted = df.sort_values("runtime_seconds")
    df_sorted["config"] = df_sorted.apply(
        lambda r: f"R={int(r['repartition'])}, S={int(r['shuffle_partitions'])}", axis=1
    )

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_sorted)))
    bars = axes[1].barh(
        df_sorted["config"],
        df_sorted["runtime_seconds"],
        color=colors,
        edgecolor="black",
    )
    axes[1].set_xlabel("Runtime (seconds)")
    axes[1].set_ylabel("Configuration")
    axes[1].set_title("Partition Configurations Ranked by Performance")
    axes[1].invert_yaxis()

    # highlight best
    bars[0].set_edgecolor("green")
    bars[0].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "partition_benchmark.png", bbox_inches="tight")
    plt.close()
    print("Generated: partition_benchmark.png")


def plot_cache_benchmark():
    df = pd.read_csv(OUTPUT_DIR / "cache_storage_benchmark.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        df["first_pass_seconds"],
        width,
        label="First Pass (Cold)",
        color="#ff7f0e",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        df["second_pass_seconds"],
        width,
        label="Second Pass (Cache Hit)",
        color="#2ca02c",
        edgecolor="black",
    )

    ax.set_xlabel("Storage Level")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Cache Storage Level Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(df["storage_level"])
    ax.legend()

    # label seconds
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    # label speedup
    for i, row in df.iterrows():
        speedup = row["first_pass_seconds"] / row["second_pass_seconds"]
        ax.annotate(
            f"{speedup:.1f}x faster",
            xy=(i, max(row["first_pass_seconds"], row["second_pass_seconds"]) + 0.05),
            ha="center",
            fontsize=9,
            color="blue",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cache_benchmark.png", bbox_inches="tight")
    plt.close()
    print("Generated: cache_benchmark.png")


def plot_fault_tolerance():
    # bar graph demonstrating initial computation, cache, recomputation
    df = pd.read_csv(OUTPUT_DIR / "fault_tolerance_benchmark.csv")
    row = df.iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    stages = ["Initial Compute\n+ Cache", "Cache Hit", "Recompute\n(After Unpersist)"]
    times = [
        row["first_time_seconds"],
        row["second_time_seconds"],
        row["third_time_seconds"],
    ]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    bars = ax.bar(stages, times, color=colors, edgecolor="black", linewidth=2)

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Fault Tolerance: Lineage-Based RDD Recomputation Demonstration")

    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(
            f"{time:.3f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    # Add explanatory annotations
    cache_speedup = row["first_time_seconds"] / row["second_time_seconds"]
    ax.annotate(
        f"Cache provides\n{cache_speedup:.1f}x speedup",
        xy=(1, row["second_time_seconds"]),
        xytext=(1.5, 0.12),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=10,
        color="green",
    )

    ax.annotate(
        "Lineage enables\nfault recovery",
        xy=(2, row["third_time_seconds"]),
        xytext=(2.3, 0.08),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fault_tolerance.png", bbox_inches="tight")
    plt.close()
    print("Generated: fault_tolerance.png")


def plot_scalability():
    df = pd.read_csv(OUTPUT_DIR / "resource_scalability_benchmark.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # throughput vs cores bar
    ax1 = axes[0]
    bars = ax1.bar(
        df["cores"].astype(str),
        df["throughput_rows_per_sec"],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black",
        linewidth=2,
    )
    ax1.set_xlabel("Number of Cores")
    ax1.set_ylabel("Throughput (rows/second)")
    ax1.set_title("Throughput Scaling with CPU Cores")

    for bar, val in zip(bars, df["throughput_rows_per_sec"]):
        ax1.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    # actual vs ideal speedup line graph
    ax2 = axes[1]
    ax2.plot(
        df["cores"],
        df["speedup_vs_1_core"],
        "o-",
        markersize=12,
        linewidth=3,
        color="#2ca02c",
        label="Actual Speedup",
    )
    ax2.plot(
        df["cores"],
        df["cores"],
        "--",
        linewidth=2,
        color="gray",
        label="Ideal Linear Scaling",
    )

    ax2.set_xlabel("Number of Cores")
    ax2.set_ylabel("Speedup vs 1 Core")
    ax2.set_title("Scalability Analysis: Actual vs Ideal Speedup")
    ax2.legend()
    ax2.set_xticks(df["cores"])

    # Efficiency percent (how much speedup vs linear)
    for i, row in df.iterrows():
        efficiency = (row["speedup_vs_1_core"] / row["cores"]) * 100
        ax2.annotate(
            f"{efficiency:.0f}% efficient",
            xy=(row["cores"], row["speedup_vs_1_core"]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scalability.png", bbox_inches="tight")
    plt.close()

    print("Generated: scalability.png")


def plot_class_imbalance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # class imbalance pie chart
    labels = ["Non-Default (0)", "Default (1)"]
    non_default_count, default_count = load_class_counts()
    sizes = [non_default_count, default_count]
    colors = ["#2ca02c", "#d62728"]
    explode = (0, 0.05)

    axes[0].pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
        textprops={"fontsize": 11},
    )
    axes[0].set_title("Class Distribution in Dataset")

    # bar chart
    bars = axes[1].bar(labels, sizes, color=colors, edgecolor="black", linewidth=2)
    axes[1].set_ylabel("Number of Samples")
    imbalance_ratio = non_default_count / max(default_count, 1)
    axes[1].set_title(f"Class Imbalance: {imbalance_ratio:.2f}:1 Ratio")

    for bar, size in zip(bars, sizes):
        axes[1].annotate(
            f"{size:,}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_imbalance.png", bbox_inches="tight")
    plt.close()
    print("Generated: class_imbalance.png")


def plot_system_model_comparison():
    df = pd.read_csv(OUTPUT_DIR / "system_model_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- AUC grouped bar chart by model across system configs ---
    ax1 = axes[0]
    models = df["Model"].unique()
    configs = df["Scenario"].unique()
    x = np.arange(len(configs))
    width = 0.22
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(models) - 1, 1)) for i in range(len(models))]

    for i, model in enumerate(models):
        subset = df[df["Model"] == model].set_index("Scenario").reindex(configs)
        bars = ax1.bar(
            x + i * width,
            subset["AUC"],
            width,
            label=model,
            color=colors[i],
            edgecolor="black",
        )
        for bar, val in zip(bars, subset["AUC"]):
            ax1.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax1.set_xlabel("System Configuration")
    ax1.set_ylabel("AUC Score")
    ax1.set_title("AUC by Model Across System Configurations")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
    ax1.set_ylim(0.5, 1.0)
    ax1.legend(title="Model", fontsize=8)

    # --- Train+Eval time grouped bar chart ---
    ax2 = axes[1]
    for i, model in enumerate(models):
        subset = df[df["Model"] == model].set_index("Scenario").reindex(configs)
        bars = ax2.bar(
            x + i * width,
            subset["Train+Eval Time (s)"],
            width,
            label=model,
            color=colors[i],
            edgecolor="black",
        )
        for bar, val in zip(bars, subset["Train+Eval Time (s)"]):
            ax2.annotate(
                f"{val:.1f}s",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax2.set_xlabel("System Configuration")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Training + Eval Time Across System Configurations")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(configs, rotation=20, ha="right", fontsize=8)
    ax2.legend(title="Model", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "system_model_comparison.png", bbox_inches="tight")
    plt.close()

    # --- Heatmap of all metrics across configs ---
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["AUC", "Accuracy", "F1 Score", "Weighted Precision", "Weighted Recall"]
    df["Config"] = df["Scenario"] + " | " + df["Model"].str[:8]
    heatmap_data = df.set_index("Config")[metrics].T

    im = ax.imshow(
        heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0
    )
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metrics)

    for i in range(len(metrics)):
        for j in range(len(heatmap_data.columns)):
            val = heatmap_data.values[i, j]
            color = "white" if val < 0.75 else "black"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8
            )

    plt.colorbar(im, ax=ax, label="Score")
    ax.set_title("Cross-System Model Performance Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "system_model_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Generated: system_model_comparison.png, system_model_heatmap.png")


def plot_cross_validation():
    df = pd.read_csv(OUTPUT_DIR / "cross_validation_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # heatmap of AUC by hyperparameters
    pivot = df.pivot(index="maxDepth", columns="numTrees", values="avg_cv_AUC")

    im = axes[0].imshow(pivot.values, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel("numTrees")
    axes[0].set_ylabel("maxDepth")
    axes[0].set_title("Cross-Validated AUC by Hyperparameters")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 0.97 else "black"
            axes[0].text(
                j,
                i,
                f"{val:.4f}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    plt.colorbar(im, ax=axes[0], label="Avg CV AUC")

    # bar chart sorted by AUC
    df_sorted = df.sort_values("avg_cv_AUC", ascending=True).copy()
    df_sorted["config"] = df_sorted.apply(
        lambda r: f"trees={int(r['numTrees'])}, depth={int(r['maxDepth'])}", axis=1
    )

    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.9, len(df_sorted)))
    bars = axes[1].barh(
        df_sorted["config"], df_sorted["avg_cv_AUC"], color=colors, edgecolor="black"
    )
    axes[1].set_xlabel("Avg CV AUC")
    axes[1].set_ylabel("Hyperparameter Combination")
    axes[1].set_title("Random Forest Hyperparameter Ranking (3-Fold CV)")
    axes[1].set_xlim(
        df_sorted["avg_cv_AUC"].min() - 0.002, df_sorted["avg_cv_AUC"].max() + 0.002
    )

    # highlight best
    bars[-1].set_edgecolor("green")
    bars[-1].set_linewidth(3)

    for bar, val in zip(bars, df_sorted["avg_cv_AUC"]):
        axes[1].annotate(
            f"{val:.4f}",
            xy=(val, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_validation.png", bbox_inches="tight")
    plt.close()
    print("Generated: cross_validation.png")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    plot_model_comparison()
    plot_partition_benchmark()
    plot_cache_benchmark()
    plot_fault_tolerance()
    plot_scalability()
    plot_class_imbalance()
    plot_cross_validation()
    plot_system_model_comparison()


if __name__ == "__main__":
    main()
