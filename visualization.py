"""
==========================================================
AI-Driven Intelligent Decision Support Framework
Visualization Module
==========================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# Generate All Graphs
# ==========================================================

def generate_all_graphs(
        dataset,
        model_results,
        evaluation_results,
        cloud_results):

    print("=" * 70)
    print("GENERATING RESEARCH VISUALIZATIONS")
    print("=" * 70)

    os.makedirs("results/graphs", exist_ok=True)

    feature_importance_graph(model_results)

    actual_vs_predicted_graph(model_results)

    rmse_comparison_graph(evaluation_results)

    mae_comparison_graph(evaluation_results)

    r2_comparison_graph(evaluation_results)

    training_time_graph(evaluation_results)

    inference_time_graph(evaluation_results)

    cpu_usage_graph(evaluation_results)

    memory_usage_graph(evaluation_results)

    cpu_distribution_graph(dataset)

    correlation_matrix_graph(dataset)

    cloud_cost_graph(cloud_results)

    cloud_latency_graph(cloud_results)

    cloud_availability_graph(cloud_results)

    provider_ranking_graph(cloud_results)

    print("\n✓ All graphs generated successfully.")
    print("=" * 70)


# ==========================================================
# Feature Importance
# ==========================================================

def feature_importance_graph(model_results):

    importance_df = model_results["feature_importance"]

    if importance_df is None:

        print("Feature importance not available.")

        return

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    plt.bar(

        importance_df["Feature"][:10],

        importance_df["Importance"][:10]

    )

    plt.xticks(rotation=45, ha="right")

    plt.xlabel("Features")

    plt.ylabel("Importance")

    plt.title("Top 10 Feature Importance")

    plt.tight_layout()

    plt.savefig(

        "results/graphs/feature_importance.png",

        dpi=300

    )

    plt.close()

    print("✓ feature_importance.png")


# ==========================================================
# Actual vs Predicted
# ==========================================================

def actual_vs_predicted_graph(model_results):

    actual = model_results["y_test"]

    predicted = model_results["xgb_prediction"]

    plt.figure(figsize=(7, 7))

    plt.scatter(

        actual,

        predicted,

        alpha=0.7

    )

    minimum = min(actual.min(), predicted.min())

    maximum = max(actual.max(), predicted.max())

    plt.plot(

        [minimum, maximum],

        [minimum, maximum]

    )

    plt.xlabel("Actual CPU")

    plt.ylabel("Predicted CPU")

    plt.title("Actual vs Predicted CPU Usage")

    plt.tight_layout()

    plt.savefig(

        "results/graphs/actual_vs_predicted.png",

        dpi=300

    )

    plt.close()

    print("✓ actual_vs_predicted.png")
    # ==========================================================
# RMSE Comparison
# ==========================================================

def rmse_comparison_graph(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8, 5))

    plt.bar(

        metrics["Model"],

        metrics["RMSE"]

    )

    plt.xlabel("Models")

    plt.ylabel("RMSE")

    plt.title("RMSE Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/rmse_comparison.png",

        dpi=300

    )

    plt.close()

    print("✓ rmse_comparison.png")


# ==========================================================
# MAE Comparison
# ==========================================================

def mae_comparison_graph(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8, 5))

    plt.bar(

        metrics["Model"],

        metrics["MAE"]

    )

    plt.xlabel("Models")

    plt.ylabel("MAE")

    plt.title("MAE Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/mae_comparison.png",

        dpi=300

    )

    plt.close()

    print("✓ mae_comparison.png")


# ==========================================================
# R² Comparison
# ==========================================================

def r2_comparison_graph(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8, 5))

    plt.bar(

        metrics["Model"],

        metrics["R2"]

    )

    plt.xlabel("Models")

    plt.ylabel("R² Score")

    plt.title("R² Score Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/r2_comparison.png",

        dpi=300

    )

    plt.close()

    print("✓ r2_comparison.png")


# ==========================================================
# Training Time Comparison
# ==========================================================

def training_time_graph(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8, 5))

    plt.bar(

        metrics["Model"],

        metrics["Training Time"]

    )

    plt.xlabel("Models")

    plt.ylabel("Time (seconds)")

    plt.title("Model Training Time Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/training_time.png",

        dpi=300

    )

    plt.close()

    print("✓ training_time.png")
    # ==========================================================
# Inference Time Comparison
# ==========================================================

def inference_time_graph(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8, 5))

    plt.bar(

        metrics["Model"],

        metrics["Inference Time"]

    )

    plt.xlabel("Models")

    plt.ylabel("Time (seconds)")

    plt.title("Model Inference Time Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/inference_time.png",

        dpi=300

    )

    plt.close()

    print("✓ inference_time.png")


# ==========================================================
# CPU Usage Comparison
# ==========================================================

def cpu_usage_graph(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8, 5))

    plt.bar(

        metrics["Model"],

        metrics["CPU Usage"]

    )

    plt.xlabel("Models")

    plt.ylabel("CPU Usage (%)")

    plt.title("CPU Usage Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/cpu_usage.png",

        dpi=300

    )

    plt.close()

    print("✓ cpu_usage.png")


# ==========================================================
# Memory Usage Comparison
# ==========================================================

def memory_usage_graph(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8, 5))

    plt.bar(

        metrics["Model"],

        metrics["Memory Usage"]

    )

    plt.xlabel("Models")

    plt.ylabel("Memory Usage (MB)")

    plt.title("Memory Usage Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/memory_usage.png",

        dpi=300

    )

    plt.close()

    print("✓ memory_usage.png")


# ==========================================================
# CPU Workload Distribution
# ==========================================================

def cpu_distribution_graph(dataset):

    plt.figure(figsize=(8, 5))

    plt.hist(

        dataset["plan_cpu"],

        bins=30

    )

    plt.xlabel("Planned CPU")

    plt.ylabel("Frequency")

    plt.title("CPU Workload Distribution")

    plt.tight_layout()

    plt.savefig(

        "results/graphs/cpu_distribution.png",

        dpi=300

    )

    plt.close()

    print("✓ cpu_distribution.png")


# ==========================================================
# Correlation Matrix
# ==========================================================

def correlation_matrix_graph(dataset):

    numeric_dataset = dataset.select_dtypes(include=[np.number])

    correlation = numeric_dataset.corr()

    plt.figure(figsize=(10, 8))

    image = plt.imshow(

        correlation,

        aspect="auto"

    )

    plt.colorbar(image)

    plt.xticks(

        range(len(correlation.columns)),

        correlation.columns,

        rotation=90,

        fontsize=8

    )

    plt.yticks(

        range(len(correlation.columns)),

        correlation.columns,

        fontsize=8

    )

    plt.title("Feature Correlation Matrix")

    plt.tight_layout()

    plt.savefig(

        "results/graphs/correlation_matrix.png",

        dpi=300

    )

    plt.close()

    print("✓ correlation_matrix.png")
    # ==========================================================
# Cloud Cost Comparison
# ==========================================================

def cloud_cost_graph(cloud_results):

    providers = cloud_results["provider_ranking"]

    cost_column = None

    for col in providers.columns:

        name = col.lower()

        if "hourly" in name or "cost" in name or "price" in name:

            cost_column = col

            break

    if cost_column is None:

        print("Cost column not found.")

        return

    plt.figure(figsize=(8, 5))

    plt.bar(

        providers["provider"],

        providers[cost_column]

    )

    plt.xlabel("Cloud Provider")

    plt.ylabel("Hourly Cost")

    plt.title("Cloud Provider Cost Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/cloud_cost_comparison.png",

        dpi=300

    )

    plt.close()

    print("✓ cloud_cost_comparison.png")


# ==========================================================
# Cloud Latency Comparison
# ==========================================================

def cloud_latency_graph(cloud_results):

    providers = cloud_results["provider_ranking"]

    latency_column = None

    for col in providers.columns:

        if "latency" in col.lower():

            latency_column = col

            break

    if latency_column is None:

        print("Latency column not found.")

        return

    plt.figure(figsize=(8,5))

    plt.bar(

        providers["provider"],

        providers[latency_column]

    )

    plt.xlabel("Cloud Provider")

    plt.ylabel("Latency (ms)")

    plt.title("Cloud Provider Latency Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/cloud_latency_comparison.png",

        dpi=300

    )

    plt.close()

    print("✓ cloud_latency_comparison.png")


# ==========================================================
# Cloud Availability Comparison
# ==========================================================

def cloud_availability_graph(cloud_results):

    providers = cloud_results["provider_ranking"]

    availability_column = None

    for col in providers.columns:

        if "availability" in col.lower():

            availability_column = col

            break

    if availability_column is None:

        print("Availability column not found.")

        return

    plt.figure(figsize=(8,5))

    plt.bar(

        providers["provider"],

        providers[availability_column]

    )

    plt.xlabel("Cloud Provider")

    plt.ylabel("Availability (%)")

    plt.title("Cloud Provider Availability Comparison")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/cloud_availability_comparison.png",

        dpi=300

    )

    plt.close()

    print("✓ cloud_availability_comparison.png")


# ==========================================================
# Entropy-Based Provider Ranking
# ==========================================================

def provider_ranking_graph(cloud_results):

    providers = cloud_results["provider_ranking"]

    plt.figure(figsize=(8,5))

    plt.bar(

        providers["provider"],

        providers["Provider_Score"]

    )

    plt.xlabel("Cloud Provider")

    plt.ylabel("Provider Score")

    plt.title("Entropy-Based Cloud Provider Ranking")

    plt.xticks(rotation=20)

    plt.tight_layout()

    plt.savefig(

        "results/graphs/provider_ranking.png",

        dpi=300

    )

    plt.close()

    print("✓ provider_ranking.png")


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)

    print("VISUALIZATION MODULE")

    print("=" * 70)

    print(

        "\nRun main.py after preprocessing, "

        "model training, evaluation and cloud "

        "selection to generate all graphs."

    )

    print("\nVisualization module loaded successfully.")