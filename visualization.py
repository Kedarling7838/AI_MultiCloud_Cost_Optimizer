# Graphs for the research paper
"""
==========================================================
AI-Driven Multi-Cloud Workload Placement
visualization.py
==========================================================
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_all_graphs(
        dataset,
        model_results,
        evaluation_results,
        cloud_pricing,
        latency_dataset):

    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs("results/graphs", exist_ok=True)

    feature_importance(model_results)

    actual_vs_predicted(model_results)

    model_comparison(evaluation_results)

    cpu_distribution(dataset)

    correlation_matrix(dataset)

    cloud_provider_graph(
        cloud_pricing,
        latency_dataset
    )

    print("Graphs Saved Successfully")

    print("=" * 60)


# ======================================================
# Feature Importance
# ======================================================

def feature_importance(model_results):

    model = model_results["xgb_model"]

    importance = model.feature_importances_

    feature_names = model_results["X_test"].columns

    importance_df = pd.DataFrame({

        "Feature": feature_names,
        "Importance": importance

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(12,6))

    plt.bar(
        importance_df["Feature"][:10],
        importance_df["Importance"][:10]
    )

    plt.xticks(rotation=45)

    plt.title("Top 10 Important Features")

    plt.tight_layout()

    plt.savefig(
        "results/graphs/feature_importance.png",
        dpi=300
    )

    plt.close()


# ======================================================
# Actual vs Predicted
# ======================================================

def actual_vs_predicted(model_results):

    actual = model_results["y_test"]

    predicted = model_results["xgb_prediction"]

    plt.figure(figsize=(7,7))

    plt.scatter(
        actual,
        predicted
    )

    plt.xlabel("Actual CPU")

    plt.ylabel("Predicted CPU")

    plt.title("Actual vs Predicted")

    plt.tight_layout()

    plt.savefig(
        "results/graphs/actual_vs_predicted.png",
        dpi=300
    )

    plt.close()


# ======================================================
# Model Comparison
# ======================================================

def model_comparison(evaluation_results):

    metrics = evaluation_results["metrics"]

    plt.figure(figsize=(8,5))

    plt.bar(
        metrics["Model"],
        metrics["RMSE"]
    )

    plt.title("RMSE Comparison")

    plt.ylabel("RMSE")

    plt.tight_layout()

    plt.savefig(
        "results/graphs/rmse_comparison.png",
        dpi=300
    )

    plt.close()


# ======================================================
# CPU Distribution
# ======================================================

def cpu_distribution(dataset):

    plt.figure(figsize=(8,5))

    plt.hist(
        dataset["plan_cpu"],
        bins=30
    )

    plt.xlabel("CPU")

    plt.ylabel("Frequency")

    plt.title("CPU Workload Distribution")

    plt.tight_layout()

    plt.savefig(
        "results/graphs/cpu_distribution.png",
        dpi=300
    )

    plt.close()


# ======================================================
# Correlation Matrix
# ======================================================

def correlation_matrix(dataset):

    corr = dataset.corr(numeric_only=True)

    plt.figure(figsize=(10,8))

    plt.imshow(
        corr,
        aspect='auto'
    )

    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns
    )

    plt.title("Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        "results/graphs/correlation_matrix.png",
        dpi=300
    )

    plt.close()


# ======================================================
# Cloud Provider Comparison
# ======================================================

def cloud_provider_graph(
        cloud_pricing,
        latency_dataset):

    merged = pd.merge(
        cloud_pricing,
        latency_dataset,
        on="provider"
    )

    cost_column = None

    for col in merged.columns:

        name = col.lower()

        if "cost" in name or "price" in name:
            cost_column = col
            break

    plt.figure(figsize=(7,5))

    plt.bar(
        merged["provider"],
        merged[cost_column]
    )

    plt.title("Cloud Provider Cost Comparison")

    plt.xlabel("Cloud Provider")

    plt.ylabel("Cost")

    plt.tight_layout()

    plt.savefig(
        "results/graphs/cloud_cost_comparison.png",
        dpi=300
    )

    plt.close()