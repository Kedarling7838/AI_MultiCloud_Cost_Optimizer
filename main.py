"""
==========================================================
AI-Driven Framework for Workload Placement and
Cost-Performance Optimization in Multi-Cloud Systems

main.py
==========================================================
"""

import pandas as pd

from preprocessing import preprocess_data
from feature_engineering import feature_engineering
from model import train_models
from evaluation import evaluate_models
from cloud_selection import recommend_cloud
from visualization import generate_all_graphs


def load_datasets():

    print("=" * 60)
    print("Loading Datasets...")
    print("=" * 60)

    batch_task = pd.read_csv(
        "datasets/batch_task_20000_with_headers.csv"
    )

    machine_meta = pd.read_csv(
        "datasets/machine_meta_with_headers.csv"
    )

    cloud_pricing = pd.read_csv(
        "datasets/cloud_pricing_large.csv"
    )

    latency = pd.read_csv(
        "datasets/latency_availability_large_5000.csv"
    )

    print(f"✓ Batch Task Dataset      : {batch_task.shape}")
    print(f"✓ Machine Meta Dataset    : {machine_meta.shape}")
    print(f"✓ Cloud Pricing Dataset   : {cloud_pricing.shape}")
    print(f"✓ Latency Dataset         : {latency.shape}")

    return (
        batch_task,
        machine_meta,
        cloud_pricing,
        latency
    )


def main():

    print("=" * 60)
    print("AI-Driven Multi-Cloud Workload Placement")
    print("=" * 60)

    (
        batch_task,
        machine_meta,
        cloud_pricing,
        latency
    ) = load_datasets()

    print("\nPreprocessing Data...")
    merged_dataset = preprocess_data(
        batch_task,
        machine_meta,
        cloud_pricing,
        latency
    )

    print("\nPerforming Feature Engineering...")
    processed = feature_engineering(
        merged_dataset
    )

    print("\nTraining Machine Learning Models...")
    model_results = train_models(
        processed
    )

    print("\nEvaluating Models...")
    evaluation_results = evaluate_models(
        model_results
    )

    print("\nSelecting Best Cloud Provider...")
    best_provider = recommend_cloud(
        cloud_pricing,
        latency,
        model_results
    )

    print("\nGenerating Graphs...")
    generate_all_graphs(
        processed["dataset"],
        model_results,
        evaluation_results,
        cloud_pricing,
        latency
    )

    print("\n")
    print("=" * 60)
    print("FINAL RESULT SUMMARY")
    print("=" * 60)

    print(evaluation_results["metrics"])

    print("\nRecommended Cloud Provider")
    print(best_provider)

    print("\nProject Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()