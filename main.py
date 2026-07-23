"""
==========================================================
AI-Driven Intelligent Decision Support Framework
Main Execution File
==========================================================
"""

import pandas as pd

from preprocessing import preprocess_data
from feature_engineering import feature_engineering
from model import train_models
from evaluation import evaluate_models
from cloud_selection import recommend_cloud
from visualization import generate_all_graphs


# ----------------------------------------------------------
# Load Datasets
# ----------------------------------------------------------

def load_datasets():

    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    batch_task = pd.read_csv(
        "datasets/batch_task_20000_with_headers.csv"
    )

    machine_meta = pd.read_csv(
        "datasets/machine_meta_with_headers.csv"
    )

    cloud_pricing = pd.read_csv(
        "datasets/cloud_pricing_large.csv"
    )

    latency_dataset = pd.read_csv(
        "datasets/latency_availability_large_5000.csv"
    )

    print(f"✓ Batch Task Dataset     : {batch_task.shape}")
    print(f"✓ Machine Meta Dataset   : {machine_meta.shape}")
    print(f"✓ Cloud Pricing Dataset  : {cloud_pricing.shape}")
    print(f"✓ Latency Dataset        : {latency_dataset.shape}")

    return (

        batch_task,

        machine_meta,

        cloud_pricing,

        latency_dataset

    )


# ----------------------------------------------------------
# Main Function
# ----------------------------------------------------------

def main():

    print("=" * 70)
    print("AI-DRIVEN INTELLIGENT DECISION SUPPORT FRAMEWORK")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Data
    # ------------------------------------------------------

    (

        batch_task,

        machine_meta,

        cloud_pricing,

        latency_dataset

    ) = load_datasets()

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    print("\nPreprocessing Dataset...\n")

    merged_dataset = preprocess_data(

        batch_task,

        machine_meta,

        cloud_pricing,

        latency_dataset

    )

    # ------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------

    print("\nPerforming Feature Engineering...\n")

    processed = feature_engineering(

        merged_dataset

    )

    # ------------------------------------------------------
    # Model Training
    # ------------------------------------------------------

    print("\nTraining Machine Learning Models...\n")

    model_results = train_models(

        processed

    )

    # ------------------------------------------------------
    # Model Evaluation
    # ------------------------------------------------------

    print("\nEvaluating Models...\n")

    evaluation_results = evaluate_models(

        model_results

    )

    # ------------------------------------------------------
    # Cloud Selection
    # ------------------------------------------------------

    print("\nSelecting Best Cloud Provider...\n")

    cloud_results = recommend_cloud(

        cloud_pricing,

        latency_dataset,

        model_results

    )

    # ------------------------------------------------------
    # Generate Graphs
    # ------------------------------------------------------

    print("\nGenerating Research Graphs...\n")

    generate_all_graphs(

        processed["dataset"],

        model_results,

        evaluation_results,

        cloud_results

    )

    # ------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL PROJECT SUMMARY")
    print("=" * 70)

    print("\nBest Machine Learning Model")

    print(evaluation_results["best_model"])

    print("\nBest Cloud Provider")

    best_provider = cloud_results["best_provider"]

    print(f"Provider : {best_provider['provider']}")

    if "region" in best_provider.index:
        print(f"Region   : {best_provider['region']}")

    print(f"Score    : {best_provider['Provider_Score']:.4f}")

    print("\nGenerated Files")

    print("-----------------------------")

    print("results/evaluation_results.csv")
    print("results/model_results.csv")
    print("results/predictions.csv")
    print("results/statistical_results.csv")
    print("results/feature_importance.csv")
    print("results/cloud_provider_ranking.csv")
    print("results/entropy_weights.csv")
    print("results/best_provider.csv")
    print("results/graphs/")

    print("\n✓ Project Completed Successfully.")
    print("=" * 70)


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------

if __name__ == "__main__":

    main()