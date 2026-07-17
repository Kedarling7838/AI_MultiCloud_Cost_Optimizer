# Multi-cloud workload placement algorithm
"""
==========================================================
AI-Driven Multi-Cloud Workload Placement
cloud_selection.py
==========================================================
"""

import pandas as pd
import numpy as np


def calculate_provider_score(cost,
                             latency,
                             availability):

    """
    Lower score indicates a better cloud provider.

    Provider Score =
    0.5 × Cost +
    0.3 × Latency +
    0.2 × (100 - Availability)
    """

    availability_penalty = 100 - availability

    score = (

        0.50 * cost +

        0.30 * latency +

        0.20 * availability_penalty

    )

    return score


def recommend_cloud(cloud_pricing,
                    latency_dataset,
                    model_results):

    print("=" * 60)
    print("MULTI-CLOUD PLACEMENT")
    print("=" * 60)

    # ---------------------------------------------
    # Predicted CPU Workload
    # ---------------------------------------------

    predicted_cpu = np.mean(
        model_results["xgb_prediction"]
    )

    print(f"Average Predicted CPU Workload : {predicted_cpu:.2f}")

    # ---------------------------------------------
    # Merge Pricing & Latency
    # ---------------------------------------------

    providers = pd.merge(

        cloud_pricing,

        latency_dataset,

        on="provider",

        how="inner"

    )

    # ---------------------------------------------
    # Detect Required Columns
    # ---------------------------------------------

    cost_column = None
    latency_column = None
    availability_column = None

    for col in providers.columns:

        name = col.lower()

        if "cost" in name or "price" in name:
            cost_column = col

        elif "latency" in name:
            latency_column = col

        elif "availability" in name:
            availability_column = col

    if cost_column is None:
        raise Exception("Cost column not found.")

    if latency_column is None:
        raise Exception("Latency column not found.")

    if availability_column is None:
        raise Exception("Availability column not found.")

    # ---------------------------------------------
    # Calculate Score
    # ---------------------------------------------

    scores = []

    for _, row in providers.iterrows():

        score = calculate_provider_score(

            row[cost_column],

            row[latency_column],

            row[availability_column]

        )

        scores.append(score)

    providers["Provider_Score"] = scores

    # ---------------------------------------------
    # Capacity Check
    # ---------------------------------------------

    if "cpu_capacity" in providers.columns:

        providers = providers[
            providers["cpu_capacity"] >= predicted_cpu
        ]

    # ---------------------------------------------
    # Sort Providers
    # ---------------------------------------------

    providers = providers.sort_values(

        by="Provider_Score",

        ascending=True

    )

    best_provider = providers.iloc[0]

    # ---------------------------------------------
    # Display Ranking
    # ---------------------------------------------

    print()

    print("Cloud Provider Ranking")

    print()

    print(

        providers[

            [

                "provider",

                cost_column,

                latency_column,

                availability_column,

                "Provider_Score"

            ]

        ]

    )

    print()

    print("=" * 60)

    print("Recommended Cloud Provider")

    print("=" * 60)

    print(f"Provider      : {best_provider['provider']}")

    print(f"Score         : {best_provider['Provider_Score']:.4f}")

    print(f"Cost          : {best_provider[cost_column]}")

    print(f"Latency       : {best_provider[latency_column]}")

    print(f"Availability  : {best_provider[availability_column]}")

    print("=" * 60)

    # ---------------------------------------------
    # Save Ranking
    # ---------------------------------------------

    providers.to_csv(

        "results/cloud_provider_ranking.csv",

        index=False

    )

    return best_provider