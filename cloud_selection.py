"""
==========================================================
AI-Driven Intelligent Decision Support Framework
Entropy-Based Cloud Selection
==========================================================
"""

import os
import numpy as np
import pandas as pd


# ----------------------------------------------------------
# Normalize Decision Matrix
# ----------------------------------------------------------

def normalize_matrix(df, columns):

    normalized = df.copy()

    for col in columns:

        minimum = df[col].min()
        maximum = df[col].max()

        if maximum == minimum:

            normalized[col] = 0

        else:

            normalized[col] = (

                df[col] - minimum

            ) / (

                maximum - minimum

            )

    return normalized


# ----------------------------------------------------------
# Entropy Weight Calculation
# ----------------------------------------------------------

def entropy_weighting(df, columns):

    data = df[columns].copy()

    data = data + 1e-12

    probability = data.div(

        data.sum(axis=0),

        axis=1

    )

    n = len(data)

    k = 1 / np.log(n)

    entropy = -k * (

        probability *

        np.log(probability)

    ).sum(axis=0)

    diversification = 1 - entropy

    weights = diversification / diversification.sum()

    weight_table = pd.DataFrame({

        "Criteria": columns,

        "Entropy": entropy,

        "Weight": weights

    })

    return weights.to_dict(), weight_table


# ----------------------------------------------------------
# Cloud Recommendation
# ----------------------------------------------------------

def recommend_cloud(

        cloud_pricing,

        latency_dataset,

        model_results):

    print("=" * 70)
    print("ENTROPY-BASED CLOUD PROVIDER SELECTION")
    print("=" * 70)

    os.makedirs(

        "results",

        exist_ok=True

    )

    # ----------------------------------------------------------
    # Merge Cloud Pricing and Latency Dataset
    # ----------------------------------------------------------

    merge_columns = ["provider"]

    if "region" in cloud_pricing.columns and "region" in latency_dataset.columns:
        merge_columns.append("region")

    providers = pd.merge(

        cloud_pricing,

        latency_dataset,

        on=merge_columns,

        how="inner"

    )

    providers = providers.drop_duplicates()

    print(f"\nMerged Provider Records : {len(providers)}")

    # ----------------------------------------------------------
    # Detect Required Columns Automatically
    # ----------------------------------------------------------

    cost_column = None
    latency_column = None
    availability_column = None

    for col in providers.columns:

        name = col.lower()

        if ("hourly" in name or
            "cost" in name or
            "price" in name):

            if cost_column is None:
                cost_column = col

        elif "latency" in name:

            if latency_column is None:
                latency_column = col

        elif "availability" in name:

            if availability_column is None:
                availability_column = col

    if cost_column is None:
        raise ValueError("Cost column not found.")

    if latency_column is None:
        raise ValueError("Latency column not found.")

    if availability_column is None:
        raise ValueError("Availability column not found.")

    print("\nDetected Columns")
    print("---------------------------")
    print("Cost Column         :", cost_column)
    print("Latency Column      :", latency_column)
    print("Availability Column :", availability_column)

    # ----------------------------------------------------------
    # Convert Cost and Latency to Benefit Criteria
    # ----------------------------------------------------------

    providers["Cost_Score"] = (

        providers[cost_column].max()

        -

        providers[cost_column]

    )

    providers["Latency_Score"] = (

        providers[latency_column].max()

        -

        providers[latency_column]

    )

    providers["Availability_Score"] = (

        providers[availability_column]

    )

    criteria = [

        "Cost_Score",

        "Latency_Score",

        "Availability_Score"

    ]

    normalized = normalize_matrix(

        providers,

        criteria

    )

    weights, weight_table = entropy_weighting(

        normalized,

        criteria

    )

    print("\n")
    print("=" * 70)
    print("ENTROPY WEIGHTS")
    print("=" * 70)

    print(weight_table)
        # ----------------------------------------------------------
    # Calculate Provider Score
    # ----------------------------------------------------------

    providers["Provider_Score"] = (

        normalized["Cost_Score"] * weights["Cost_Score"]

        +

        normalized["Latency_Score"] * weights["Latency_Score"]

        +

        normalized["Availability_Score"] * weights["Availability_Score"]

    )

    # ----------------------------------------------------------
    # Predicted CPU Requirement
    # ----------------------------------------------------------

    predicted_cpu = np.mean(

        model_results["xgb_prediction"]

    )

    print("\nPredicted Average CPU Requirement :")

    print(round(predicted_cpu, 4))

    # ----------------------------------------------------------
    # Optional Capacity Filtering
    # ----------------------------------------------------------

    cpu_column = None

    for col in providers.columns:

        name = col.lower()

        if "cpu_capacity" in name:

            cpu_column = col

            break

    if cpu_column is not None:

        providers = providers[
            providers[cpu_column] >= predicted_cpu
        ]

        print(f"\nProviders after CPU filtering : {len(providers)}")

    # ----------------------------------------------------------
    # Rank Providers
    # ----------------------------------------------------------

    providers = providers.sort_values(

        by="Provider_Score",

        ascending=False

    ).reset_index(drop=True)

    providers["Rank"] = np.arange(

        1,

        len(providers) + 1

    )

    # ----------------------------------------------------------
    # Best Provider
    # ----------------------------------------------------------

    best_provider = providers.iloc[0]

    print("\n")
    print("=" * 70)
    print("CLOUD PROVIDER RANKING")
    print("=" * 70)

    display_columns = [

        "Rank",

        "provider"

    ]

    if "region" in providers.columns:
        display_columns.append("region")

    display_columns.extend([

        cost_column,

        latency_column,

        availability_column,

        "Provider_Score"

    ])

    print(

        providers[display_columns]

    )

    print("\n")
    print("=" * 70)
    print("BEST CLOUD PROVIDER")
    print("=" * 70)

    print(f"Provider      : {best_provider['provider']}")

    if "region" in providers.columns:
        print(f"Region        : {best_provider['region']}")

    print(f"Rank          : {best_provider['Rank']}")
    print(f"Score         : {best_provider['Provider_Score']:.4f}")
    print(f"Cost          : {best_provider[cost_column]}")
    print(f"Latency       : {best_provider[latency_column]}")
    print(f"Availability  : {best_provider[availability_column]}")

    # ----------------------------------------------------------
    # Save Results
    # ----------------------------------------------------------

    providers.to_csv(

        "results/cloud_provider_ranking.csv",

        index=False

    )

    weight_table.to_csv(

        "results/entropy_weights.csv",

        index=False

    )

    best_provider_df = pd.DataFrame({

        "Provider": [

            best_provider["provider"]

        ],

        "Region": [

            best_provider["region"]

            if "region" in providers.columns else "N/A"

        ],

        "Provider Score": [

            best_provider["Provider_Score"]

        ],

        "Cost": [

            best_provider[cost_column]

        ],

        "Latency": [

            best_provider[latency_column]

        ],

        "Availability": [

            best_provider[availability_column]

        ]

    })

    best_provider_df.to_csv(

        "results/best_provider.csv",

        index=False

    )

    print("\n")
    print("=" * 70)
    print("FILES GENERATED")
    print("=" * 70)

    print("results/cloud_provider_ranking.csv")
    print("results/entropy_weights.csv")
    print("results/best_provider.csv")

    print("=" * 70)

    # ----------------------------------------------------------
    # Return Results
    # ----------------------------------------------------------

    return {

        "best_provider": best_provider,

        "provider_ranking": providers,

        "entropy_weights": weight_table,

        "predicted_cpu": predicted_cpu

    }


# ----------------------------------------------------------
# Standalone Testing
# ----------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("CLOUD SELECTION MODULE")
    print("=" * 70)

    print(
        "\nRun main.py after model training to "
        "perform entropy-based cloud selection."
    )

    print("\nCloud Selection module loaded successfully.")