"""
==========================================================
AI-Driven Intelligent Decision Support Framework
Feature Engineering
==========================================================
"""

import pandas as pd
import numpy as np


def feature_engineering(dataset):

    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    dataset = dataset.copy()

    # --------------------------------------------------
    # Remove Unnecessary Columns
    # --------------------------------------------------

    remove_columns = [
        "job_id",
        "task_status"
    ]

    for col in remove_columns:
        if col in dataset.columns:
            dataset.drop(columns=col, inplace=True)

    print("✓ Unnecessary columns removed")

    # --------------------------------------------------
    # Execution Time Feature
    # --------------------------------------------------

    if (
        "start_time" in dataset.columns
        and
        "end_time" in dataset.columns
    ):

        dataset["execution_time"] = (
            dataset["end_time"] -
            dataset["start_time"]
        )

        dataset["execution_time"] = dataset[
            "execution_time"
        ].clip(lower=0)

    # --------------------------------------------------
    # Memory / CPU Ratio
    # --------------------------------------------------

    if (
        "plan_mem" in dataset.columns
        and
        "plan_cpu" in dataset.columns
    ):

        dataset["memory_cpu_ratio"] = (

            dataset["plan_mem"] /

            (dataset["plan_cpu"] + 1)

        )

    # --------------------------------------------------
    # Latency Score
    # --------------------------------------------------

    if "network_latency_ms" in dataset.columns:

        dataset["latency_score"] = (

            100 /

            (dataset["network_latency_ms"] + 1)

        )

    # --------------------------------------------------
    # Availability Score
    # --------------------------------------------------

    if "availability_percent" in dataset.columns:

        dataset["availability_score"] = (

            dataset["availability_percent"] / 100

        )

    # --------------------------------------------------
    # Replace Infinite Values
    # --------------------------------------------------

    dataset.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # --------------------------------------------------
    # Fill Missing Values
    # --------------------------------------------------

    numeric_columns = dataset.select_dtypes(
        include=np.number
    ).columns

    dataset[numeric_columns] = dataset[
        numeric_columns
    ].fillna(

        dataset[numeric_columns].median()

    )

    dataset.dropna(inplace=True)

    # --------------------------------------------------
    # Target Variable
    # --------------------------------------------------

    target = "plan_cpu"

    if target not in dataset.columns:

        raise ValueError(
            f"{target} column not found."
        )

    # --------------------------------------------------
    # Remove Target Leakage Columns
    # --------------------------------------------------

    leakage_columns = [

        "cpu_efficiency",
        "estimated_cost"

    ]

    for col in leakage_columns:

        if col in dataset.columns:

            dataset.drop(
                columns=col,
                inplace=True
            )

    # --------------------------------------------------
    # Prepare Features
    # --------------------------------------------------

    X = dataset.drop(
        columns=[target]
    )

    y = dataset[target]

    # --------------------------------------------------
    # Dataset Information
    # --------------------------------------------------

    print()

    print(f"Dataset Shape       : {dataset.shape}")

    print(f"Number of Features : {X.shape[1]}")

    print(f"Number of Samples  : {X.shape[0]}")

    print(f"Target Variable    : {target}")

    print("\nFeature List")

    for feature in X.columns:

        print(f"  - {feature}")

    print("=" * 70)

    return {

        "dataset": dataset,

        "X": X,

        "y": y,

        "feature_names": list(X.columns),

        "target": target

    }


# ------------------------------------------------------
# Standalone Testing
# ------------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv(
        "results/preprocessed_dataset.csv"
    )

    result = feature_engineering(df)

    print("\nReturned Keys")

    print(result.keys())

    print("\nFeature Engineering Completed Successfully.")