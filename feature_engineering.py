# Feature engineering
"""
==========================================================
AI-Driven Multi-Cloud Workload Placement
feature_engineering.py
==========================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def feature_engineering(dataset):

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # ---------------------------------------------
    # Remove unnecessary columns
    # ---------------------------------------------

    remove_columns = [
        "job_id",
        "task_status"
    ]

    for column in remove_columns:

        if column in dataset.columns:

            dataset.drop(column,
                         axis=1,
                         inplace=True)

    print("Unused Columns Removed")

    # ---------------------------------------------
    # Create Execution Time
    # ---------------------------------------------

    if "start_time" in dataset.columns and \
       "end_time" in dataset.columns:

        dataset["execution_time"] = (
            dataset["end_time"] -
            dataset["start_time"]
        )

    # ---------------------------------------------
    # Resource Efficiency
    # ---------------------------------------------

    if "cpu_num" in dataset.columns:

        dataset["cpu_efficiency"] = (
            dataset["plan_cpu"] /
            (dataset["cpu_num"] + 1)
        )

    if "memory_size" in dataset.columns:

        dataset["memory_efficiency"] = (
            dataset["plan_mem"] /
            (dataset["memory_size"] + 1)
        )

    # ---------------------------------------------
    # Cost Feature
    # ---------------------------------------------

    if "hourly_vm_cost_usd" in dataset.columns:

        dataset["estimated_cost"] = (
            dataset["plan_cpu"] *
            dataset["hourly_vm_cost_usd"]
        )

    # ---------------------------------------------
    # Latency Score
    # ---------------------------------------------

    if "network_latency_ms" in dataset.columns:

        dataset["latency_score"] = (
            100 /
            (dataset["network_latency_ms"] + 1)
        )

    # ---------------------------------------------
    # Availability Score
    # ---------------------------------------------

    if "availability_percent" in dataset.columns:

        dataset["availability_score"] = (
            dataset["availability_percent"] / 100
        )

    # ---------------------------------------------
    # Remove Missing Values
    # ---------------------------------------------

    dataset = dataset.dropna()

    print("Feature Engineering Completed")

    # ---------------------------------------------
    # Define Target Variable
    # ---------------------------------------------

    target = "plan_cpu"

    X = dataset.drop(target,
                     axis=1)

    y = dataset[target]

    print()

    print("Feature Matrix Shape :", X.shape)

    print("Target Shape :", y.shape)

    # ---------------------------------------------
    # Train Test Split
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        shuffle=True

    )

    print()

    print("Training Samples :", len(X_train))

    print("Testing Samples :", len(X_test))

    print()

    print("=" * 60)

    return {

        "dataset": dataset,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test

    }