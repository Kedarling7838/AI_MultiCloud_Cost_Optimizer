"""
==========================================================
AI-Driven Intelligent Decision Support Framework
Data Preprocessing
==========================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(batch_task,
                    machine_meta,
                    cloud_pricing,
                    latency):

    print("=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    # ----------------------------------------------------
    # Dataset Information
    # ----------------------------------------------------

    print("\nOriginal Dataset Sizes")

    print(f"Batch Task      : {batch_task.shape}")
    print(f"Machine Meta    : {machine_meta.shape}")
    print(f"Cloud Pricing   : {cloud_pricing.shape}")
    print(f"Latency Dataset : {latency.shape}")

    # ----------------------------------------------------
    # Remove Duplicate Records
    # ----------------------------------------------------

    duplicate_info = {

        "batch_task": batch_task.duplicated().sum(),

        "machine_meta": machine_meta.duplicated().sum(),

        "cloud_pricing": cloud_pricing.duplicated().sum(),

        "latency": latency.duplicated().sum()

    }

    batch_task = batch_task.drop_duplicates()

    machine_meta = machine_meta.drop_duplicates()

    cloud_pricing = cloud_pricing.drop_duplicates()

    latency = latency.drop_duplicates()

    print("\nDuplicate Records Removed")

    print(duplicate_info)

    # ----------------------------------------------------
    # Missing Value Handling
    # ----------------------------------------------------

    datasets = [
        batch_task,
        machine_meta,
        cloud_pricing,
        latency
    ]

    for df in datasets:

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        categorical_columns = df.select_dtypes(
            include="object"
        ).columns

        for col in numeric_columns:

            df[col] = df[col].fillna(df[col].median())

        for col in categorical_columns:

            df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing Values Filled")

    # ----------------------------------------------------
    # Encode Categorical Columns
    # ----------------------------------------------------

    encoder = LabelEncoder()

    for table in datasets:

        object_columns = table.select_dtypes(
            include="object"
        ).columns

        for column in object_columns:

            table[column] = encoder.fit_transform(
                table[column].astype(str)
            )

    print("Categorical Encoding Completed")

    # ----------------------------------------------------
    # Merge Batch Task & Machine Metadata
    # ----------------------------------------------------

    if (
        "machine_id" in batch_task.columns
        and
        "machine_id" in machine_meta.columns
    ):

        merged = pd.merge(
            batch_task,
            machine_meta,
            on="machine_id",
            how="left"
        )

    else:

        merged = batch_task.copy()

    print("\nBatch Task + Machine Meta merged")
        # ----------------------------------------------------
    # Merge Cloud Pricing
    # ----------------------------------------------------

    pricing_sample = cloud_pricing.sample(
        n=len(merged),
        replace=True,
        random_state=42
    ).reset_index(drop=True)

    # ----------------------------------------------------
    # Merge Latency Dataset
    # ----------------------------------------------------

    latency_sample = latency.sample(
        n=len(merged),
        replace=True,
        random_state=42
    ).reset_index(drop=True)

    merged = merged.reset_index(drop=True)

    merged = pd.concat(
        [
            merged,
            pricing_sample,
            latency_sample
        ],
        axis=1
    )

    print("Cloud Pricing Merged")

    print("Latency Dataset Merged")

    # ----------------------------------------------------
    # Remove Duplicate Columns
    # ----------------------------------------------------

    merged = merged.loc[:, ~merged.columns.duplicated()]

    print("Duplicate Columns Removed")

    # ----------------------------------------------------
    # Preserve Cloud Metrics
    # ----------------------------------------------------

    protected_columns = [

        "hourly_vm_cost_usd",
        "storage_cost_gb_month_usd",
        "data_transfer_cost_gb_usd",

        "network_latency_ms",
        "response_time_ms",
        "availability_percent",
        "resource_utilization_percent"

    ]

    protected_columns = [

        col

        for col in protected_columns

        if col in merged.columns

    ]

    original_cloud_metrics = merged[
        protected_columns
    ].copy()

    # ----------------------------------------------------
    # Select Numeric Features
    # ----------------------------------------------------

    numeric_columns = merged.select_dtypes(
        include=np.number
    ).columns.tolist()

    feature_columns = [

        col

        for col in numeric_columns

        if col not in protected_columns

    ]

    # ----------------------------------------------------
    # Remove Target Variable
    # ----------------------------------------------------

    if "plan_cpu" in feature_columns:

        feature_columns.remove("plan_cpu")

    # ----------------------------------------------------
    # Feature Scaling
    # ----------------------------------------------------

    scaler = StandardScaler()

    if len(feature_columns) > 0:

        merged[feature_columns] = scaler.fit_transform(

            merged[feature_columns]

        )

    # ----------------------------------------------------
    # Restore Original Cloud Metrics
    # ----------------------------------------------------

    for col in protected_columns:

        merged[col] = original_cloud_metrics[col]

    print("\nFeature Scaling Completed")

    print(f"Scaled Features : {len(feature_columns)}")

    print(f"Protected Cloud Metrics : {len(protected_columns)}")

    # ----------------------------------------------------
    # Dataset Statistics
    # ----------------------------------------------------

    statistics = pd.DataFrame({

        "Feature": merged.columns,

        "Missing Values":
            merged.isnull().sum().values,

        "Unique Values":
            merged.nunique().values,

        "Data Type":
            merged.dtypes.astype(str).values

    })

    print("\nDataset Statistics Generated")

    # ----------------------------------------------------
    # Numerical Summary
    # ----------------------------------------------------

    numerical_summary = merged.describe(include="all").T

    numerical_summary.to_csv(
        "results/numerical_summary.csv"
    )

    print("Numerical Summary Saved")
        # ----------------------------------------------------
    # Save Dataset Statistics
    # ----------------------------------------------------

    statistics.to_csv(
        "results/dataset_statistics.csv",
        index=False
    )

    print("Dataset Statistics Saved")

    # ----------------------------------------------------
    # Save Preprocessed Dataset
    # ----------------------------------------------------

    merged.to_csv(
        "results/preprocessed_dataset.csv",
        index=False
    )

    print("Preprocessed Dataset Saved")

    # ----------------------------------------------------
    # Display Summary
    # ----------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(f"Final Dataset Shape      : {merged.shape}")
    print(f"Total Features           : {merged.shape[1]}")
    print(f"Scaled Features          : {len(feature_columns)}")
    print(f"Protected Cloud Metrics  : {len(protected_columns)}")
    print(f"Missing Values Remaining : {merged.isnull().sum().sum()}")

    print("=" * 70)

    # ----------------------------------------------------
    # Return Processed Dataset
    # ----------------------------------------------------

    return merged


# ==========================================================
# Standalone Testing
# ==========================================================

if __name__ == "__main__":

    print("\nRunning Preprocessing Module...\n")

    batch_task = pd.read_csv(
        "data/batch_task_20000_with_headers.csv"
    )

    machine_meta = pd.read_csv(
        "data/machine_meta_with_headers.csv"
    )

    cloud_pricing = pd.read_csv(
        "data/cloud_pricing_large.csv"
    )

    latency = pd.read_csv(
        "data/latency_availability_large_5000.csv"
    )

    processed_dataset = preprocess_data(
        batch_task=batch_task,
        machine_meta=machine_meta,
        cloud_pricing=cloud_pricing,
        latency=latency
    )

    print("\nReturned Dataset Shape:")
    print(processed_dataset.shape)

    print("\nFirst Five Rows:")
    print(processed_dataset.head())

    print("\nPreprocessing Completed Successfully.")