# Data loading and preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(batch_task,
                    machine_meta,
                    cloud_pricing,
                    latency):

    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------
    # Remove Duplicate Records
    # --------------------------------------------------

    batch_task = batch_task.drop_duplicates()

    machine_meta = machine_meta.drop_duplicates()

    cloud_pricing = cloud_pricing.drop_duplicates()

    latency = latency.drop_duplicates()

    print("Duplicate Records Removed")

    # --------------------------------------------------
    # Handle Missing Values
    # --------------------------------------------------

    for df in [batch_task,
               machine_meta,
               cloud_pricing,
               latency]:

        numeric_columns = df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        categorical_columns = df.select_dtypes(
            include=["object"]
        ).columns

        for col in numeric_columns:

            df[col].fillna(df[col].median(),
                           inplace=True)

        for col in categorical_columns:

            df[col].fillna(df[col].mode()[0],
                           inplace=True)

    print("Missing Values Filled")

    # --------------------------------------------------
    # Encode Categorical Variables
    # --------------------------------------------------

    encoder = LabelEncoder()

    categorical_tables = [
        batch_task,
        machine_meta,
        cloud_pricing,
        latency
    ]

    for table in categorical_tables:

        object_columns = table.select_dtypes(
            include=["object"]
        ).columns

        for column in object_columns:

            table[column] = encoder.fit_transform(
                table[column].astype(str)
            )

    print("Categorical Encoding Completed")

    # --------------------------------------------------
    # Rename Columns for Merge
    # --------------------------------------------------

    if "machine_id" in batch_task.columns and \
       "machine_id" in machine_meta.columns:

        merged = pd.merge(
            batch_task,
            machine_meta,
            on="machine_id",
            how="left"
        )

    else:

        merged = batch_task.copy()

    # --------------------------------------------------
    # Merge Cloud Pricing
    # --------------------------------------------------

    pricing_sample = cloud_pricing.sample(
        n=len(merged),
        replace=True,
        random_state=42
    ).reset_index(drop=True)

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

    print("Datasets Successfully Merged")

    # --------------------------------------------------
    # Remove Duplicate Columns
    # --------------------------------------------------

    merged = merged.loc[
        :,
        ~merged.columns.duplicated()
    ]

    # --------------------------------------------------
    # Standardization
    # --------------------------------------------------

    scaler = StandardScaler()

    numeric_columns = merged.select_dtypes(
        include=["int64", "float64"]
    ).columns

    merged[numeric_columns] = scaler.fit_transform(
        merged[numeric_columns]
    )

    print("Feature Scaling Completed")

    # --------------------------------------------------
    # Dataset Information
    # --------------------------------------------------

    print()

    print("Final Dataset Shape")

    print(merged.shape)

    print()

    print("First Five Records")

    print(merged.head())

    print()

    print("Missing Values")

    print(merged.isnull().sum())

    print()

    print("=" * 60)

    return merged