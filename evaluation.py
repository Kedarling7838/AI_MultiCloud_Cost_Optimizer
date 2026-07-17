# Evaluation metrics
"""
==========================================================
AI-Driven Multi-Cloud Workload Placement
evaluation.py
==========================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def mean_absolute_percentage_error(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    epsilon = 1e-10

    return np.mean(
        np.abs(
            (y_true - y_pred) /
            np.maximum(np.abs(y_true), epsilon)
        )
    ) * 100


def evaluate_models(model_results):

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    y_test = model_results["y_test"]

    xgb_prediction = model_results["xgb_prediction"]
    rf_prediction = model_results["rf_prediction"]

    # ----------------------------------------
    # XGBoost Metrics
    # ----------------------------------------

    xgb_mae = mean_absolute_error(
        y_test,
        xgb_prediction
    )

    xgb_mse = mean_squared_error(
        y_test,
        xgb_prediction
    )

    xgb_rmse = np.sqrt(xgb_mse)

    xgb_r2 = r2_score(
        y_test,
        xgb_prediction
    )

    xgb_mape = mean_absolute_percentage_error(
        y_test,
        xgb_prediction
    )

    # ----------------------------------------
    # Random Forest Metrics
    # ----------------------------------------

    rf_mae = mean_absolute_error(
        y_test,
        rf_prediction
    )

    rf_mse = mean_squared_error(
        y_test,
        rf_prediction
    )

    rf_rmse = np.sqrt(rf_mse)

    rf_r2 = r2_score(
        y_test,
        rf_prediction
    )

    rf_mape = mean_absolute_percentage_error(
        y_test,
        rf_prediction
    )

    # ----------------------------------------
    # Results Table
    # ----------------------------------------

    results = pd.DataFrame({

        "Model": [

            "XGBoost",

            "Random Forest"

        ],

        "MAE": [

            xgb_mae,

            rf_mae

        ],

        "MSE": [

            xgb_mse,

            rf_mse

        ],

        "RMSE": [

            xgb_rmse,

            rf_rmse

        ],

        "R2 Score": [

            xgb_r2,

            rf_r2

        ],

        "MAPE (%)": [

            xgb_mape,

            rf_mape

        ]

    })

    print(results)

    # ----------------------------------------
    # Save Results
    # ----------------------------------------

    os.makedirs("results", exist_ok=True)

    results.to_csv(

        "results/model_results.csv",

        index=False

    )

    # ----------------------------------------
    # Save Predictions
    # ----------------------------------------

    prediction_table = pd.DataFrame({

        "Actual": y_test.values,

        "XGBoost Prediction": xgb_prediction,

        "Random Forest Prediction": rf_prediction

    })

    prediction_table.to_csv(

        "results/predictions.csv",

        index=False

    )

    print()

    print("Evaluation Results Saved")

    print("results/model_results.csv")

    print("results/predictions.csv")

    print("=" * 60)

    return {

        "metrics": results,

        "prediction_table": prediction_table

    }