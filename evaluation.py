"""
==========================================================
AI-Driven Intelligent Decision Support Framework
Model Evaluation
==========================================================
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import ttest_rel

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ----------------------------------------------------------
# Mean Absolute Percentage Error
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# Calculate Evaluation Metrics
# ----------------------------------------------------------

def calculate_metrics(
        y_true,
        prediction):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    mse = mean_squared_error(
        y_true,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        prediction
    )

    mape = mean_absolute_percentage_error(
        y_true,
        prediction
    )

    return {

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse,

        "R2": r2,

        "MAPE": mape

    }


# ----------------------------------------------------------
# Evaluate Models
# ----------------------------------------------------------

def evaluate_models(model_results):

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    os.makedirs(
        "results",
        exist_ok=True
    )

    y_test = model_results["y_test"]

    predictions = {

        "XGBoost":
            model_results["xgb_prediction"],

        "Random Forest":
            model_results["rf_prediction"],

        "Decision Tree":
            model_results["dt_prediction"],

        "Gradient Boosting":
            model_results["gb_prediction"],

        "Extra Trees":
            model_results["et_prediction"]

    }

    metrics_table = []

    print("\nEvaluating Models...\n")

    # ----------------------------------------------------------
    # Evaluate Every Model
    # ----------------------------------------------------------

    for model_name, prediction in predictions.items():

        metrics = calculate_metrics(
            y_test,
            prediction
        )

        metrics["Model"] = model_name

        metrics["CV Score"] = model_results[
            "cv_scores"
        ][model_name]

        metrics["Training Time"] = model_results[
            "training_times"
        ][model_name]

        metrics["Inference Time"] = model_results[
            "inference_times"
        ][model_name]

        metrics["Latency"] = model_results[
            "latency_times"
        ][model_name]

        metrics["CPU Usage"] = model_results[
            "cpu_usage"
        ][model_name]

        metrics["Memory Usage"] = model_results[
            "memory_usage"
        ][model_name]

        metrics_table.append(metrics)

        print(f"✓ {model_name} Evaluation Completed")

    results = pd.DataFrame(metrics_table)

    results = results[[
        "Model",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "MAPE",
        "CV Score",
        "Training Time",
        "Inference Time",
        "Latency",
        "CPU Usage",
        "Memory Usage"
    ]]

    print("\n")
    print("=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)

    print(results)

    # ----------------------------------------------------------
    # Statistical Test
    # ----------------------------------------------------------

    xgb = predictions["XGBoost"]

    rf = predictions["Random Forest"]

    t_stat, p_value = ttest_rel(
        xgb,
        rf
    )

    print("\nPaired T-Test")

    print(f"T Statistic : {t_stat:.4f}")

    print(f"P Value     : {p_value:.6f}")

    results["P Value"] = p_value
        # ----------------------------------------------------------
    # Best Model Selection
    # ----------------------------------------------------------

    best_model = results.sort_values(
        by="RMSE",
        ascending=True
    ).iloc[0]

    print("\n")
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(best_model)

    # ----------------------------------------------------------
    # Save Evaluation Results
    # ----------------------------------------------------------

    results.to_csv(
        "results/evaluation_results.csv",
        index=False
    )

    print("\n✓ evaluation_results.csv saved")

    # ----------------------------------------------------------
    # Save Prediction Table
    # ----------------------------------------------------------

    prediction_table = pd.DataFrame({

        "Actual":
            y_test.values,

        "XGBoost":
            predictions["XGBoost"],

        "Random Forest":
            predictions["Random Forest"],

        "Decision Tree":
            predictions["Decision Tree"],

        "Gradient Boosting":
            predictions["Gradient Boosting"],

        "Extra Trees":
            predictions["Extra Trees"]

    })

    prediction_table.to_csv(
        "results/predictions.csv",
        index=False
    )

    print("✓ predictions.csv saved")

    # ----------------------------------------------------------
    # Save Statistical Results
    # ----------------------------------------------------------

    statistical_table = pd.DataFrame({

        "Test": [
            "Paired T-Test"
        ],

        "T Statistic": [
            t_stat
        ],

        "P Value": [
            p_value
        ]

    })

    statistical_table.to_csv(
        "results/statistical_results.csv",
        index=False
    )

    print("✓ statistical_results.csv saved")

    # ----------------------------------------------------------
    # Save Feature Importance
    # ----------------------------------------------------------

    feature_importance = model_results["feature_importance"]

    if feature_importance is not None:

        feature_importance.to_csv(
            "results/feature_importance.csv",
            index=False
        )

        print("✓ feature_importance.csv saved")

    # ----------------------------------------------------------
    # Final Summary
    # ----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print(results)

    print("\n")
    print("=" * 70)
    print("GENERATED FILES")
    print("=" * 70)

    print("results/evaluation_results.csv")
    print("results/predictions.csv")
    print("results/statistical_results.csv")

    if feature_importance is not None:
        print("results/feature_importance.csv")

    print("=" * 70)

    # ----------------------------------------------------------
    # Return Results
    # ----------------------------------------------------------

    return {

        "metrics": results,

        "prediction_table": prediction_table,

        "statistical_test": statistical_table,

        "feature_importance": feature_importance,

        "best_model": best_model

    }


# ----------------------------------------------------------
# Standalone Testing
# ----------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("MODEL EVALUATION TEST")
    print("=" * 70)

    print(
        "\nRun main.py or model.py first to generate "
        "model_results dictionary before calling "
        "evaluate_models()."
    )

    print("\nEvaluation module loaded successfully.")