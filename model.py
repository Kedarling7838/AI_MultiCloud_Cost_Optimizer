"""
==========================================================
AI-Driven Intelligent Decision Support Framework
Model Training using 5-Fold Cross Validation
==========================================================
"""

import os
import time
import platform
import psutil
import xgboost

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor


def train_models(processed):

    print("=" * 70)
    print("MODEL TRAINING USING 5-FOLD CROSS VALIDATION")
    print("=" * 70)

    os.makedirs("results", exist_ok=True)

    X = processed["X"]
    y = processed["y"]

    # ---------------------------------------------------
    # 5-Fold Cross Validation
    # ---------------------------------------------------

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # ---------------------------------------------------
    # Machine Learning Models
    # ---------------------------------------------------

    models = {

        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror"
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    }

    # ---------------------------------------------------
    # Storage Variables
    # ---------------------------------------------------

    trained_models = {}

    predictions = {}

    training_times = {}

    inference_times = {}

    latency_times = {}

    cpu_usage = {}

    memory_usage = {}

    cv_scores = {}

    last_X_test = None

    last_y_test = None

    # ---------------------------------------------------
    # Train Each Model
    # ---------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"Training {model_name}")
        print("=" * 60)

        fold_scores = []

        train_times = []

        infer_times = []

        latency_list = []

        cpu_list = []

        memory_list = []

        fold = 1

        for train_index, test_index in kfold.split(X):

            X_train = X.iloc[train_index]

            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]

            y_test = y.iloc[test_index]

            train_start = time.time()

            model.fit(
                X_train,
                y_train
            )

            training_time = time.time() - train_start

            inference_start = time.time()

            prediction = model.predict(
                X_test
            )

            inference_time = (
                time.time() - inference_start
            )

            latency = inference_time / len(X_test)

            score = model.score(
                X_test,
                y_test
            )

            cpu = psutil.cpu_percent()

            memory = psutil.virtual_memory().percent

            fold_scores.append(score)

            train_times.append(training_time)

            infer_times.append(inference_time)

            latency_list.append(latency)

            cpu_list.append(cpu)

            memory_list.append(memory)

            print(
                f"Fold {fold} | "
                f"R²={score:.4f} | "
                f"Train={training_time:.3f}s | "
                f"Inference={inference_time:.4f}s"
            )

            fold += 1
                    # ---------------------------------------------------
        # Store Average Metrics
        # ---------------------------------------------------

        trained_models[model_name] = model

        predictions[model_name] = prediction

        training_times[model_name] = np.mean(train_times)

        inference_times[model_name] = np.mean(infer_times)

        latency_times[model_name] = np.mean(latency_list)

        cpu_usage[model_name] = np.mean(cpu_list)

        memory_usage[model_name] = np.mean(memory_list)

        cv_scores[model_name] = np.mean(fold_scores)

        last_X_test = X_test

        last_y_test = y_test

    print("\n")
    print("=" * 70)
    print("CROSS VALIDATION COMPLETED")
    print("=" * 70)

    print("\nAverage Cross Validation Scores\n")

    for name, score in cv_scores.items():

        print(f"{name:20s}: {score:.4f}")

    print("\nSystem Information")

    print(f"Operating System : {platform.system()}")

    print(f"Platform         : {platform.platform()}")

    print(f"XGBoost Version  : {xgboost.__version__}")

    # ---------------------------------------------------
    # Best Model Selection
    # ---------------------------------------------------

    best_model_name = max(
        cv_scores,
        key=cv_scores.get
    )

    best_model = trained_models[best_model_name]

    print("\n" + "=" * 70)

    print("BEST MODEL")

    print("=" * 70)

    print(f"Best Model : {best_model_name}")

    print(
        f"Average CV Score : "
        f"{cv_scores[best_model_name]:.4f}"
    )

    # ---------------------------------------------------
    # Feature Importance
    # ---------------------------------------------------

    feature_importance = None

    if hasattr(best_model, "feature_importances_"):

        feature_importance = pd.DataFrame({

            "Feature": X.columns,

            "Importance":
                best_model.feature_importances_

        })

        feature_importance = feature_importance.sort_values(

            by="Importance",

            ascending=False

        )

        feature_importance.to_csv(

            "results/feature_importance.csv",

            index=False

        )

    # ---------------------------------------------------
    # Prediction Table
    # ---------------------------------------------------

    prediction_table = pd.DataFrame({

        "Actual":

            last_y_test.values,

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

    # ---------------------------------------------------
    # Model Summary
    # ---------------------------------------------------

    summary = pd.DataFrame({

        "Model":

            list(models.keys()),

        "CV Score": [

            cv_scores["XGBoost"],

            cv_scores["Random Forest"],

            cv_scores["Decision Tree"],

            cv_scores["Gradient Boosting"],

            cv_scores["Extra Trees"]

        ],

        "Training Time (s)": [

            training_times["XGBoost"],

            training_times["Random Forest"],

            training_times["Decision Tree"],

            training_times["Gradient Boosting"],

            training_times["Extra Trees"]

        ],

        "Inference Time (s)": [

            inference_times["XGBoost"],

            inference_times["Random Forest"],

            inference_times["Decision Tree"],

            inference_times["Gradient Boosting"],

            inference_times["Extra Trees"]

        ],

        "Latency (s/sample)": [

            latency_times["XGBoost"],

            latency_times["Random Forest"],

            latency_times["Decision Tree"],

            latency_times["Gradient Boosting"],

            latency_times["Extra Trees"]

        ],

        "CPU Usage (%)": [

            cpu_usage["XGBoost"],

            cpu_usage["Random Forest"],

            cpu_usage["Decision Tree"],

            cpu_usage["Gradient Boosting"],

            cpu_usage["Extra Trees"]

        ],

        "Memory Usage (%)": [

            memory_usage["XGBoost"],

            memory_usage["Random Forest"],

            memory_usage["Decision Tree"],

            memory_usage["Gradient Boosting"],

            memory_usage["Extra Trees"]

        ]

    })

    summary.to_csv(

        "results/model_results.csv",

        index=False

    )
        # ---------------------------------------------------
    # Display Model Summary
    # ---------------------------------------------------

    print("\n")
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)

    print(summary)

    print("\n")
    print("=" * 70)
    print("MODEL TRAINING COMPLETED")
    print("=" * 70)

    # ---------------------------------------------------
    # Return Results
    # ---------------------------------------------------

    return {

        "trained_models": trained_models,

        "best_model": best_model,

        "best_model_name": best_model_name,

        "summary": summary,

        "prediction_table": prediction_table,

        "feature_importance": feature_importance,

        "cv_scores": cv_scores,

        "training_times": training_times,

        "inference_times": inference_times,

        "latency_times": latency_times,

        "cpu_usage": cpu_usage,

        "memory_usage": memory_usage,

        "X_test": last_X_test,

        "y_test": last_y_test,

        "xgb_prediction": predictions["XGBoost"],

        "rf_prediction": predictions["Random Forest"],

        "dt_prediction": predictions["Decision Tree"],

        "gb_prediction": predictions["Gradient Boosting"],

        "et_prediction": predictions["Extra Trees"]

    }


# ---------------------------------------------------
# Standalone Testing
# ---------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("MODEL TESTING")
    print("=" * 70)

    from preprocessing import preprocess_data
    from feature_engineering import feature_engineering

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
        batch_task,
        machine_meta,
        cloud_pricing,
        latency
    )

    features = feature_engineering(
        processed_dataset
    )

    results = train_models(
        features
    )

    print("\nBest Model")

    print(results["best_model_name"])

    print("\nCross Validation Scores")

    for model, score in results["cv_scores"].items():

        print(f"{model:20s} : {score:.4f}")

    print("\nTraining completed successfully.")