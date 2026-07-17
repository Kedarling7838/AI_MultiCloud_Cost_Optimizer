"""
==========================================================
AI-Driven Multi-Cloud Workload Placement
model.py
==========================================================
"""

import time

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor


def train_models(processed):

    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    X_train = processed["X_train"]
    X_test = processed["X_test"]

    y_train = processed["y_train"]
    y_test = processed["y_test"]

    # ----------------------------------------
    # XGBoost Regressor
    # ----------------------------------------

    print("\nTraining XGBoost...")

    start = time.time()

    xgb_model = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=6,

        subsample=0.80,

        colsample_bytree=0.80,

        random_state=42,

        objective="reg:squarederror"

    )

    xgb_model.fit(

        X_train,

        y_train

    )

    xgb_time = time.time() - start

    print("XGBoost Completed")

    # ----------------------------------------
    # Random Forest
    # ----------------------------------------

    print("\nTraining Random Forest...")

    start = time.time()

    rf_model = RandomForestRegressor(

        n_estimators=200,

        max_depth=10,

        random_state=42,

        n_jobs=-1

    )

    rf_model.fit(

        X_train,

        y_train

    )

    rf_time = time.time() - start

    print("Random Forest Completed")

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    print("\nGenerating Predictions...")

    xgb_prediction = xgb_model.predict(

        X_test

    )

    rf_prediction = rf_model.predict(

        X_test

    )

    print("\nTraining Time")

    print("XGBoost :", round(xgb_time, 2), "Seconds")

    print("Random Forest :", round(rf_time, 2), "Seconds")

    print("=" * 60)

    return {

        "xgb_model": xgb_model,

        "rf_model": rf_model,

        "X_test": X_test,

        "y_test": y_test,

        "xgb_prediction": xgb_prediction,

        "rf_prediction": rf_prediction

    }