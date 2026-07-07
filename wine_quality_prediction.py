"""
Wine Quality Prediction
------------------------
Compares multiple regression models on the UCI Wine Quality dataset,
evaluates them with proper metrics (RMSE, MAE, R2), and reports
feature importance to explain what actually drives wine quality.

Dataset: UCI Wine Quality (red wine)
Source: https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed — skipping XGBoost model. Install with: pip install xgboost")


DATA_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
RANDOM_STATE = 42


def load_data():
    """Load the wine quality dataset. Original source: UCI Machine Learning
    Repository (Cortez et al., 2009). Fetched here via a GitHub mirror since
    the UCI site frequently blocks scripted/non-browser requests with a 403."""
    df = pd.read_csv(DATA_URL, sep=";")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df


def explore_data(df):
    """Quick EDA: distribution of target, correlations, missing values."""
    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Quality score distribution ---")
    print(df["quality"].value_counts().sort_index())

    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=150)
    plt.close()
    print("Saved correlation_matrix.png")

    # What correlates most strongly with quality?
    quality_corr = corr["quality"].drop("quality").sort_values(key=abs, ascending=False)
    print("\n--- Features most correlated with quality ---")
    print(quality_corr)


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare them on proper regression metrics."""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE
        )

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

        results.append({
            "Model": name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
            "CV R2 (mean)": round(cv_scores.mean(), 4),
            "CV R2 (std)": round(cv_scores.std(), 4),
        })
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE")
    print("\n--- Model comparison ---")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest model by RMSE: {best_model_name}")

    return fitted_models, results_df, best_model_name


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_, skipping plot.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh")
    plt.title(f"Feature importance — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved feature_importance.png")

    print("\n--- Feature importance (highest first) ---")
    print(importances.sort_values(ascending=False))


def main():
    df = load_data()
    explore_data(df)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Scale features (helps linear regression; harmless for tree models)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    fitted_models, results_df, best_model_name = train_and_compare_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Feature importance from the best tree-based model available
    tree_model_name = "XGBoost" if "XGBoost" in fitted_models else "Random Forest"
    plot_feature_importance(fitted_models[tree_model_name], X.columns, tree_model_name)

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved model_comparison_results.csv")


if __name__ == "__main__":
    main()
