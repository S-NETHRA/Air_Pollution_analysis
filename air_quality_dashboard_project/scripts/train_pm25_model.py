"""Train and evaluate a next-day PM2.5 model without random time-series splits."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "air_quality_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "pm25_forecast.joblib"
METRICS_PATH = MODEL_DIR / "pm25_metrics.json"
PREDICTIONS_PATH = MODEL_DIR / "pm25_test_predictions.csv"

POLLUTION_AND_WEATHER_COLUMNS = ["PM2.5", "PM10", "CO", "NO2", "Ozone", "Temperature"]
HISTORY_FEATURES = [
    "pm25_today",
    "pm25_previous_day",
    "pm25_rolling_3day_mean",
]
TARGET_COLUMN = "target_pm25"
TARGET_DATE_COLUMN = "forecast_date"


def load_daily_records(path: Path) -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(path)
    required = {"Date", "City", "PM2.5"}
    missing_required = required.difference(raw.columns)
    if missing_required:
        raise ValueError(
            "The source dataset is missing required columns: "
            + ", ".join(sorted(missing_required))
        )

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "City"]).copy()
    available_inputs = [
        column
        for column in POLLUTION_AND_WEATHER_COLUMNS
        if column in raw.columns
    ]
    for column in available_inputs:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    daily = (
        raw.groupby(["City", "Date"], as_index=False)[available_inputs]
        .mean()
        .sort_values(["City", "Date"])
    )
    if daily.empty:
        raise ValueError("No valid city/date records were found in the source dataset.")

    # Reindex each city to calendar days; absent measurements remain missing.
    city_records: list[pd.DataFrame] = []
    overall_start = daily["Date"].min().normalize()
    overall_end = daily["Date"].max().normalize()
    for city, city_frame in daily.groupby("City", sort=True):
        city_frame = city_frame.set_index("Date").reindex(
            pd.date_range(overall_start, overall_end, freq="D")
        )
        city_frame.index.name = "Date"
        city_frame["City"] = city
        city_records.append(city_frame.reset_index())

    return pd.concat(city_records, ignore_index=True), available_inputs


def make_supervised_records(
    daily: pd.DataFrame,
    available_inputs: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    feature_records: list[pd.DataFrame] = []
    optional_current_features = [
        f"{column}_today"
        for column in ("PM10", "CO", "NO2", "Ozone", "Temperature")
        if column in available_inputs
    ]
    feature_columns = ["City", *HISTORY_FEATURES, *optional_current_features]

    for _, city_frame in daily.groupby("City", sort=False):
        city_frame = city_frame.sort_values("Date").copy()
        pm25 = city_frame["PM2.5"]
        city_frame["pm25_today"] = pm25
        city_frame["pm25_previous_day"] = pm25.shift(1)
        city_frame["pm25_rolling_3day_mean"] = pm25.rolling(
            window=3, min_periods=1
        ).mean()
        for column in ("PM10", "CO", "NO2", "Ozone", "Temperature"):
            if column in available_inputs:
                city_frame[f"{column}_today"] = city_frame[column]

        city_frame[TARGET_COLUMN] = pm25.shift(-1)
        city_frame[TARGET_DATE_COLUMN] = city_frame["Date"] + pd.Timedelta(days=1)
        feature_records.append(city_frame)

    records = pd.concat(feature_records, ignore_index=True)
    return records, feature_columns


def make_pipeline(regressor: object, numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_features,
            ),
            (
                "city",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["City"],
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def calculate_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | int]:
    actual_values = actual.to_numpy(dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual_values, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual_values, predicted))),
        "R2": float(r2_score(actual_values, predicted)),
        "n": int(len(actual_values)),
    }


def main() -> None:
    daily, available_inputs = load_daily_records(DATA_PATH)
    records, feature_columns = make_supervised_records(daily, available_inputs)
    calendar_dates = sorted(records[TARGET_DATE_COLUMN].dropna().unique())
    if len(calendar_dates) < 10:
        raise ValueError("At least 10 daily forecast dates are required for a time split.")

    train_cutoff = pd.Timestamp(calendar_dates[int(len(calendar_dates) * 0.70)])
    validation_cutoff = pd.Timestamp(calendar_dates[int(len(calendar_dates) * 0.85)])
    labeled = records.dropna(subset=[TARGET_COLUMN]).copy()
    train = labeled[labeled[TARGET_DATE_COLUMN] <= train_cutoff]
    validation = labeled[
        (labeled[TARGET_DATE_COLUMN] > train_cutoff)
        & (labeled[TARGET_DATE_COLUMN] <= validation_cutoff)
    ]
    test = labeled[labeled[TARGET_DATE_COLUMN] > validation_cutoff]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "The chronological train/validation/test split produced an empty set; "
            "more dated PM2.5 observations are needed."
        )

    numeric_features = [column for column in feature_columns if column != "City"]
    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    validation_scores: dict[str, float] = {}
    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator, numeric_features)
        pipeline.fit(train[feature_columns], train[TARGET_COLUMN])
        validation_prediction = pipeline.predict(validation[feature_columns])
        validation_scores[name] = float(
            np.sqrt(
                mean_squared_error(
                    validation[TARGET_COLUMN], validation_prediction
                )
            )
        )

    selected_model_name = min(validation_scores, key=validation_scores.get)
    selected_estimator = estimators[selected_model_name]

    # Compare all methods on identical held-out test dates where persistence applies.
    test = test[test["pm25_today"].notna()].copy()
    if len(test) < 2:
        raise ValueError(
            "Fewer than two test examples have both observed current-day PM2.5 "
            "and next-day PM2.5, so the baseline and test metrics cannot be compared."
        )

    train_validation = labeled[labeled[TARGET_DATE_COLUMN] <= validation_cutoff]
    comparison_predictions: dict[str, np.ndarray] = {}
    for name, estimator in estimators.items():
        pipeline = make_pipeline(estimator, numeric_features)
        pipeline.fit(
            train_validation[feature_columns],
            train_validation[TARGET_COLUMN],
        )
        comparison_predictions[name] = pipeline.predict(test[feature_columns])
    comparison_predictions["Today's PM2.5 baseline"] = test["pm25_today"].to_numpy(
        dtype=float
    )
    test_metrics = {
        name: calculate_metrics(test[TARGET_COLUMN], predictions)
        for name, predictions in comparison_predictions.items()
    }

    output_predictions = test[
        ["City", "Date", TARGET_DATE_COLUMN, TARGET_COLUMN]
    ].rename(
        columns={
            "Date": "feature_date",
            TARGET_COLUMN: "actual_pm25",
        }
    )
    output_predictions["linear_regression_prediction"] = comparison_predictions[
        "Linear Regression"
    ]
    output_predictions["random_forest_prediction"] = comparison_predictions[
        "Random Forest"
    ]
    output_predictions["baseline_prediction"] = comparison_predictions[
        "Today's PM2.5 baseline"
    ]
    selected_prediction_column = (
        "linear_regression_prediction"
        if selected_model_name == "Linear Regression"
        else "random_forest_prediction"
    )
    output_predictions["selected_model_prediction"] = output_predictions[
        selected_prediction_column
    ]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_predictions.to_csv(PREDICTIONS_PATH, index=False)

    # The deployment pipeline is then refitted on all labeled history.
    final_pipeline = make_pipeline(selected_estimator, numeric_features)
    final_pipeline.fit(labeled[feature_columns], labeled[TARGET_COLUMN])
    bundle = {
        "pipeline": final_pipeline,
        "feature_columns": feature_columns,
        "selected_model": selected_model_name,
        "available_input_columns": available_inputs,
        "history_end_date": daily["Date"].max().strftime("%Y-%m-%d"),
        "trained_through_forecast_date": labeled[TARGET_DATE_COLUMN]
        .max()
        .strftime("%Y-%m-%d"),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    metrics = {
        "selected_model": selected_model_name,
        "selection_method": "lowest validation RMSE among Linear Regression and Random Forest",
        "validation_rmse": validation_scores,
        "split_dates": {
            "train_through": train_cutoff.strftime("%Y-%m-%d"),
            "validation_through": validation_cutoff.strftime("%Y-%m-%d"),
            "test_after": validation_cutoff.strftime("%Y-%m-%d"),
        },
        "split_labeled_rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test_comparison": int(len(test)),
        },
        "source_columns": list(pd.read_csv(DATA_PATH, nrows=0).columns),
        "available_weather_variables": [
            column for column in ("Temperature",) if column in available_inputs
        ],
        "test_metrics": test_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Dataset: {DATA_PATH}")
    print(f"Daily dates per city: {daily['Date'].min():%Y-%m-%d} to {daily['Date'].max():%Y-%m-%d}")
    print(f"Cities: {daily['City'].nunique()}")
    print(f"Input columns: {', '.join(available_inputs)}")
    print(f"Selected model: {selected_model_name}")
    print("Held-out test metrics (same examples for all methods):")
    for name, values in test_metrics.items():
        print(
            f"  {name}: MAE={values['MAE']:.3f}, "
            f"RMSE={values['RMSE']:.3f}, R2={values['R2']:.3f}, n={values['n']}"
        )
    print(f"Saved model pipeline: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved historical test predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
