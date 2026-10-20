import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from pathlib import Path

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# Load Cleaned Data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(PROJECT_ROOT / "data" / "clean_air_quality_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Title
st.title("🌍 Environmental Air Quality & Pollution Dashboard")

# Sidebar Filters
st.sidebar.header("Filter Data")
selected_city = st.sidebar.selectbox("Select a City", df["City"].unique())
filtered_df = df[df["City"] == selected_city].copy()

# ---- 1️⃣ AQI Trends Over Time ----
st.subheader(f"📈 AQI Trends Over Time for {selected_city}")
fig1 = px.line(filtered_df, x="Date", y=["PM2.5", "PM10"], 
               labels={"value": "Concentration (µg/m³)", "variable": "Pollutant"},
               title=f"PM2.5 & PM10 Levels in {selected_city}")
st.plotly_chart(fig1, use_container_width=True)

# ---- 2️⃣ City-Wise Pollution Comparison ----
st.subheader("🏙️ City-wise Pollution Comparison")
city_avg = df.groupby("City")[["PM2.5", "PM10"]].mean().reset_index()
fig2 = px.bar(city_avg, x="City", y="PM2.5", title="Average PM2.5 Levels by City", color="PM2.5")
st.plotly_chart(fig2, use_container_width=True)

# ---- 3️⃣ Correlation Between Temperature and Air Quality ----
st.subheader(f"🌡️ Correlation Between Temperature & Air Quality in {selected_city}")
fig3, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x="Temperature", y="PM2.5", hue="PM2.5", palette="coolwarm", ax=ax)
ax.set_title(f"Temperature vs PM2.5 Levels in {selected_city}")
st.pyplot(fig3)

# ---- 4️⃣ Public Health Alerts ----
st.subheader("🚦 Public Health Alert System")

def aqi_alert(value):
    if value <= 50:
        return "✅ Good (Green)"
    elif value <= 100:
        return "🟡 Moderate (Yellow)"
    elif value <= 150:
        return "🟠 Unhealthy for Sensitive Groups (Orange)"
    elif value <= 200:
        return "🔴 Unhealthy (Red)"
    else:
        return "🟣 Hazardous (Purple)"

filtered_df["AQI_Alert"] = filtered_df["PM2.5"].apply(aqi_alert)
num_rows = st.slider("Select number of rows to display:", min_value=5, max_value=100, value=10, step=5)
st.write(filtered_df[["Date", "PM2.5", "AQI_Alert"]].tail(num_rows))

# ---- Run the dashboard ----
# ---- 5️⃣ City-Wide Public Health Alert System ----
st.subheader("🏙️ Overall Air Quality Status by City")

# Function to classify cities based on average PM2.5 levels
def city_aqi_alert(avg_pm25):
    if avg_pm25 <= 50:
        return "✅ Good (Green)"
    elif avg_pm25 <= 100:
        return "🟡 Moderate (Yellow)"
    elif avg_pm25 <= 150:
        return "🟠 Unhealthy for Sensitive Groups (Orange)"
    elif avg_pm25 <= 200:
        return "🔴 Unhealthy (Red)"
    else:
        return "🟣 Hazardous (Purple)"

# Calculate city-wise average PM2.5
city_avg_pm25 = df.groupby("City")["PM2.5"].mean().reset_index()
city_avg_pm25["AQI_Alert"] = city_avg_pm25["PM2.5"].apply(city_aqi_alert)

# Display city-wise air quality status
st.dataframe(city_avg_pm25.rename(columns={"PM2.5": "Avg PM2.5 Level", "AQI_Alert": "Health Alert"}), 
             height=400, use_container_width=True)

# ---- 6️⃣ PM2.5 Prediction ----
st.subheader("🤖 PM2.5 Prediction")
model_path = PROJECT_ROOT / "models" / "pm25_forecast.joblib"
metrics_path = PROJECT_ROOT / "models" / "pm25_metrics.json"
predictions_path = PROJECT_ROOT / "models" / "pm25_test_predictions.csv"
raw_data_path = PROJECT_ROOT / "data" / "air_quality_dataset.csv"

if not all(path.exists() for path in (model_path, metrics_path, predictions_path)):
    st.info(
        "Forecast artifacts are not available yet. From the project directory, "
        "run `python scripts/train_pm25_model.py` once to train and save the model."
    )
else:
    with model_path.open("rb") as model_file:
        model_bundle = joblib.load(model_file)
    with metrics_path.open(encoding="utf-8") as metrics_file:
        model_metrics = json.load(metrics_file)
    test_predictions = pd.read_csv(predictions_path, parse_dates=["forecast_date"])

    if raw_data_path.exists():
        forecast_history = pd.read_csv(raw_data_path)
        forecast_history["Date"] = pd.to_datetime(
            forecast_history["Date"], errors="coerce"
        )
        forecast_cities = sorted(forecast_history["City"].dropna().unique())
        if forecast_cities:
            prediction_city = st.selectbox(
                "City for next-day forecast",
                forecast_cities,
                key="pm25_prediction_city",
            )
            city_history = forecast_history[
                (forecast_history["City"] == prediction_city)
                & forecast_history["Date"].notna()
            ].copy()

            if not city_history.empty:
                history_end = city_history["Date"].max()
                latest_values = city_history[city_history["Date"] == history_end]
                numeric_columns = [
                    column
                    for column in (
                        "PM2.5",
                        "PM10",
                        "CO",
                        "NO2",
                        "Ozone",
                        "Temperature",
                    )
                    if column in city_history.columns
                ]
                daily_values = (
                    city_history.sort_values("Date")
                    .groupby("Date", as_index=False)[numeric_columns]
                    .mean()
                )
                daily_values = daily_values.set_index("Date").asfreq("D")
                for column in numeric_columns:
                    daily_values[column] = pd.to_numeric(
                        daily_values[column], errors="coerce"
                    )
                daily_values["pm25_today"] = daily_values["PM2.5"]
                daily_values["pm25_previous_day"] = daily_values["PM2.5"].shift(1)
                daily_values["pm25_rolling_3day_mean"] = (
                    daily_values["PM2.5"].rolling(window=3, min_periods=1).mean()
                )
                daily_values["City"] = prediction_city
                feature_row = daily_values.loc[[history_end]].copy()
                feature_row = feature_row.rename(
                    columns={
                        column: f"{column}_today"
                        for column in ("PM10", "CO", "NO2", "Ozone", "Temperature")
                        if column in numeric_columns
                    }
                )
                feature_row = feature_row.reindex(
                    columns=model_bundle["feature_columns"]
                )
                forecast_value = float(
                    model_bundle["pipeline"].predict(feature_row)[0]
                )
                forecast_date = history_end + pd.Timedelta(days=1)

                left, right = st.columns(2)
                left.metric(
                    "Next-day predicted PM2.5",
                    f"{forecast_value:.2f} µg/m³",
                )
                right.metric("Forecast date", forecast_date.strftime("%Y-%m-%d"))
                st.caption(
                    f"Forecast for {prediction_city} based on the latest dataset "
                    f"record ({history_end:%Y-%m-%d}); this is a model estimate, "
                    "not a real-time measurement."
                )
            else:
                st.warning("No dated history is available for this city.")
    else:
        st.warning("The raw dataset needed to create a forecast was not found.")

    selected_model_name = model_metrics["selected_model"]
    st.write(f"Model selected using validation data: **{selected_model_name}**")
    st.caption(
        "Historical test results below are predictions for held-out dates, "
        "not future or real-time measurements."
    )

    comparison = model_metrics["test_metrics"]
    comparison_rows = [
        {
            "Model": name,
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "R²": values["R2"],
            "Test observations": values["n"],
        }
        for name, values in comparison.items()
    ]
    st.dataframe(
        pd.DataFrame(comparison_rows).set_index("Model"),
        use_container_width=True,
    )

    city_test = test_predictions[
        test_predictions["City"] == (
            prediction_city if "prediction_city" in locals() else ""
        )
    ].sort_values("forecast_date")
    if not city_test.empty:
        chart_data = city_test.melt(
            id_vars=["forecast_date"],
            value_vars=["actual_pm25", "selected_model_prediction"],
            var_name="Series",
            value_name="PM2.5 (µg/m³)",
        )
        chart_data["Series"] = chart_data["Series"].map(
            {
                "actual_pm25": "Actual PM2.5",
                "selected_model_prediction": "Predicted PM2.5",
            }
        )
        fig4 = px.line(
            chart_data,
            x="forecast_date",
            y="PM2.5 (µg/m³)",
            color="Series",
            markers=True,
            title=f"Held-out historical PM2.5: actual vs predicted in {prediction_city}",
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("There are no held-out historical test predictions for this city.")

    st.markdown(
        """
        **How this forecast works:** The model uses the selected city's current
        and recent PM2.5 history, current-day PM10/CO/NO2/Ozone when available,
        and Temperature when present. Missing inputs are imputed using values
        learned from the training period only. Linear Regression and Random
        Forest are compared with a persistence baseline (tomorrow equals
        today's PM2.5); the machine-learning model is selected by validation
        RMSE, and all displayed scores come from later held-out test dates.

        **Limitations:** The dataset covers only 100 days in 2024 and has
        substantial missing PM2.5 values. Weather forecasts for tomorrow,
        emissions, and other external conditions are not available, so this is
        a short-horizon estimate based on observed history rather than a
        real-time or operational air-quality forecast.
        """
    )
