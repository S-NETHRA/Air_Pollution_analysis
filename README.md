# Air Quality Monitoring & Pollution Dashboard

An interactive Streamlit dashboard for exploring air quality by city. Existing
features include PM2.5/PM10 trend charts, city comparisons, temperature
visualization, and health alerts. The dashboard also includes a next-day PM2.5
forecast trained from the project's historical CSV data.

## Dataset and forecasting scope

The supplied raw dataset contains daily records for 10 cities from
2024-01-01 through 2024-04-09. Columns include `Date`, `City`, `PM2.5`,
`PM10`, `CO`, `NO2`, `Ozone`, `Temperature`, and `AQI_Level`. PM2.5 and some
other pollutant fields are missing in the source; the training script preserves
those missing values and does not create synthetic measurements. The weather
field available in this dataset is `Temperature`.

The prediction target is the next calendar day's PM2.5. Historical inputs are
the current and prior PM2.5 readings, a trailing three-day PM2.5 average, and
available current-day pollutant/temperature measurements. Missing model inputs
are imputed inside a preprocessing pipeline fitted on training data only.
There are no tomorrow-weather forecasts or external emissions inputs, and the
short history limits long-term seasonal forecasting.

## Setup and run

From the repository root, install dependencies and generate the model artifacts
once:

```powershell
cd air_quality_dashboard_project
python -m pip install -r requirements.txt
python scripts/train_pm25_model.py
```

Start the dashboard:

```powershell
streamlit run dashboard/dashboard.py
```

Training is a separate command and is not repeated by Streamlit. It writes
`models/pm25_forecast.joblib` (the selected model and preprocessing pipeline),
`models/pm25_metrics.json` (validation selection and held-out test metrics),
and `models/pm25_test_predictions.csv` (historical test predictions). Rerun
the training command after changing the source dataset or training code.

## How to interpret the forecast

The app shows a next-day estimate for the selected city based on that city's
latest dated source record. It is explicitly a model estimate, not a
real-time measurement. The historical actual-versus-predicted chart and MAE,
RMSE, and R² values are from chronological held-out test dates. Linear
Regression and Random Forest are compared against a baseline that predicts
tomorrow using today's PM2.5; the app selects between the two learned models
using validation RMSE. The baseline is a benchmark, not a trained model.
