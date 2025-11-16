# Air Quality Forecasting — O3 & NO2 (Per-site models)

## Project summary
This repository contains code and saved models to forecast **Ozone (O3)** and **Nitrogen Dioxide (NO2)** for three separate monitoring sites. Models are trained per-site (site_1, site_2, site_3) and per-pollutant. The final prediction is produced by an ensemble of three architectures (LSTM, GRU, Transformer).

## Targets
- O3 (Ozone) — target column: `O3_target`
- NO2 (Nitrogen Dioxide) — target column: `NO2_target`

## Datasets
- Three site datasets are used: `site_1_train_data.csv`, `site_2_train_data.csv`, `site_3_train_data.csv`.
- Each file contains meteorological and forecast features, timestamp columns (`year`, `month`, `day`, `hour`), and the target pollutant columns.

## Preprocessing (implemented)
- Construct `datetime` from `year, month, day, hour` and sort chronologically.
- Linear interpolation and forward/backward fill for missing numeric values.
- Drop satellite columns that contain many missing values:
  - `NO2_satellite`, `HCHO_satellite`, `ratio_satellite`
- Create cyclic time features: `hour_sin`, `hour_cos`, `month_sin`, `month_cos`.
- Create short-difference features: `O3_diff`, `NO2_diff`.
- Create lagged features for targets (lags 1..72).
- Standard scaling applied separately to X and y.
- Sequence length (lookback window) used in training/prediction: **72** timesteps.

## Models (per site, per pollutant)
For each site and for each pollutant the code trains:
- LSTM (single-output)
- GRU (single-output)
- Transformer (single-output)

Saved model format: native Keras `.keras`.

Expected saved files (example):
saved_models/site_1/site_1_O3_LSTM.keras
saved_models/site_1/site_1_O3_GRU.keras
saved_models/site_1/site_1_O3_TRANS.keras
saved_models/site_1/site_1_NO2_LSTM.keras
Total saved base models = **18** (3 sites × 2 pollutants × 3 model types).

## Ensemble
- Ensemble prediction used in evaluation and in the app is a weighted average:
  0.4 * GRU + 0.4 * LSTM + 0.2 * Transformer
  - The ensemble itself is not saved as a separate trained model; it is computed at runtime from the three saved base models.

## Outputs
- Saved models in `saved_models/` (structure above)
- Scalers saved with joblib in `models/scalers/` (one X scaler per site and one y-scaler per site/pollutant)
- Prediction outputs (CSV) optionally saved as `predicted_vs_actual.csv` or `predictions.csv` for new inputs
- Numeric evaluation summary (RMSE, MAE, R², RIA) exported to a CSV (e.g., `all_sites_summary.csv`)

## Evaluation metrics
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of determination (R²)
- Refined Index of Agreement (RIA) — calculated in code

## Notes and design choices
- Satellite columns (`NO2_satellite`, `HCHO_satellite`, `ratio_satellite`) were dropped because they had many missing values and degraded model performance in experiments.
- Models are trained independently per-site because sites are not combined due to differing local characteristics.
- Sequence length 72 chosen to capture multi-day temporal patterns; this can be changed in preprocessing/train code.
- All models are saved in `.keras` format to avoid legacy HDF5 warnings.

## Files of interest (high level)
- Training & model pipeline: the main notebook / script you ran (contains preprocessing, sequence creation, model training)
- `saved_models/` — trained model weights (per-site subfolders)
- `models/scalers/` — saved scalers used for X and y transformations
- `predicted_vs_actual.csv` or per-site CSVs — prediction outputs used for evaluation
- Streamlit app files (if present) — for visualization and inference using the saved models

## Academic note
This repository is prepared for academic presentation: code, saved models, scalers, evaluation summaries, and the Streamlit app (if included) together demonstrate the full pipeline from preprocessing to inference for per-site air pollution forecasting.
