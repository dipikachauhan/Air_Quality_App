AirPredict: Delhi Air Quality Forecasting System
Overview

AirPredict is a deep learning–powered system designed to forecast ground-level air pollutants in Delhi, specifically Ozone (O₃) and Nitrogen Dioxide (NO₂). The system uses a hybrid ensemble of LSTM, GRU, and Transformer models, trained on meteorological and pollution time-series data. A Streamlit web application provides real-time forecasting, health recommendations, and spatial visualization.

Features
Deep Learning Ensemble

72-hour sliding window input for temporal modeling

Combines:

LSTM (captures long-term patterns)

GRU (fast and efficient temporal modeling)

Transformer Encoder (attention-based pattern learning)

Weighted ensemble for final prediction

0.4 × GRU + 0.4 × LSTM + 0.2 × Transformer

Meteorology-Aware Forecasting

Model considers:

Temperature

Specific humidity

Wind components (u, v, w)

Weather model O₃ forecast

Weather model NO₂ forecast

Interactive Streamlit Dashboard

Predict pollutant concentration for chosen site and time

Auto-classify air quality (Good, Moderate, Poor, Severe)

Provide health recommendations

Show 24–72 hour forecast trends

Display real-station scatter map for pollutants

Supported Monitoring Sites

Three stations from Delhi with geographic coordinates:

Site	Latitude	Longitude
Site 1	28.69536	77.18168
Site 2	28.57180	77.07125
Site 3	28.58278	77.23441
Methodology
Data Preprocessing

Removed unused satellite-derived columns

Constructed datetime index and sorted data

Performed linear interpolation and removed missing values

Added cyclic time features (sin/cos of hour and month)

Computed hourly differences (O₃_diff, NO₂_diff)

Generated 72 lag features for both target pollutants

Created 72-step sliding windows

Scaled inputs and outputs using StandardScaler

Split dataset into training (75%) and testing (25%)

Model Training

Each pollutant and each site was trained separately

One scaler per site for inputs

One scaler per pollutant per site for outputs

Three independent models trained per pollutant per site

LSTM

GRU

Transformer

Best weights saved using early stopping and LR scheduler

Predictions combined into a unified ensemble output

Application Workflow
User Inputs

Site (Site 1, Site 2, Site 3)

Date and hour

Temperature

Specific humidity

Wind (u, v, w)

Weather model O₃ forecast

Weather model NO₂ forecast

Prediction Process

Inputs are transformed into a synthetic 72-hour sequence

Sequence is scaled using stored training scalers

Passed to LSTM, GRU, and Transformer models

Model predictions are ensembled

Final pollutant value is obtained after inverse-scaling

Dashboard Output

Predicted O₃ and NO₂ concentrations

Air Quality Index classification

Health recommendations

Time-series graph for 24/48/72-hour forecast

Spatial scatter map showing pollutant levels at all 3 sites

Directory Structure
📁 saved_models/
    ├── site_1/
    ├── site_2/
    └── site_3/
📁 saved_scalers/
    ├── site_1_X_scaler.pkl
    ├── site_1_O3_Y_scaler.pkl
    └── site_2_...
📄 app.py                 # Streamlit application
📄 training_notebook.ipynb # Preprocessing + training
📄 README.md
