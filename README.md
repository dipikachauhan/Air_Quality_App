# AirPredict: Delhi Air Quality Forecasting System

## Overview
AirPredict is a deep learning–powered system designed to forecast ground-level air pollutants in Delhi, specifically **Ozone (O₃)** and **Nitrogen Dioxide (NO₂)**. It uses an ensemble of **LSTM**, **GRU**, and **Transformer** models trained on meteorological and pollution time-series data.  
A Streamlit dashboard provides real-time predictions, AQI classification, and spatial mapping.

## Features

### Deep Learning Ensemble
- 72-hour sliding window sequences  
- LSTM, GRU and Transformer encoder models  
- Weighted ensemble output:
- Final Prediction = 0.4 × GRU + 0.4 × LSTM + 0.2 × Transformer

### Meteorological Inputs
- Temperature  
- Specific humidity  
- u, v, w wind components  
- Weather model O₃ forecast  
- Weather model NO₂ forecast  

### Streamlit Dashboard
- Predict pollutant concentrations for any selected date and site  
- Displays AQI category (Good / Moderate / Poor / Severe)  
- Shows health recommendations  
- Provides 24/48/72-hour forecast graphs  
- Displays a spatial scatter map for all three monitoring stations

## Monitoring Sites

| Site   | Latitude  | Longitude |
|--------|-----------|-----------|
| Site 1 | 28.69536  | 77.18168  |
| Site 2 | 28.57180  | 77.07125  |
| Site 3 | 28.58278  | 77.23441  |

## Methodology

### Data Preprocessing
1. Removed unused satellite columns  
2. Constructed datetime index and sorted chronologically  
3. Interpolated missing values  
4. Added cyclic time encodings (hour sin/cos, month sin/cos)  
5. Computed hourly differences for O₃ and NO₂  
6. Generated **72 lag features** per pollutant  
7. Built **72-step sliding windows**  
8. Standardized features and targets  
9. Performed a 75/25 train-test split  

### Model Training
- Individual LSTM, GRU and Transformer models trained per site and pollutant  
- One input scaler per site  
- One output scaler per pollutant per site  
- Early stopping used to prevent overfitting  
- Best model weights saved  

## Application Workflow

### User Inputs
- Site  
- Date and hour  
- Temperature  
- Specific humidity  
- Wind components (u, v, w)  
- Weather-model O₃ forecast  
- Weather-model NO₂ forecast  

### Prediction Pipeline
1. User inputs converted into a **synthetic 72-step sequence**  
2. Inputs scaled using saved scalers  
3. Sequence passed through LSTM, GRU, Transformer  
4. Weighted ensemble applied  
5. Output inverse-transformed to obtain µg/m³ concentrations  

### Dashboard Outputs
- O₃ and NO₂ predicted values  
- AQI level  
- Health advisory  
- Time-series forecast  
- Spatial scatter plot  

## Project Structure
📁 saved_models/
    ├── site_1/
    ├── site_2/
    └── site_3/
📁 saved_scalers/
    ├── site_1_X_scaler.pkl
    ├── site_1_O3_Y_scaler.pkl
    └── site_2_...
📄 app.py                 # Streamlit application
📄 final_project_ann.ipynb # Preprocessing + training
📄 README.md


 
