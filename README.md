# Air Quality Forecasting Using Deep Learning (LSTM, GRU, Transformer)

This project predicts ambient concentrations of **NO₂**, **HCHO**, and the **HCHO/NO₂ ratio** using three deep-learning architectures trained on satellite-based and ground-level datasets.  
A **Streamlit-based dashboard** is provided for real-time forecasting and visualization.

---

## 📌 Project Overview

This project implements **three separate models (LSTM, GRU, Transformer)** for each pollutant across **three sites**, resulting in:

- 3 LSTM models  
- 3 GRU models  
- 3 Transformer models  

A simple **ensemble model** is also included, created by averaging predictions from the three architectures.

---

## 📂 Project Structure

project/
│
├── saved_models/
│ ├── site_1_NO2_LSTM.keras
│ ├── site_1_NO2_GRU.keras
│ ├── site_1_NO2_TRANSFORMER.keras
│ ├── ... (similar for site 2 and site 3)
│
├── data/
│ ├── site_1.csv
│ ├── site_2.csv
│ ├── site_3.csv
│
├── app/
│ ├── Home.py
│ ├── Forecast.py
│ ├── Visualize.py
│
├── preprocessing.py
├── model_utils.py
├── requirements.txt
└── README.md

---

## 🚀 Features

### ✔ Deep Learning Models  
Each pollutant for each site is predicted with:
- **LSTM**
- **GRU**
- **Transformer**

### ✔ Ensemble Prediction  
Mean of predictions from the three models.

### ✔ Streamlit Web App
The app includes:
- A **dashboard** for viewing site statistics
- A **forecast page** for real-time predictions
- **Graphs and visualizations**
- Automatic selection of the **best-performing model** per site

---

The app will open in your browser automatically.

---

## 📊 Dataset Description

Each site's dataset contains:
- Meteorological parameters
- Satellite-based variables (NO2, HCHO, Ratio)
- Temporal lags
- Ground pollutant truth values

Satellite variables **NO2_satellite**, **HCHO_satellite**, **ratio_satellite** are dropped as per the project requirement.

---

## 🤖 Model Training (Summary)

Models were trained on:
- Scaled features  
- 30-step sequences  
- Adam optimizer  
- MAE/MSE loss functions  

Each model was saved in the **native Keras format (`.keras`)**.

---


