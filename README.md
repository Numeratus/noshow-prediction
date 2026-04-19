# Outpatient No-Show Risk Stratifier: ML for Healthcare Logistics

## Project Overview
Missed clinical appointments (no-shows) lead to significant revenue loss, disrupt resource planning, and unnecessarily extend wait times for other patients. This project aims to address this logistical bottleneck by predicting the likelihood of a patient missing an outpatient appointment based on operational and clinical determinants.

## The Solution
This repository contains a complete end-to-end Machine Learning pipeline and an interactive clinical dashboard developed in Python. The process encompasses data ingestion, Exploratory Data Analysis (EDA), and the training of a Random Forest classification algorithm optimized for clinical recall.

Instead of relying on a "black box" approach, the model emphasizes feature interpretability using SHAP, focusing on actionable metrics like appointment lead time and patient demographics.

### Key Features:
* **Predictive Modeling:** A Random Forest classifier heavily weighted to prioritize "Recall" to ensure high-risk no-shows do not slip through the cracks.
* **Interactive Dashboard:** A Streamlit-powered web app (`app.py`) that allows clinic staff to input patient logistics and receive real-time risk assessments.
* **Clinical Interpretability:** Dynamic SHAP waterfall charts explain exactly *why* the model made its prediction for each specific patient.
* **Cost-Benefit Analysis:** A data-driven intervention framework balancing the cost of a missed appointment ($150) versus the cost of a manual staff outreach call ($5).

## Clinical & Business Impact
By analyzing social and operational determinants, the model successfully identifies patient groups with a high risk of no-shows. The dashboard recommends a 3-tier operational workflow:
* **Low Risk (<30%):** Standard automated reminders.
* **Medium Risk (30-70%):** Targeted 2-way SMS confirmation.
* **High Risk (>70%):** Proactive manual phone call or strategic overbooking to mitigate revenue loss.

## Repository Structure

```text
noshow-prediction/
├── app.py                   # Streamlit interactive dashboard & UI
├── models/                  # Serialized machine learning models
│   └── rf_noshow_v1.pkl     # Trained Random Forest classifier
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA, cleaning, and business problem analysis
│   └── 02_model_training.ipynb     # Algorithm training and cost-benefit evaluation
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py     # Modular scripts for data preprocessing
│   └── model_utils.py       # Reusable functions for evaluation & visualization
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
