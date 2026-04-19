import pandas as pd
import numpy as np
import logging
import os

# Configure logging for production-level monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Ingests the raw clinical dataset.

    Args:
        file_path (str): The relative or absolute path to the raw CSV file.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully ingested data from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Critical Error: File not found at {file_path}.")
        raise

def clean_clinical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the clinical appointment dataset and engineers operational features.

    Args:
        df (pd.DataFrame): Raw pandas dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe ready for exploratory analysis or modeling.
    """
    df_cleaned = df.copy()

    # 1. Standardize column nomenclature
    rename_dict = {
        'Hipertension': 'Hypertension',
        'Handcap': 'Handicap',
        'No-show': 'No_Show'
    }
    df_cleaned.rename(columns=rename_dict, inplace=True)
    logging.info("Standardized clinical column nomenclature.")

    # 2. Parse temporal features
    df_cleaned['ScheduledDay'] = pd.to_datetime(df_cleaned['ScheduledDay']).dt.date
    df_cleaned['AppointmentDay'] = pd.to_datetime(df_cleaned['AppointmentDay']).dt.date

    # 3. Engineer 'Lead Time' (Logistical Feature)
    df_cleaned['LeadDays'] = (df_cleaned['AppointmentDay'] - df_cleaned['ScheduledDay']).dt.days

    # 4. Filter anomalous records (e.g., scheduling after appointment, negative age)
    initial_shape = df_cleaned.shape[0]
    df_cleaned = df_cleaned[(df_cleaned['LeadDays'] >= 0) & (df_cleaned['Age'] >= 0)]
    records_dropped = initial_shape - df_cleaned.shape[0]
    logging.info(f"Filtered anomalous operational records. Dropped {records_dropped} rows.")

    # 5. Binarize Target Variable for ML
    if df_cleaned['No_Show'].dtype == 'O':
        df_cleaned['No_Show'] = df_cleaned['No_Show'].map({'Yes': 1, 'No': 0})
        logging.info("Binarized the 'No_Show' target variable (1 = No-Show, 0 = Attended).")

    return df_cleaned

def preprocess_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the cleaned dataset for machine learning by dropping operational IDs
    and encoding categorical variables.
    """
    df_model = df.copy()

    # Drop non-predictive operational IDs and temporal data
    columns_to_drop = ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood']
    df_model.drop(columns=[col for col in columns_to_drop if col in df_model.columns], inplace=True)

    # Encode categorical variables (Gender)
    if 'Gender' in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=['Gender'], drop_first=True)
        logging.info("Applied one-hot encoding to categorical features.")

    logging.info(f"Data successfully preprocessed for modeling. Final shape: {df_model.shape}")
    return df_model

if __name__ == "__main__":
    # This block allows the script to be run directly from the terminal
    # Example: python src/data_cleaning.py

    RAW_DATA_PATH = 'data/raw/KaggleV2-May-2016.csv'
    PROCESSED_DATA_PATH = 'data/processed/model_ready_appointments.csv'

    logging.info("Starting data pipeline execution...")

    # Execute Pipeline
    raw_df = load_data(RAW_DATA_PATH)
    cleaned_df = clean_clinical_data(raw_df)
    model_ready_df = preprocess_for_modeling(cleaned_df)

    # Save the processed data
    # Ensure the output directory exists
    os.makedirs('data/processed', exist_ok=True)
    model_ready_df.to_csv(PROCESSED_DATA_PATH, index=False)

    logging.info(f"Pipeline complete. Processed data saved to {PROCESSED_DATA_PATH}")
