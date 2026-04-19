import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Evaluates the classification model focusing on clinical operational metrics,
    specifically Recall for the minority class (No-Shows) and overall ROC-AUC.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted binary labels.
        y_prob (np.ndarray): Predicted probability scores for the positive class.

    Returns:
        dict: A dictionary containing key performance metrics.
    """
    logging.info("Evaluating model performance...")

    # Calculate ROC-AUC
    auc_score = roc_auc_score(y_true, y_prob)

    # Print Classification Report
    print("\n--- Operational Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Attended (0)', 'No-Show (1)']))
    print(f"ROC-AUC Score: {auc_score:.4f}\n")

    return {'roc_auc': auc_score}

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Generates and displays a styled confusion matrix for operational outcomes.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Attended', 'No-Show'])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.title('Confusion Matrix: Operational Outcomes', fontsize=14, fontweight='bold')
    plt.grid(False) # Turn off seaborn grid lines for this specific plot
    plt.show()

def plot_feature_importance(importances: np.ndarray, feature_names: pd.Index, top_n: int = 10) -> None:
    """
    Plots the top N most important features driving the model's predictions.

    Args:
        importances (np.ndarray): Feature importance scores from the trained model.
        feature_names (pd.Index): The names of the features.
        top_n (int): Number of top features to display. Default is 10.
    """
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df, hue='Feature', palette='viridis', legend=False)
    plt.title(f'Top {top_n} Primary Drivers of Clinic No-Shows', fontsize=14, fontweight='bold')
    plt.xlabel('Relative Importance (Gini)', fontsize=12)
    plt.ylabel('Clinical/Operational Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

    logging.info("Feature importance plot generated successfully.")
