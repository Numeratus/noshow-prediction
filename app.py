import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Try to import SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="No-Show Risk Stratifier",
    layout="centered"
)

# Constants
MODEL_PATH = Path(__file__).parent / "models" / "rf_noshow_v1.pkl"

FEATURE_ORDER = [
    "Age", "Scholarship", "Hypertension", "Diabetes",
    "Alcoholism", "Handicap", "SMS_received", "LeadDays", "Gender_M"
]

# Helper: load or train a model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """
    Loads the saved Random Forest from disk.
    If no saved model exists, trains a fresh one from the raw Kaggle CSV
    (expects data/raw/KaggleV2-May-2016.csv relative to this file).
    """
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # Fallback: train from scratch
    st.warning("No saved model found – training from raw data. This may take a moment.")

    RAW_DATA = Path(__file__).parent / "data" / "raw" / "KaggleV2-May-2016.csv"
    if not RAW_DATA.exists():
        st.error(
            f"Could not find raw data at `{RAW_DATA}`. "
            "Please either place the CSV there or put a trained model at "
            f"`{MODEL_PATH}`."
        )
        st.stop()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(RAW_DATA)
    df.rename(columns={"Hipertension": "Hypertension", "Handcap": "Handicap", "No-show": "No_Show"}, inplace=True)
    df["ScheduledDay"]  = pd.to_datetime(df["ScheduledDay"]).dt.date
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"]).dt.date
    df["LeadDays"] = (df["AppointmentDay"] - df["ScheduledDay"]).apply(lambda x: x.days)
    df = df[(df["LeadDays"] >= 0) & (df["Age"] >= 0)]
    df["No_Show"] = df["No_Show"].map({"Yes": 1, "No": 0})
    df = pd.get_dummies(df.drop(columns=["PatientId","AppointmentID","ScheduledDay","AppointmentDay","Neighbourhood"]),
                        columns=["Gender"], drop_first=True)

    X = df[FEATURE_ORDER]
    y = df["No_Show"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def risk_tier(prob: float) -> tuple[str, str]:
    """Returns (label, hex_colour) for a given probability."""
    if prob < 0.30:
        return "🟢 Low Risk", "#2ca02c"
    elif prob < 0.70:
        return "🟡 Medium Risk", "#ff7f0e"
    else:
        return "🔴 High Risk", "#d62728"


def build_input_row(age, gender, lead_days, sms, hypertension,
                    diabetes, alcoholism, handicap, scholarship) -> pd.DataFrame:
    return pd.DataFrame([{
        "Age":          age,
        "Scholarship":  int(scholarship),
        "Hypertension": int(hypertension),
        "Diabetes":     int(diabetes),
        "Alcoholism":   int(alcoholism),
        "Handicap":     int(handicap),
        "SMS_received": int(sms),
        "LeadDays":     lead_days,
        "Gender_M":     1 if gender == "Male" else 0,
    }])[FEATURE_ORDER]


# UI
st.title("Outpatient No-Show Risk Stratifier")
st.caption("Enter appointment details below to calculate the patient's no-show probability.")

model = load_model()

with st.form("patient_form"):
    st.subheader("Patient Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age     = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
        gender  = st.selectbox("Gender", ["Female", "Male"])
    with col2:
        scholarship  = st.checkbox("Welfare/Scholarship recipient")
        hypertension = st.checkbox("Hypertension")
        diabetes     = st.checkbox("Diabetes")
        alcoholism   = st.checkbox("Alcoholism")
        handicap     = st.checkbox("Handicap")

    st.divider()
    st.subheader("Appointment Logistics")
    col3, col4 = st.columns(2)
    with col3:
        lead_days = st.number_input("Lead Time (days between booking & appointment)",
                                    min_value=0, max_value=365, value=7)
    with col4:
        sms = st.checkbox("SMS reminder already sent?")

    submitted = st.form_submit_button("Calculate Risk", use_container_width=True, type="primary")


if submitted:
    patient_df = build_input_row(age, gender, lead_days, sms,
                                 hypertension, diabetes, alcoholism, handicap, scholarship)
    prob       = model.predict_proba(patient_df)[0, 1]
    label, colour = risk_tier(prob)

    st.divider()
    st.subheader("Risk Assessment")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("No-Show Probability", f"{prob:.1%}")
        st.markdown(f"<h3 style='color:{colour}'>{label}</h3>", unsafe_allow_html=True)
    with col_b:
        st.progress(float(prob))
        st.caption("0 % ──────────────────── 100 %")

    # Recommended intervention
    st.divider()
    st.subheader("Recommended Intervention")
    if prob < 0.30:
        st.success("**Standard protocol** – automated reminder only. No manual follow-up required.")
    elif prob < 0.70:
        st.warning("**Targeted SMS** – send a two-way confirmation message and await patient reply.")
    else:
        st.error("**Proactive outreach** – manual phone call by clinic staff recommended. "
                 "Consider strategic overbooking for this time slot to mitigate revenue impact.")

    # SHAP explanation (if library is available)
    if SHAP_AVAILABLE:
        with st.expander("Why this prediction? (SHAP Feature Contributions)"):
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(patient_df)

            # shap_values is list [class0, class1] for classifiers
            if isinstance(shap_values, list):
                sv_1d = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                sv_1d = shap_values[0, :, 1]
            else:
                sv_1d = shap_values[0]

            shap_df = pd.DataFrame({
                "Feature":      FEATURE_ORDER,
                "Contribution": sv_1d,
                "Value":        patient_df.iloc[0].values
            }).sort_values("Contribution", key=abs, ascending=False)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            colours = ["#d62728" if c > 0 else "#2ca02c" for c in shap_df["Contribution"]]
            ax.barh(shap_df["Feature"], shap_df["Contribution"], color=colours)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP Value (impact on no-show probability)")
            ax.set_title("Feature Contributions for This Patient")
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("🔴 Red = pushes toward no-show  |  🟢 Green = pushes toward attendance")

    # Raw details
    with st.expander("📋 Input summary"):
        st.dataframe(patient_df.T.rename(columns={0: "Value"}))
