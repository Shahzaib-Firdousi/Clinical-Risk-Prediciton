import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="30-Day Readmission Risk Demo", layout="wide")

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_pipeline.joblib"
DEMO_PATH = PROJECT_ROOT / "data" / "processed" / "demo_patients.csv"


# ---------- Loaders / Caching ----------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_demo_data():
    if not DEMO_PATH.exists():
        raise FileNotFoundError(f"Demo data not found: {DEMO_PATH}")
    return pd.read_csv(DEMO_PATH)


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def threshold_for_alert_budget(scores: np.ndarray, budget_frac: float) -> float:
    """Return score threshold such that approximately top budget_frac are flagged."""
    scores = np.asarray(scores)
    k = max(1, int(budget_frac * len(scores)))
    return float(np.sort(scores)[-k])


def shap_vector_for_positive_class(shap_output) -> np.ndarray:
    """
    Robustly extract a 1D SHAP vector for the positive class.
    Handles:
      - ndarray (n_samples, n_features) or (n_features,)
      - list of arrays (per class)
      - shap.Explanation (with .values)
    """
    if isinstance(shap_output, list):
        sv = shap_output[1] if len(shap_output) > 1 else shap_output[0]
        return np.asarray(sv).reshape(-1)

    values = shap_output.values if hasattr(shap_output, "values") else shap_output
    return np.asarray(values).reshape(-1)


def make_patient_options(scores: np.ndarray):
    return [f"Row {i} | risk={scores[i]:.3f}" for i in range(len(scores))]


# ---------- App ----------
st.title("30-Day Readmission Risk Prediction (Demo)")
st.caption("Select an example patient, view predicted risk, and see top contributing factors (SHAP).")

clf = load_model()
preprocess = clf.named_steps["prep"]
model = clf.named_steps["model"]

demo_df = load_demo_data()

# Compute demo scores (fast for ~200 rows; avoids cache hashing issues with sklearn Pipeline)
all_scores = clf.predict_proba(demo_df)[:, 1]

# Sidebar controls
st.sidebar.header("Controls")

budget_label = st.sidebar.selectbox("Alert budget", ["5%", "10%", "20%"], index=1)
budget_frac = {"5%": 0.05, "10%": 0.10, "20%": 0.20}[budget_label]

st.sidebar.subheader("Select a patient example")
options = make_patient_options(all_scores)
selection = st.sidebar.selectbox("Patient", options, index=0)
idx = int(selection.split("|")[0].replace("Row", "").strip())

x_row = demo_df.iloc[[idx]]  # keep as DataFrame
proba = float(all_scores[idx])

# Budget threshold computed over demo set (illustrative)
budget_threshold = threshold_for_alert_budget(all_scores, budget_frac)
flagged = proba >= budget_threshold

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Predicted Risk")
    st.metric("Risk of <30-day readmission", f"{proba:.3f}")

    st.write(f"**Alert budget:** Top {budget_label} of patients")
    st.write(f"**Budget threshold (demo set):** {budget_threshold:.3f}")
    st.write("**Flagged for follow-up?**", "✅ Yes" if flagged else "❌ No")

    st.subheader("Selected Patient Features")
    st.dataframe(x_row.T, use_container_width=True)

with col2:
    st.subheader("Top 10 Drivers (SHAP)")
    st.caption("Positive values push risk higher; negative values push risk lower.")

    feature_names = preprocess.get_feature_names_out()
    x_trans = preprocess.transform(x_row)

    explainer = get_explainer(model)
    sv_raw = explainer.shap_values(x_trans)
    sv_1 = shap_vector_for_positive_class(sv_raw)

    top_n = 10
    top_idx = np.argsort(np.abs(sv_1))[::-1][:top_n]

    drivers = pd.DataFrame({
        "Feature": [feature_names[i] for i in top_idx],
        "SHAP contribution": [float(sv_1[i]) for i in top_idx],
    }).sort_values("SHAP contribution", ascending=False).reset_index(drop=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#ff4b4b" if v > 0 else "#1f77b4" for v in drivers["SHAP contribution"]]
    ax.barh(drivers["Feature"][::-1], drivers["SHAP contribution"][::-1], color=colors[::-1])
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("SHAP contribution")
    ax.set_ylabel("")
    ax.set_title("Top drivers for this prediction")
    plt.tight_layout()
    st.pyplot(fig)

    st.table(drivers)

st.divider()
st.caption(
    "Note: This demo uses pre-sampled example patients. "
    "Alert-budget thresholding is computed over the demo set to illustrate operational decision-making."
)
