import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# Diabetes Risk Prediction
# ============================================================
# This version intentionally uses ONLY the saved CatBoost model.
#
# The cleaned notebook trained CatBoost on:
#   1. zero-as-missing median imputation
#   2. NO StandardScaler
#
# The exact medians below are taken from the saved notebook
# preprocessor, so this app can run even if the separate
# diabetes_preprocessor.pkl file is not present.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "diabetes_catboost_model.pkl"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Exact imputation values from the notebook's saved preprocessor.
MEDIANS = {
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 28.0,
    "Insulin": 102.5,
    "BMI": 32.4,
    "Pregnancies": 3.0,
    "DiabetesPedigreeFunction": 0.3825,
    "Age": 29.0,
}

# Exact feature order produced by the notebook preprocessor.
MODEL_FEATURE_ORDER = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "Pregnancies",
    "DiabetesPedigreeFunction",
    "Age",
]


st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found:\n{MODEL_PATH}\n\n"
            "Put diabetes_catboost_model.pkl in the same folder as app.py."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error("❌ Model could not be loaded.")
    st.code(str(exc))
    st.stop()


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 600;
        margin-top: 20px;
    }

    .disclaimer {
        font-size: 13px;
        color: #777;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="main-title">🩺 Diabetes Risk Prediction</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Machine Learning Based Diabetes Risk Assessment"
    "</div>",
    unsafe_allow_html=True,
)

st.divider()


with st.expander("ℹ️ About this project"):
    st.write(
        """
        This application uses the final CatBoost model from the cleaned
        diabetes prediction notebook.

        The deployment pipeline is:

        Patient Input
        → Exact notebook preprocessing
        → CatBoost Model
        → Prediction + Probability

        Important: the KNN StandardScaler is NOT used by this CatBoost
        model.

        Zero values in Glucose, Blood Pressure, Skin Thickness, Insulin,
        and BMI are treated as missing values and replaced with the
        training-set median, exactly as in the notebook.
        """
    )


st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True,
)

st.write("Enter the patient's information below.")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
        help="Number of times the patient has been pregnant.",
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120,
        step=1,
        help="Plasma glucose concentration.",
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1,
        help="Diastolic blood pressure.",
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Triceps skin fold thickness.",
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80,
        step=1,
        help="Serum insulin level.",
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index.",
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.01,
        help="Diabetes pedigree function value.",
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1,
        help="Patient age.",
    )


st.divider()


def prepare_model_input(values):
    """Reproduce the notebook's exact CatBoost preprocessing."""

    raw = pd.DataFrame(
        [values],
        columns=FEATURES,
    )

    # Notebook behavior:
    # 0 means missing for these five medical measurements.
    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    for feature in zero_as_missing:
        if raw.loc[0, feature] == 0:
            raw.loc[0, feature] = MEDIANS[feature]

    # Other features use ordinary median imputation if missing.
    for feature in ["Pregnancies", "DiabetesPedigreeFunction", "Age"]:
        if pd.isna(raw.loc[0, feature]):
            raw.loc[0, feature] = MEDIANS[feature]

    # The ColumnTransformer in the notebook outputs this exact order.
    return raw[MODEL_FEATURE_ORDER]


def predict_patient(values):
    model_input = prepare_model_input(values)

    prediction = int(np.asarray(model.predict(model_input)).ravel()[0])
    probability = float(model.predict_proba(model_input)[0, 1])

    return prediction, probability, model_input


if st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True,
    type="primary",
):

    values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]

    try:
        prediction, probability, model_input = predict_patient(values)
    except Exception as exc:
        st.error("❌ Prediction failed.")
        st.code(str(exc))
        st.stop()

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True,
    )

    if prediction == 1:
        st.error("⚠️ Higher Risk of Diabetes")
        st.write(
            "The model predicts the positive diabetes-risk class "
            "for the provided information."
        )
    else:
        st.success("✅ Lower Risk of Diabetes")
        st.write(
            "The model predicts the negative diabetes-risk class "
            "for the provided information."
        )

    st.subheader("Prediction Probability")

    st.metric(
        label="Estimated Probability of Diabetes",
        value=f"{probability * 100:.2f}%",
    )

    st.progress(float(np.clip(probability, 0.0, 1.0)))

    if probability < 0.30:
        st.info(
            "The model assigns a relatively low probability "
            "to the positive class."
        )
    elif probability < 0.70:
        st.warning(
            "The model assigns an intermediate probability "
            "to the positive class."
        )
    else:
        st.warning(
            "The model assigns a relatively high probability "
            "to the positive class."
        )

    with st.expander("🔎 View submitted values"):
        st.dataframe(
            pd.DataFrame(
                {"Feature": FEATURES, "Value": values}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("🧪 View values sent to CatBoost"):
        st.dataframe(
            model_input,
            use_container_width=True,
            hide_index=True,
        )


st.divider()

st.markdown(
    '<div class="disclaimer">'
    "⚠️ This application is an academic machine learning project. "
    "It is not a medical diagnostic tool and should not replace "
    "professional medical advice."
    "</div>",
    unsafe_allow_html=True,
)
