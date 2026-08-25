# Diabetes Prediction App

## Files

- `app.py` — Streamlit application
- `diabetes_catboost_model.pkl` — trained CatBoost model
- `requirements.txt` — required Python packages
- `run_app.bat` — Windows launcher

## Run on Windows

1. Put all four files in the same folder.
2. Double-click `run_app.bat`.

Or open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app does NOT require `diabetes_scaler.pkl` or
`diabetes_preprocessor.pkl`.

The preprocessing needed by the final CatBoost model is reproduced
inside `app.py` using the exact notebook medians and feature order.
