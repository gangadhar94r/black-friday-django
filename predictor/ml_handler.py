"""
Machine learning handler for the predictor app.
Loads trained models and makes predictions on form input.
"""
import os
import json
import pickle
import pandas as pd
from pathlib import Path

# Path to the ml_models folder (at the project root)
ML_DIR = Path(__file__).resolve().parent.parent / 'ml_models'

# Cache models in memory so we don't reload on every request
_cache = {}


def _load_models():
    """Lazily load all models and encoders. Called once, then cached."""
    if _cache:
        return _cache

    with open(ML_DIR / 'random_forest.pkl', 'rb') as f:
        _cache['random_forest'] = pickle.load(f)

    with open(ML_DIR / 'linear_regression.pkl', 'rb') as f:
        _cache['linear_regression'] = pickle.load(f)

    with open(ML_DIR / 'encoders.pkl', 'rb') as f:
        _cache['encoders'] = pickle.load(f)

    with open(ML_DIR / 'feature_cols.json', 'r') as f:
        _cache['feature_cols'] = json.load(f)

    with open(ML_DIR / 'metrics.json', 'r') as f:
        _cache['metrics'] = json.load(f)

    return _cache


def classify_tier(amount):
    """Categorize the predicted spending amount into a tier."""
    if amount >= 15000:
        return 'Premium Spender'
    elif amount >= 10000:
        return 'High Spender'
    elif amount >= 5000:
        return 'Average Shopper'
    else:
        return 'Budget Buyer'


def predict(form_data, model_choice='random_forest'):
    """
    Run prediction using the chosen model.
    Takes form data, returns dict with prediction amount and tier.
    """
    cache = _load_models()
    encoders = cache['encoders']
    feature_cols = cache['feature_cols']

    # Validate model choice
    if model_choice not in ('random_forest', 'linear_regression'):
        model_choice = 'random_forest'
    model = cache[model_choice]

    # Build input row from form data
    row = {
        'Gender': form_data.get('Gender', 'M'),
        'Age': form_data.get('Age', '26-35'),
        'Occupation': int(form_data.get('Occupation', 0)),
        'City_Category': form_data.get('City_Category', 'A'),
        'Stay_In_Current_City_Years': form_data.get('Stay_In_Current_City_Years', '2'),
        'Marital_Status': int(form_data.get('Marital_Status', 0)),
        'Product_Category_1': int(form_data.get('Product_Category_1', 1)),
        'Product_Category_2': float(form_data.get('Product_Category_2', 0)),
        'Product_Category_3': float(form_data.get('Product_Category_3', 0)),
    }
    df = pd.DataFrame([row])

    # Apply the same encoders used during training
    for col, le in encoders.items():
        df[col] = df[col].astype(str).map(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
        df[col] = le.transform(df[col])

    # Ensure column order matches training
    df = df[feature_cols]

    # Make the prediction
    amount = float(model.predict(df)[0])
    amount = max(0.0, amount)  # No negative purchases

    return {
        'amount': round(amount, 2),
        'tier': classify_tier(amount),
        'model_used': model_choice,
        'model_display_name': 'Random Forest' if model_choice == 'random_forest' else 'Linear Regression',
    }