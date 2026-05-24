"""
train_models.py
Trains Random Forest and Linear Regression on the Black Friday dataset.
Saves trained models to disk so Django can load them later.
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# === STEP 1: LOAD THE DATA ===
print("Step 1: Loading dataset...")
csv_file = glob.glob("*.csv")[0]
df = pd.read_csv(csv_file)
print(f"  Loaded {len(df):,} rows from {csv_file}")


# === STEP 2: CLEAN THE DATA ===
print("`nStep 2: Cleaning data...")

# Fill missing Product_Category_2 and Product_Category_3 with 0 (means "no category")
df["Product_Category_2"] = df["Product_Category_2"].fillna(0)
df["Product_Category_3"] = df["Product_Category_3"].fillna(0)
print("  Filled missing values with 0")

# Drop columns that aren't useful as features
df = df.drop(["User_ID", "Product_ID"], axis=1)
print("  Dropped User_ID and Product_ID (not useful for prediction)")


# === STEP 3: CONVERT TEXT TO NUMBERS ===
print("`nStep 3: Encoding text columns as numbers...")
# ML models only understand numbers, so we convert text columns:
# Gender M/F -> 1/0
# Age 26-35 -> 2 (third category)
# City A/B/C -> 0/1/2
encoders = {}
text_cols = ["Gender", "Age", "City_Category", "Stay_In_Current_City_Years"]
for col in text_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  Encoded {col}: {list(le.classes_)}")


# === STEP 4: SPLIT INTO FEATURES (X) AND TARGET (y) ===
print("`nStep 4: Splitting features and target...")
X = df.drop("Purchase", axis=1)   # all columns except Purchase
y = df["Purchase"]                # only the Purchase column
print(f"  X (features): {X.shape}")
print(f"  y (target):   {y.shape}")


# === STEP 5: SPLIT INTO TRAINING AND TESTING SETS ===
print("`nStep 5: Train/test split (80% train, 20% test)...")
# We use 80% of data to teach the model, and 20% to test how well it learned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Training set: {len(X_train):,} rows")
print(f"  Testing set:  {len(X_test):,} rows")


# === STEP 6: TRAIN RANDOM FOREST ===
print("`n" + "=" * 60)
print("Training Random Forest (this takes 1-2 minutes)...")
print("=" * 60)
rf = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"  RMSE: `${rf_rmse:,.2f}")
print(f"  MAE:  `${rf_mae:,.2f}")
print(f"  R²:   {rf_r2:.4f}")


# === STEP 7: TRAIN LINEAR REGRESSION ===
print("`n" + "=" * 60)
print("Training Linear Regression...")
print("=" * 60)
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"  RMSE: `${lr_rmse:,.2f}")
print(f"  MAE:  `${lr_mae:,.2f}")
print(f"  R²:   {lr_r2:.4f}")


# === STEP 8: SAVE EVERYTHING ===
print("`n" + "=" * 60)
print("Saving models...")
print("=" * 60)
os.makedirs("ml_models", exist_ok=True)

with open("ml_models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)
print("  Saved random_forest.pkl")

with open("ml_models/linear_regression.pkl", "wb") as f:
    pickle.dump(lr, f)
print("  Saved linear_regression.pkl")

with open("ml_models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
print("  Saved encoders.pkl")

feature_cols = list(X.columns)
with open("ml_models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)
print("  Saved feature_cols.json")

metrics = {
    "random_forest": {"rmse": round(rf_rmse, 2), "mae": round(rf_mae, 2), "r2": round(rf_r2, 4),
                      "feature_importance": dict(zip(feature_cols, [round(float(x), 4) for x in rf.feature_importances_]))},
    "linear_regression": {"rmse": round(lr_rmse, 2), "mae": round(lr_mae, 2), "r2": round(lr_r2, 4)}
}
with open("ml_models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("  Saved metrics.json")

print("`n" + "=" * 60)
print(f"DONE! Best model: {'Random Forest' if rf_r2 > lr_r2 else 'Linear Regression'}")
print("=" * 60)
