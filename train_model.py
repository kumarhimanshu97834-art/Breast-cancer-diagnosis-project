"""
Train and save ML models for Breast Cancer Diagnosis prediction.
Run this script once to generate model files before running the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

print("=" * 70)
print("BREAST CANCER DIAGNOSIS - MODEL TRAINING")
print("=" * 70)

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================
print("\n📊 Loading data...")
data = pd.read_csv("data.csv")
print(f"✓ Data loaded: {data.shape[0]} records, {data.shape[1]} features")

# Encode target variable
print("\n🔄 Preprocessing data...")
le = LabelEncoder()
data['diagnosis'] = le.fit_transform(data['diagnosis'])

# Separate features and target
X = data.drop(["diagnosis", "id"], axis=1)
y = data["diagnosis"]

print(f"✓ Features: {X.shape[1]}")
print(f"✓ Classes: Benign (0), Malignant (1)")

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save preprocessors
print("\n💾 Saving preprocessors...")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: scaler.pkl")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("✓ Label Encoder saved: label_encoder.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Training set: {X_train.shape[0]} records")
print(f"✓ Test set: {X_test.shape[0]} records")

# ============================================================================
# TRAIN MODELS
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

models = {}

# 1. Random Forest
print("\n🌲 Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
models['random_forest'] = rf
print(f"✓ Random Forest trained - Accuracy: {rf_acc:.4f}")

# 2. Support Vector Machine
print("\n📍 Training Support Vector Machine...")
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
models['svm'] = svm
print(f"✓ SVM trained - Accuracy: {svm_acc:.4f}")

# 3. Logistic Regression
print("\n📊 Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
models['logistic_regression'] = lr
print(f"✓ Logistic Regression trained - Accuracy: {lr_acc:.4f}")

# 4. XGBoost
print("\n🚀 Training XGBoost Classifier...")
xgb = XGBClassifier(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
models['xgboost'] = xgb
print(f"✓ XGBoost trained - Accuracy: {xgb_acc:.4f}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

for model_name, model in models.items():
    filename = f"{model_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ {model_name.upper()} saved: {filename}")

# ============================================================================
# DETAILED PERFORMANCE METRICS
# ============================================================================
print("\n" + "=" * 70)
print("DETAILED MODEL PERFORMANCE METRICS")
print("=" * 70)

model_predictions = [
    ('Random Forest', rf_pred),
    ('SVM', svm_pred),
    ('Logistic Regression', lr_pred),
    ('XGBoost', xgb_pred)
]

results = []

for name, predictions in model_predictions:
    print(f"\n{name}:")
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

best_model = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"\n🏆 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")

print("\n" + "=" * 70)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 70)
print("\n🚀 You can now run the Streamlit app:")
print("   Command: streamlit run app.py")
print("\n")
