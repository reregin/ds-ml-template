# src/train.py
import pandas as pd
import joblib
import os
import sys

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

# XGBoost (Optional - Handle import error if not installed)
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

# Dynamic Path Setup to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config

# ==========================================
# 1. THE MODEL FACTORY
# ==========================================
def get_model(model_name, task_type="classification"):
    """
    Returns an un-trained model instance based on the name in config.
    """
    print(f"🔧 Initializing model: {model_name} ({task_type})")
    
    # --- CLASSIFICATION MODELS ---
    if task_type == "classification":
        if model_name == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
        elif model_name == "logistic_regression":
            return LogisticRegression(random_state=config.RANDOM_STATE, max_iter=1000)
        elif model_name == "svm":
            return SVC(probability=True, random_state=config.RANDOM_STATE)
        elif model_name == "xgboost" and XGBClassifier:
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=config.RANDOM_STATE)
            
    # --- REGRESSION MODELS ---
    elif task_type == "regression":
        if model_name == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
        elif model_name == "linear_regression":
            return LinearRegression()
        elif model_name == "xgboost" and XGBRegressor:
            return XGBRegressor(random_state=config.RANDOM_STATE)

    raise ValueError(f"❌ Model '{model_name}' not supported for task '{task_type}'")

# ==========================================
# 2. THE TRAINING PIPELINE
# ==========================================
def run_training():
    # A. Load Processed Data
    data_path = os.path.join("data", "processed", "clean_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Clean data not found at {data_path}. Run preprocessing first!")
        
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # B. Split Features & Target
    target = config.TARGET_COLUMN
    if target not in df.columns:
        raise ValueError(f"❌ Target column '{target}' not found in dataset.")
        
    X = df.drop(columns=[target])
    y = df[target]
    
    # C. Train/Test Split
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )
    
    # D. Get Model from Factory
    model = get_model(config.MODEL_NAME, task_type=config.TASK_TYPE) 
    
    # E. Train
    print("🚀 Training started...")
    model.fit(X_train, y_train)
    print("✅ Training complete.")
    
    # F. Evaluate
    print("📊 Evaluating model...")
    preds = model.predict(X_test)
    
    # Simple check: Classification or Regression metrics?
    if hasattr(model, "predict_proba"): # Likely classification
        print(classification_report(y_test, preds))
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    else:
        mse = mean_squared_error(y_test, preds)
        print(f"Mean Squared Error: {mse:.4f}")

    # G. Save Model
    save_path = os.path.join("models", f"{config.MODEL_NAME}_v1.pkl")
    joblib.dump(model, save_path)
    print(f"💾 Model saved to: {save_path}")

if __name__ == "__main__":
    run_training()