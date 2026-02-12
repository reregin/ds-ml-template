# src/data_loader.py
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.config as config

def load_raw_data(path=None):
    """
    Loads the raw data from the path specified in config.py.
    Handles CSV, Excel, and Parquet automatically.
    """
    # Use config path if none provided
    if path is None:
        path = config.RAW_DATA_PATH

    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Error: Data file not found at {path}. Check src/config.py")

    print(f"Loading data from: {path}...")
    
    # Determine file type by extension
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    elif ext == '.parquet':
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"❌ Unsupported file format: {ext}")

    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Test the function if run directly
    try:
        df = load_raw_data()
        print(df.head())
    except Exception as e:
        print(e)