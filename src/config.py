# src/config.py
import os

# --- PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_DATA_FILE = "dataset.csv" # CHANGE THIS
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", RAW_DATA_FILE)

# --- TARGET VARIABLE ---
TARGET_COLUMN = "target_variable_name" # CHANGE THIS

# --- RANDOM SEED ---
RANDOM_STATE = 42