# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_column_names(df):
    """
    Standardizes column names to snake_case (lowercase with underscores).
    Example: "Pay Amount" -> "pay_amount"
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    return df

def handle_missing_values(df, strategy='median', fill_value=None):
    """
    Fills missing values based on the chosen strategy.
    Strategies: 'median', 'mean', 'mode', 'constant' (requires fill_value)
    """
    # Select numeric and categorical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Fill Numeric
    if strategy == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif strategy == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == 'constant':
        df[num_cols] = df[num_cols].fillna(fill_value)

    # Fill Categorical (Always mode or 'Unknown')
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    
    return df

def remove_outliers_iqr(df, columns, factor=1.5):
    """
    Removes rows containing outliers in the specified columns using IQR.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        
        # Filter
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
    return df

def encode_categorical(df, target_col=None):
    """
    Applies Label Encoding to all categorical columns.
    Note: For production, you might want OneHotEncoder or save the LabelEncoder objects.
    """
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        if col != target_col: # Don't encode the target yet (if want to keep it separate)
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df