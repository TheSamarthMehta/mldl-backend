"""
Data Processing Utilities
Functions for data preprocessing and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(file_path, file_type='csv'):
    """
    Load data from file
    
    Args:
        file_path: Path to data file
        file_type: Type of file (csv, json, xlsx)
    
    Returns:
        DataFrame with loaded data
    """
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path)
    elif file_type == 'xlsx':
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def preprocess_data(df, target_column=None, scale=True, handle_missing=True):
    """
    Preprocess data for training or prediction
    
    Args:
        df: Input DataFrame
        target_column: Name of target column (if applicable)
        scale: Whether to scale features
        handle_missing: Whether to handle missing values
    
    Returns:
        Tuple of (X, y, scaler) or (X, scaler) if no target column
    """
    # Create a copy
    data = df.copy()
    
    # Handle missing values
    if handle_missing:
        # Fill numeric columns with mean
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != target_column:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
    
    # Separate features and target
    if target_column and target_column in data.columns:
        X = data.drop(columns=[target_column])
        y = data[target_column]
    else:
        X = data
        y = None
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale features
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    if y is not None:
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            label_encoders['target'] = le
        
        return X, y, scaler, label_encoders
    else:
        return X, scaler, label_encoders


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def extract_features(data, feature_list=None):
    """
    Extract specific features from data
    
    Args:
        data: Input data (dict or DataFrame)
        feature_list: List of feature names to extract
    
    Returns:
        DataFrame with extracted features
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
    
    if feature_list:
        # Only keep specified features
        available_features = [f for f in feature_list if f in df.columns]
        df = df[available_features]
    
    return df
