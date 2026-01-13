"""
Input Validation Utilities
Validates API inputs and data
"""

import numpy as np
import pandas as pd


def validate_prediction_input(data):
    """
    Validate input data for predictions
    
    Args:
        data: Input data (dict or DataFrame)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if data is None:
        return False, "No data provided"
    
    if isinstance(data, dict):
        # Check if features key exists
        if 'features' in data:
            features = data['features']
            if not isinstance(features, (dict, list)):
                return False, "Features must be a dictionary or list"
            
            if isinstance(features, dict) and not features:
                return False, "Features dictionary is empty"
            
            if isinstance(features, list) and len(features) == 0:
                return False, "Features list is empty"
        
        elif not data:
            return False, "Data dictionary is empty"
    
    elif isinstance(data, (list, pd.DataFrame, np.ndarray)):
        if len(data) == 0:
            return False, "Data is empty"
    
    else:
        return False, f"Invalid data type: {type(data)}"
    
    return True, None


def validate_training_data(X, y):
    """
    Validate training data
    
    Args:
        X: Features
        y: Labels
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if X is None or y is None:
        return False, "Features or labels are None"
    
    # Convert to numpy arrays if needed
    if isinstance(X, (list, pd.DataFrame)):
        X = np.array(X) if isinstance(X, list) else X.values
    
    if isinstance(y, (list, pd.Series)):
        y = np.array(y) if isinstance(y, list) else y.values
    
    if len(X) == 0 or len(y) == 0:
        return False, "Features or labels are empty"
    
    if len(X) != len(y):
        return False, f"Feature and label count mismatch: {len(X)} vs {len(y)}"
    
    # Check for NaN values
    if np.any(np.isnan(X)):
        return False, "Features contain NaN values"
    
    if np.any(np.isnan(y)):
        return False, "Labels contain NaN values"
    
    return True, None


def validate_file_upload(file, allowed_extensions=None):
    """
    Validate uploaded file
    
    Args:
        file: Uploaded file object
        allowed_extensions: List of allowed file extensions
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_extensions is None:
        allowed_extensions = ['csv', 'json', 'xlsx']
    
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if '.' not in file.filename:
        return False, "File has no extension"
    
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        return False, f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
    
    return True, None


def sanitize_input(input_string):
    """
    Sanitize string input to prevent injection attacks
    
    Args:
        input_string: String to sanitize
    
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        return input_string
    
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
    sanitized = input_string
    
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    return sanitized.strip()
