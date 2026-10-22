def prepare_features_serve(df):
    # Ensure 'total sulfur dioxide' is in the same units as the training data (mg/L)
    # If the input is in g/L, multiply by 1000 to convert to mg/L
    if 'total sulfur dioxide' in df.columns:
        df['total sulfur dioxide'] = df['total sulfur dioxide'] * 1000
    
    # ... rest of feature engineering logic ...
    return df