def prepare_features_serve(df):
    # Ensure 'total sulfur dioxide' is in mg/L to match training data
    # If the input is in g/L, multiply by 1000
    if 'total sulfur dioxide' in df.columns:
        # Assuming the input is in g/L based on the 0.001 ratio (1/1000)
        df['total sulfur dioxide'] = df['total sulfur dioxide'] * 1000
    return df