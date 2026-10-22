def prepare_features_serve(df):
    # Ensure total_sulfur_dioxide is in the same unit as training (e.g., mg/L)
    # If serving data is in g/L, convert to mg/L by multiplying by 1000
    if 'total_sulfur_dioxide' in df.columns:
        df['total_sulfur_dioxide'] = df['total_sulfur_dioxide'] * 1000
    return df