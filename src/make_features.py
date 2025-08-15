import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['lag_1'] = df['sales'].shift(1)
    df['lag_7'] = df['sales'].shift(7)
    df['roll_mean_7'] = df['sales'].shift(1).rolling(window=7).mean()
    df['roll_mean_28'] = df['sales'].shift(1).rolling(window=28).mean()
    return df
