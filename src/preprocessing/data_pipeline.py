import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df = df.copy()
    
    df.dropna(inplace=True)
    
    df = df[df['age'] > 0]
    df = df[df['age'] < 100]
    
    return df

def engineer_features(df):
    df = df.copy()
    
    if 'tenure' in df.columns and 'age' in df.columns:
        df['tenure_ratio'] = df['tenure'] / df['age']
    
    transaction_cols = [col for col in df.columns if 'month' in col.lower()]
    if transaction_cols:
        df['activity_score'] = df[transaction_cols].sum(axis=1) / (df.get('tenure', 1) + 1)
    
    return df

def preprocess_data(df, scaler=None, fit=True):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'churn' in numeric_cols:
        numeric_cols.remove('churn')
    
    if fit:
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df, scaler

def prepare_dataset(filepath, save_scaler=False):
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)
    df, scaler = preprocess_data(df, fit=True)
    
    if save_scaler:
        joblib.dump(scaler, 'deployment/models/scaler.pkl')
    
    return df, scaler
