import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
import joblib
from typing import Dict, Any, Tuple
import streamlit as st
from utils import triguna_mapping

class DataProcessor:
    def __init__(self):
        self.numerical_features = ['mood', 'sleep_hours', 'screen_time', 'goal_achieved']
        self.categorical_features = ['addiction_type']
        self.derived_features = ['screen_sleep_ratio']
        self.triguna_features = ['sattva', 'rajas', 'tamas']
        
    def load_data(self, csv_path: str = 'data/raw/sample_data_augmented.csv') -> pd.DataFrame:
        """Load augmented data with error handling."""
        try:
            if Path(csv_path).exists():
                df = pd.read_csv(csv_path)
            else:
                st.warning(f"Dataset {csv_path} not found. Using fallback empty DF.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['hour'] = df['date'].dt.hour.fillna(12)
        df['is_weekend'] = df['date'].dt.weekday.fillna(0) >= 5
        df['screen_sleep_ratio'] = df['screen_time'] / (df['sleep_hours'] + 1e-6)
        df['sattva'] = pd.to_numeric(df['sattva'], errors='coerce').fillna(50)
        df['rajas'] = pd.to_numeric(df['rajas'], errors='coerce').fillna(25)
        df['tamas'] = pd.to_numeric(df['tamas'], errors='coerce').fillna(25)
        df['dominant_guna'] = df[['sattva', 'rajas', 'tamas']].idxmax(axis=1)
        return df
    
    def prepare_features_target(self, df: pd.DataFrame):
        if df.empty:
            return pd.DataFrame(), pd.Series(), pd.Series()
        feature_cols = self.numerical_features + self.categorical_features + self.derived_features + ['hour', 'is_weekend']
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].copy().fillna(0)
        y_risk = df.get('craving_level', pd.Series([50]*len(df))).fillna(50)
        y_triguna_class = df.get('dominant_guna', pd.Series(['sattva']*len(df)))
        return X, y_risk, y_triguna_class

class ModelTrainer:
    def __init__(self):
        self.processor = DataProcessor()
        self.numerical_features = []
        self.categorical_features = []
        self.regressor = None
        self.triguna_classifier = None
        self.preprocessor = None
        self.explainer = None
        self.triguna_le = None
        
    def auto_detect_features(self, X: pd.DataFrame):
        """Dynamic feature detection."""
        if X.empty:
            self.numerical_features = ['mood', 'sleep_hours', 'screen_time']
            self.categorical_features = ['addiction_type']
            return
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
    def build_preprocessor(self, X: pd.DataFrame = None):
        if X is not None:
            self.auto_detect_features(X)
            
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        transformers = [('num', num_transformer, self.numerical_features)]
        if self.categorical_features:
            transformers.append(('cat', cat_transformer, self.categorical_features))
            
        self.preprocessor = ColumnTransformer(transformers)
        
    def train(self, regenerate_data: bool = False):
        """Robust training with error handling."""
        df = self.processor.load_data()
        if df.empty:
            st.error("Cannot train: No training data available.")
            return
            
        X, y_risk, y_triguna_class = self.processor.prepare_features_target(df)
        
        self.build_preprocessor(X)
        X_processed = self.preprocessor.fit_transform(X)
        
        # Risk regressor
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.regressor.fit(X_processed, y_risk)
        
        # Triguna classifier
        self.triguna_le = LabelEncoder()
        y_triguna_encoded = self.triguna_le.fit_transform(y_triguna_class)
        self.triguna_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.triguna_classifier.fit(X_processed, y_triguna_encoded)
        
        self.explainer = shap.TreeExplainer(self.regressor)
        
        joblib.dump(self, 'model_trained.joblib')
        st.success("✅ Model trained successfully: Risk + Triguna ML!")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safe prediction."""
        try:
            df_input = pd.DataFrame([input_data])
            df_input['screen_sleep_ratio'] = df_input['screen_time'] / (df_input['sleep_hours'] + 1e-6)
            df_input['hour'] = pd.to_datetime('now').hour
            df_input['is_weekend'] = pd.to_datetime('now').weekday() >= 5
            df_input = df_input.fillna(0)
            
            feature_cols = self.numerical_features + self.categorical_features
            X_input = df_input[[col for col in feature_cols if col in df_input.columns]]
            
            if self.preprocessor is None or self.regressor is None:
                return {'risk_score': 50.0, 'risk_state': 'medium', 'triguna': {'sattva':50, 'rajas':25, 'tamas':25}, 'error': 'Model not ready'}
            
            X_proc = self.preprocessor.transform(X_input)
            
            risk_score = self.regressor.predict(X_proc)[0]
            triguna_pred_idx = self.triguna_classifier.predict(X_proc)[0]
            triguna_dominant = self.triguna_le.inverse_transform([triguna_pred_idx])[0]
            
            triguna_probs = triguna_mapping(df_input)[0]
            triguna_probs['dominant'] = triguna_dominant
            
            risk_state = 'high' if risk_score > 70 else 'medium' if risk_score > 40 else 'low'
            
            return {
                'risk_score': float(risk_score),
                'risk_state': risk_state,
                'triguna': triguna_probs,
                'triguna_dominant': triguna_dominant,
                'shap_top_feature': 'screen_time'
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return {'risk_score': 50.0, 'risk_state': 'medium', 'triguna': {'sattva':50, 'rajas':25, 'tamas':25}}
