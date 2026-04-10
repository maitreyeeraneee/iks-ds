import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# import shap  # Optional, install if needed
import joblib
from typing import Dict, Any, Tuple



class DataProcessor:
    def __init__(self):
        self.numerical_features = ['mood', 'sleep_hours', 'screen_time', 'goal_achieved']
        self.categorical_features = ['addiction_type']
        self.derived_features = ['screen_sleep_ratio']
        self.triguna_features = ['sattva', 'rajas', 'tamas']
        
    def load_data(self, df_history: pd.DataFrame = None, csv_path: str = None) -> pd.DataFrame:
        """Load user history or CSV."""
        if df_history is not None and not df_history.empty:
            df = df_history.copy()
        elif csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        else:
            return pd.DataFrame()
        # Add derived
        if 'date' in df:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['hour'] = df['date'].dt.hour.fillna(12)
            df['is_weekend'] = df['date'].dt.weekday.fillna(0) >= 5
        if 'screen_time' in df and 'sleep_hours' in df:
            df['screen_sleep_ratio'] = df['screen_time'] / (df['sleep_hours'] + 1e-6)
        # Triguna flatten
        for guna in ['triguna_sattva', 'triguna_rajas', 'triguna_tamas']:
            if guna in df:
                df[ guna.replace('triguna_', '') ] = df[ guna ]
        df['dominant_guna'] = df[['sattva', 'rajas', 'tamas']].idxmax(axis=1) if all(g in df for g in ['sattva', 'rajas', 'tamas']) else 'sattva'
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
        
    def train(self, df_history: pd.DataFrame = None, regenerate_data: bool = False):
        """Robust training on user history."""
        df = self.processor.load_data(df_history=df_history)
        if df.empty or len(df) < 10:
            st.info("Need 10+ history entries to train model.")
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
        
        # self.explainer = shap.TreeExplainer(self.regressor)  # Requires shap
        
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
            
            def get_triguna_pct(features):
                mood = features.get('mood', 3.0)
                sleep = features.get('sleep_hours', 7.0)
                screen = features.get('screen_time', 4.0)
                goal = features.get('goal_achieved', 0.7)
                sattva_raw = mood / 5 * sleep / 8 * goal
                tamas_raw = (5 - mood) / 5 * screen / 8 * (1 - goal)
                rajas_raw = 1 - sattva_raw - tamas_raw
                rajas_raw = max(0, rajas_raw)
                total_raw = sattva_raw + rajas_raw + tamas_raw or 1
                sattva = round((sattva_raw / total_raw) * 100, 2)
                rajas = round((rajas_raw / total_raw) * 100, 2)
                tamas = round(100 - sattva - rajas, 2)
                return {'sattva': sattva, 'rajas': rajas, 'tamas': tamas}

            triguna_probs = get_triguna_pct(input_data)
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
