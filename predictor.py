"""
ML Prediction Module
Simple ML models to forecast trends or salary ranges.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Optional, Tuple
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class JobPredictor:
    """ML models for predicting job market trends and salaries."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize predictor with model directory."""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
        self.label_encoders = {}
    
    def prepare_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for salary prediction."""
        df_clean = df.copy()
        
        # Extract numeric salary from string
        def extract_avg_salary(salary_str):
            if pd.isna(salary_str) or salary_str == 'N/A':
                return None
            import re
            numbers = re.findall(r'\d+', str(salary_str).replace(',', ''))
            if len(numbers) >= 2:
                return (int(numbers[0]) + int(numbers[1])) / 2
            elif len(numbers) == 1:
                return int(numbers[0])
            return None
        
        df_clean['salary_numeric'] = df_clean['salary'].apply(extract_avg_salary)
        df_clean = df_clean[df_clean['salary_numeric'].notna()]
        
        return df_clean
    
    def encode_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Handle unseen categories
                    known_classes = set(self.label_encoders[col].classes_)
                    df_encoded[col] = df_encoded[col].astype(str).apply(
                        lambda x: self.label_encoders[col].transform([x])[0] if x in known_classes else -1
                    )
        
        return df_encoded
    
    def predict_salary(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict:
        """Train a model to predict salary based on job features."""
        df_clean = self.prepare_salary_data(df)
        
        if len(df_clean) < 10:
            return {
                'error': 'Insufficient data for salary prediction. Need at least 10 jobs with salary information.'
            }
        
        # Select features
        if features is None:
            features = ['title', 'location', 'company']
        
        # Filter available features
        available_features = [f for f in features if f in df_clean.columns]
        
        if not available_features:
            return {'error': 'No valid features found for prediction.'}
        
        # Encode categorical features
        df_encoded = self.encode_features(df_clean, available_features)
        
        # Prepare X and y
        X = df_encoded[available_features]
        y = df_encoded['salary_numeric']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models_to_try = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_model_name = None
        
        results = {}
        
        for name, model in models_to_try.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2_score': r2
            }
            
            if mae < best_score:
                best_score = mae
                best_model = model
                best_model_name = name
        
        # Save best model
        self.models['salary_predictor'] = best_model
        model_path = os.path.join(self.model_dir, 'salary_predictor.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        return {
            'model': best_model_name,
            'metrics': results,
            'best_model_metrics': results[best_model_name],
            'features_used': available_features,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_future_salary(self, title: str, location: str = "", company: str = "") -> Optional[float]:
        """Predict salary for a given job title, location, and company."""
        if 'salary_predictor' not in self.models:
            return None
        
        model = self.models['salary_predictor']
        
        # Create feature vector
        features = {}
        if 'title' in self.label_encoders:
            features['title'] = title if title in self.label_encoders['title'].classes_ else title
        if 'location' in self.label_encoders:
            features['location'] = location if location in self.label_encoders['location'].classes_ else location
        if 'company' in self.label_encoders:
            features['company'] = company if company in self.label_encoders['company'].classes_ else company
        
        # Encode features
        encoded_features = []
        feature_names = []
        for feat_name, feat_value in features.items():
            if feat_name in self.label_encoders:
                try:
                    encoded = self.label_encoders[feat_name].transform([feat_value])[0]
                    encoded_features.append(encoded)
                    feature_names.append(feat_name)
                except ValueError:
                    # Unknown category
                    encoded_features.append(-1)
                    feature_names.append(feat_name)
        
        if not encoded_features:
            return None
        
        # Predict
        X_pred = np.array([encoded_features])
        prediction = model.predict(X_pred)[0]
        
        return max(0, prediction)  # Ensure non-negative
    
    def predict_hiring_trends(self, df: pd.DataFrame, days_ahead: int = 30) -> pd.DataFrame:
        """Predict future hiring trends based on historical data."""
        if 'posted_date' not in df.columns:
            return pd.DataFrame()
        
        df['posted_date'] = pd.to_datetime(df['posted_date'])
        df['date'] = df['posted_date'].dt.date
        
        # Aggregate daily counts
        daily_counts = df.groupby('date').size().reset_index(name='job_count')
        daily_counts = daily_counts.sort_values('date')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        # Create time features
        daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
        daily_counts['day_of_month'] = daily_counts['date'].dt.day
        daily_counts['month'] = daily_counts['date'].dt.month
        daily_counts['days_since_start'] = (daily_counts['date'] - daily_counts['date'].min()).dt.days
        
        # Simple moving average for trend
        window = min(7, len(daily_counts) // 2)
        if window > 0:
            daily_counts['ma'] = daily_counts['job_count'].rolling(window=window, center=True).mean().fillna(
                daily_counts['job_count'].mean()
            )
        else:
            daily_counts['ma'] = daily_counts['job_count'].mean()
        
        # Predict future dates
        last_date = daily_counts['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        # Simple trend prediction (using moving average and seasonal patterns)
        avg_trend = daily_counts['ma'].tail(7).mean() if len(daily_counts) >= 7 else daily_counts['job_count'].mean()
        
        predictions = []
        for i, future_date in enumerate(future_dates):
            day_of_week = future_date.dayofweek
            day_of_month = future_date.day
            month = future_date.month
            
            # Get similar historical patterns
            similar_days = daily_counts[
                (daily_counts['date'].dt.dayofweek == day_of_week) &
                (daily_counts['date'].dt.month == month)
            ]
            
            if len(similar_days) > 0:
                predicted_count = similar_days['job_count'].mean()
            else:
                predicted_count = avg_trend
            
            predictions.append({
                'date': future_date.date(),
                'predicted_job_count': max(0, round(predicted_count, 2))
            })
        
        return pd.DataFrame(predictions)
    
    def predict_skill_demand(self, df: pd.DataFrame, skills_list: List[List[str]], 
                            days_ahead: int = 30) -> Dict[str, float]:
        """Predict future demand for skills based on historical trends."""
        if 'posted_date' not in df.columns:
            return {}
        
        # Count skill frequency over time
        df['posted_date'] = pd.to_datetime(df['posted_date'])
        
        skill_demand = {}
        for skills in skills_list:
            if isinstance(skills, list):
                for skill in skills:
                    skill_lower = skill.lower().strip()
                    if skill_lower not in skill_demand:
                        skill_demand[skill_lower] = []
        
        # Calculate trend for each skill
        skill_trends = {}
        for skill in skill_demand.keys():
            # Count occurrences over time windows
            recent_date = df['posted_date'].max()
            old_date = recent_date - pd.Timedelta(days=30)
            
            recent_jobs = df[df['posted_date'] >= old_date]
            older_jobs = df[df['posted_date'] < old_date]
            
            recent_count = 0
            older_count = 0
            
            for idx, row in recent_jobs.iterrows():
                job_skills = row.get('skills_list', [])
                if isinstance(job_skills, str):
                    job_skills = [s.strip() for s in job_skills.split(',')]
                if skill in [s.lower().strip() for s in job_skills]:
                    recent_count += 1
            
            for idx, row in older_jobs.iterrows():
                job_skills = row.get('skills_list', [])
                if isinstance(job_skills, str):
                    job_skills = [s.strip() for s in job_skills.split(',')]
                if skill in [s.lower().strip() for s in job_skills]:
                    older_count += 1
            
            # Calculate growth rate
            if older_count > 0:
                growth_rate = (recent_count - older_count) / older_count
            else:
                growth_rate = 0.1 if recent_count > 0 else 0
            
            skill_trends[skill] = {
                'current_demand': recent_count,
                'growth_rate': growth_rate,
                'predicted_demand': max(0, recent_count * (1 + growth_rate))
            }
        
        return skill_trends
    
    def load_model(self, model_name: str):
        """Load a saved model."""
        # Security: Sanitize model name to prevent path traversal
        import re
        safe_model_name = re.sub(r'[^\w\-_]', '_', str(model_name))[:50]
        
        model_path = os.path.join(self.model_dir, f'{safe_model_name}.pkl')
        model_path = os.path.normpath(model_path)
        
        # Ensure path is within model directory
        abs_model_dir = os.path.abspath(self.model_dir)
        abs_model_path = os.path.abspath(model_path)
        if not abs_model_path.startswith(abs_model_dir):
            raise ValueError("Invalid model path detected")
        
        if os.path.exists(model_path):
            # Security: Only load pickle files from trusted sources
            # In production, consider using a safer serialization format or verifying file integrity
            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            return True
        return False


if __name__ == "__main__":
    predictor = JobPredictor()
    # Example usage
    # salary_results = predictor.predict_salary(df)
    # trends = predictor.predict_hiring_trends(df, days_ahead=30)

