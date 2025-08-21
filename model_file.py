import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class AdvancedSalesForecaster:
    def __init__(self):
        """Initialize the forecaster with available models"""
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
        }
        self.scaler = StandardScaler()
        self.feature_names = None  # store features used for training

    def generate_sample_data(self, start_date, periods, trend_strength=1.0, seasonality_strength=0.3, noise_level=0.1):
        """
        Generate synthetic sales data with trend, seasonality, and noise
        
        Args:
            start_date: Start date for the data
            periods: Number of data points to generate
            trend_strength: Strength of the upward trend
            seasonality_strength: Strength of seasonal patterns
            noise_level: Amount of random noise
            
        Returns:
            DataFrame with generated sales data
        """
        dates = pd.date_range(start=start_date, periods=periods, freq='D')
        
        # Create trend component
        trend = np.linspace(1000, 1000 + trend_strength * 500, periods)
        
        # Create seasonal patterns
        weekly_season = 200 * np.sin(2 * np.pi * np.arange(periods) / 7) * seasonality_strength
        monthly_season = 300 * np.sin(2 * np.pi * np.arange(periods) / 30) * seasonality_strength
        
        # Add random holiday effects
        holiday_effects = np.random.choice([0, 0, 0, 0, 500], periods, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        
        # Add noise
        noise = np.random.normal(0, 100 * noise_level, periods)
        
        # Combine all components
        sales = trend + weekly_season + monthly_season + holiday_effects + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative sales
        
        return pd.DataFrame({
            'date': dates,
            'sales': sales,
            'day_of_week': dates.dayofweek,
            'day_of_month': dates.day,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int)
        })

    def create_features(self, df):
        """
        Create engineered features for the model
        
        Args:
            df: DataFrame with date and sales columns
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Lag features
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_7'] = df['sales'].shift(7)
        
        # Moving averages
        df['sales_ma_7'] = df['sales'].rolling(window=7).mean()
        df['sales_ma_30'] = df['sales'].rolling(window=30).mean()
        
        # Trend feature
        df['trend'] = np.arange(len(df))
        
        # Cyclical features for day of month
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_month'] / 30)
        
        # Cyclical features for day of week
        df['sin_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df.dropna()

    def prepare_data(self, df, target_col='sales'):
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
            
        Returns:
            X (features), y (target)
        """
        feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
        X = df[feature_cols]
        y = df[target_col]
        return X, y

    def train_models(self, X_train, y_train):
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        trained_models = {}
        self.feature_names = list(X_train.columns)
        
        for name, model in self.models.items():
            if name == 'Linear Regression':
                # Scale features for linear regression
                X_scaled = self.scaler.fit_transform(X_train)
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            trained_models[name] = model
            
        return trained_models

    def predict(self, model_name, model, X_test):
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the model
            model: Trained model object
            X_test: Test features
            
        Returns:
            Predictions array
        """
        # Align feature order and fill missing columns
        X_test = X_test.copy()
        for col in self.feature_names:
            if col not in X_test:
                X_test[col] = 0
        X_test = X_test[self.feature_names]
        
        if model_name == 'Linear Regression':
            X_scaled = self.scaler.transform(X_test)
            return model.predict(X_scaled)
        else:
            return model.predict(X_test)

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100  # Avoid division by zero
        
        return {
            'MAE': mae, 
            'MSE': mse, 
            'RMSE': rmse, 
            'R²': r2, 
            'MAPE': mape
        }

    def get_feature_importance(self, model_name, model):
        """
        Get feature importance for tree-based models
        
        Args:
            model_name: Name of the model
            model: Trained model object
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        else:
            return None

    def validate_data(self, df):
        """
        Validate input data format
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Boolean indicating if data is valid
        """
        required_columns = ['date', 'sales']
        if not all(col in df.columns for col in required_columns):
            return False, f"Missing required columns: {required_columns}"
        
        if df['sales'].isnull().any():
            return False, "Sales column contains null values"
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            return False, "Date column must be datetime type"
        
        return True, "Data is valid"

    def create_forecast_dataframe(self, dates, predictions, confidence_intervals=None):
        """
        Create a formatted forecast dataframe
        
        Args:
            dates: Future dates
            predictions: Predicted values
            confidence_intervals: Optional confidence intervals
            
        Returns:
            Formatted forecast DataFrame
        """
        forecast_df = pd.DataFrame({
            'date': dates,
            'predicted_sales': predictions,
        })
        
        if confidence_intervals is not None:
            forecast_df['lower_bound'] = confidence_intervals[0]
            forecast_df['upper_bound'] = confidence_intervals[1]
        
        return forecast_df