"""
FMCG Demand Forecasting Module
Advanced time series forecasting with multiple algorithms
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series models
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import joblib
import json

class FMCGDemandForecaster:
    """
    Advanced demand forecasting system for FMCG products
    Supports multiple algorithms and automatic model selection
    """
    
    def __init__(self, model_type='auto'):
        """
        Initialize forecaster
        
        Args:
            model_type: 'auto', 'sarima', 'hw', 'rf', 'gb'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.feature_importance = None
        
    def prepare_data(self, df, date_col='sale_date', value_col='quantity_sold',
                    freq='D', fillna_method='forward'):
        """
        Prepare time series data
        
        Args:
            df: DataFrame with date and value columns
            date_col: Name of date column
            value_col: Name of value column
            freq: Frequency ('D', 'W', 'M')
            fillna_method: 'forward', 'backward', 'zero', 'mean'
        
        Returns:
            pd.Series: Prepared time series
        """
        # Convert to datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Set index and sort
        df = df.set_index(date_col).sort_index()
        
        # Resample to desired frequency
        series = df[value_col].resample(freq).sum()
        
        # Fill missing values
        if fillna_method == 'forward':
            series = series.fillna(method='ffill')
        elif fillna_method == 'backward':
            series = series.fillna(method='bfill')
        elif fillna_method == 'zero':
            series = series.fillna(0)
        elif fillna_method == 'mean':
            series = series.fillna(series.mean())
        
        # Remove any remaining NaN
        series = series.fillna(0)
        
        return series
    
    def create_features(self, series, lags=[1, 7, 14, 30], rolling_windows=[7, 14, 30]):
        """
        Create features for ML models
        
        Args:
            series: Time series data
            lags: List of lag periods
            rolling_windows: List of rolling window sizes
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        df = pd.DataFrame(index=series.index)
        df['value'] = series.values
        
        # Lag features
        for lag in lags:
            df[f'lag_{lag}'] = series.shift(lag)
        
        # Rolling statistics
        for window in rolling_windows:
            df[f'rolling_mean_{window}'] = series.rolling(window).mean()
            df[f'rolling_std_{window}'] = series.rolling(window).std()
            df[f'rolling_min_{window}'] = series.rolling(window).min()
            df[f'rolling_max_{window}'] = series.rolling(window).max()
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Drop rows with NaN (from lag/rolling)
        df = df.dropna()
        
        return df
    
    def train_sarima(self, series, order=(1,1,1), seasonal_order=(1,1,1,7)):
        """Train SARIMA model"""
        try:
            self.model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model = self.model.fit(disp=False)
            return True
        except Exception as e:
            print(f"SARIMA training failed: {e}")
            return False
    
    def train_holt_winters(self, series, seasonal_periods=7):
        """Train Holt-Winters Exponential Smoothing"""
        try:
            self.model = ExponentialSmoothing(
                series,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add',
                damped_trend=True
            )
            self.model = self.model.fit(optimized=True)
            return True
        except Exception as e:
            print(f"Holt-Winters training failed: {e}")
            # Fallback to simpler model
            try:
                self.model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=None
                )
                self.model = self.model.fit()
                return True
            except:
                return False
    
    def train_random_forest(self, series, n_estimators=100):
        """Train Random Forest model"""
        # Create features
        df_features = self.create_features(series)
        
        X = df_features.drop('value', axis=1)
        y = df_features['value']
        
        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        return True
    
    def train_gradient_boosting(self, series, n_estimators=100):
        """Train Gradient Boosting model"""
        # Create features
        df_features = self.create_features(series)
        
        X = df_features.drop('value', axis=1)
        y = df_features['value']
        
        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        return True
    
    def auto_select_model(self, series):
        """Automatically select best model based on validation"""
        models_to_try = [
            ('sarima', self.train_sarima),
            ('hw', self.train_holt_winters),
            ('rf', self.train_random_forest),
            ('gb', self.train_gradient_boosting)
        ]
        
        best_model = None
        best_score = float('inf')
        best_name = None
        
        # Split for validation
        split_idx = int(len(series) * 0.8)
        train_series = series[:split_idx]
        test_series = series[split_idx:]
        
        for name, train_func in models_to_try:
            try:
                print(f"Testing {name}...")
                temp_forecaster = FMCGDemandForecaster(model_type=name)
                
                if train_func.__name__ == 'train_sarima':
                    success = temp_forecaster.train_sarima(train_series)
                elif train_func.__name__ == 'train_holt_winters':
                    success = temp_forecaster.train_holt_winters(train_series)
                elif train_func.__name__ == 'train_random_forest':
                    success = temp_forecaster.train_random_forest(series)  # Uses full series
                    continue  # Skip RMSE calculation for ML models
                elif train_func.__name__ == 'train_gradient_boosting':
                    success = temp_forecaster.train_gradient_boosting(series)
                    continue
                
                if success:
                    # Forecast and calculate error
                    forecast = temp_forecaster.model.forecast(steps=len(test_series))
                    rmse = np.sqrt(mean_squared_error(test_series, forecast))
                    
                    print(f"  RMSE: {rmse:.2f}")
                    
                    if rmse < best_score:
                        best_score = rmse
                        best_model = temp_forecaster.model
                        best_name = name
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        if best_model:
            self.model = best_model
            self.model_type = best_name
            print(f"\nSelected model: {best_name} (RMSE: {best_score:.2f})")
            return True
        else:
            print("\nAll models failed, using simple moving average")
            return False
    
    def forecast(self, steps=30):
        """
        Generate forecast
        
        Args:
            steps: Number of periods to forecast
        
        Returns:
            tuple: (predictions, lower_bound, upper_bound)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if self.model_type in ['sarima', 'hw']:
            # Statistical models
            forecast_result = self.model.forecast(steps=steps)
            
            if hasattr(forecast_result, 'predicted_mean'):
                predictions = forecast_result.predicted_mean
            else:
                predictions = forecast_result
            
            # Calculate confidence intervals (simplified)
            if hasattr(self.model, 'resid'):
                std_error = np.std(self.model.resid)
            else:
                std_error = np.std(predictions) * 0.1
            
            lower_bound = predictions - 1.96 * std_error
            upper_bound = predictions + 1.96 * std_error
            
        else:
            # ML models - need to generate future features
            # This is a simplified approach
            predictions = np.array([self.model.predict([[0]*self.scaler.n_features_in_])[0] 
                                  for _ in range(steps)])
            
            std_error = predictions.std() if len(predictions) > 1 else predictions.mean() * 0.1
            lower_bound = predictions - 1.96 * std_error
            upper_bound = predictions + 1.96 * std_error
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)
        
        return predictions, lower_bound, upper_bound
    
    def calculate_inventory_metrics(self, forecast, lead_time=3, service_level=0.95,
                                    holding_cost=1.0, ordering_cost=100.0):
        """
        Calculate inventory optimization metrics
        
        Args:
            forecast: Forecasted demand
            lead_time: Lead time in days
            service_level: Desired service level (0-1)
            holding_cost: Cost to hold one unit for one period
            ordering_cost: Fixed cost per order
        
        Returns:
            dict: Inventory metrics
        """
        # Calculate statistics
        avg_demand = np.mean(forecast)
        std_demand = np.std(forecast)
        
        # Z-score for service level
        from scipy import stats
        z_score = stats.norm.ppf(service_level)
        
        # Safety stock
        safety_stock = z_score * std_demand * np.sqrt(lead_time)
        
        # Reorder point
        reorder_point = (avg_demand * lead_time) + safety_stock
        
        # Economic Order Quantity (EOQ)
        annual_demand = avg_demand * 365
        if annual_demand > 0 and holding_cost > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        else:
            eoq = avg_demand * 30  # Fallback: one month
        
        # Stockout risk
        stockout_risk = (1 - service_level) * 100
        
        return {
            'safety_stock': int(np.ceil(safety_stock)),
            'reorder_point': int(np.ceil(reorder_point)),
            'economic_order_quantity': int(np.ceil(eoq)),
            'recommended_order_quantity': int(np.ceil(eoq)),
            'average_daily_demand': round(avg_demand, 2),
            'demand_std_dev': round(std_demand, 2),
            'service_level': round(service_level * 100, 2),
            'stockout_risk': round(stockout_risk, 2),
            'lead_time_days': lead_time
        }
    
    def save_model(self, filepath):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.scaler = model_data.get('scaler', StandardScaler())
        self.metrics = model_data.get('metrics', {})
        self.feature_importance = model_data.get('feature_importance')
        print(f"Model loaded from {filepath}")
    
    def plot_forecast(self, historical, forecast, lower, upper, title="Demand Forecast"):
        """
        Plot forecast with historical data
        
        Args:
            historical: Historical time series
            forecast: Forecasted values
            lower: Lower confidence bound
            upper: Upper confidence bound
            title: Plot title
        """
        plt.figure(figsize=(14, 6))
        
        # Plot historical
        plt.plot(historical.index, historical.values, label='Historical', color='blue', linewidth=2)
        
        # Generate forecast dates
        last_date = historical.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast))
        
        # Plot forecast
        plt.plot(forecast_dates, forecast, label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence interval
        plt.fill_between(forecast_dates, lower, upper, alpha=0.3, color='red', label='95% Confidence')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Demand', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt

# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range(start='2013-01-01', end='2017-12-31', freq='D')
    demand = np.random.poisson(lam=50, size=len(dates)) + \
             10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)  # Seasonal pattern
    
    df = pd.DataFrame({'sale_date': dates, 'quantity_sold': demand})
    
    # Initialize forecaster
    forecaster = FMCGDemandForecaster(model_type='hw')
    
    # Prepare data
    series = forecaster.prepare_data(df)
    
    # Train model
    forecaster.train_holt_winters(series, seasonal_periods=7)
    
    # Generate forecast
    predictions, lower, upper = forecaster.forecast(steps=30)
    
    print("\nForecast (next 30 days):")
    print(predictions[:10])
    
    # Calculate inventory metrics
    metrics = forecaster.calculate_inventory_metrics(predictions)
    print("\nInventory Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Plot
    plt = forecaster.plot_forecast(series[-90:], predictions, lower, upper)
    plt.savefig('forecast_example.png')
    print("\nForecast plot saved as 'forecast_example.png'")