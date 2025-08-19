import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="🚀 Advanced Sales Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px;
        color: #262730;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedSalesForecaster:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
        }
        self.scaler = StandardScaler()
        
    def generate_sample_data(self, start_date, periods, trend_strength=1.0, seasonality_strength=0.3, noise_level=0.1):
        """Generate realistic sample sales data with trend, seasonality, and noise"""
        dates = pd.date_range(start=start_date, periods=periods, freq='D')
        
        # Base trend
        trend = np.linspace(1000, 1000 + trend_strength * 500, periods)
        
        # Seasonal patterns (weekly and monthly)
        weekly_season = 200 * np.sin(2 * np.pi * np.arange(periods) / 7) * seasonality_strength
        monthly_season = 300 * np.sin(2 * np.pi * np.arange(periods) / 30) * seasonality_strength
        
        # Holiday effects (random spikes)
        holiday_effects = np.random.choice([0, 0, 0, 0, 500], periods, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        
        # Noise
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
        """Create additional features for modeling"""
        df = df.copy()
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_7'] = df['sales'].shift(7)
        df['sales_ma_7'] = df['sales'].rolling(window=7).mean()
        df['sales_ma_30'] = df['sales'].rolling(window=30).mean()
        
        # Trend feature
        df['trend'] = np.arange(len(df))
        
        # Cyclical features
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_month'] / 30)
        df['sin_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df.dropna()
    
    def prepare_data(self, df, target_col='sales'):
        """Prepare data for modeling"""
        feature_cols = [col for col in df.columns if col not in [target_col, 'date']]
        X = df[feature_cols]
        y = df[target_col]
        return X, y
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        trained_models = {}
        for name, model in self.models.items():
            if name == 'Linear Regression':
                X_scaled = self.scaler.fit_transform(X_train)
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models
    
    def predict(self, model_name, model, X_test):
        """Make predictions"""
        if model_name == 'Linear Regression':
            X_scaled = self.scaler.transform(X_test)
            return model.predict(X_scaled)
        else:
            return model.predict(X_test)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Advanced Sales Forecaster</h1>
        <p>Multi-Model Sales Prediction with Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize forecaster
    forecaster = AdvancedSalesForecaster()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Parameters")
    data_periods = st.sidebar.slider("Historical Data Points", 100, 1000, 365)
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 3.0, 1.5, 0.1)
    seasonality_strength = st.sidebar.slider("Seasonality Strength", 0.0, 1.0, 0.4, 0.1)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.15, 0.05)
    
    # Forecasting parameters
    st.sidebar.subheader("Forecast Parameters")
    forecast_days = st.sidebar.slider("Days to Forecast", 7, 90, 30)
    test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20)
    
    # Generate or upload data
    data_source = st.sidebar.radio("Data Source", ["Generate Sample Data", "Upload CSV"])
    
    if data_source == "Generate Sample Data":
        start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=data_periods))
        
        # Generate data
        if st.sidebar.button("🔄 Generate New Data"):
            st.session_state.data_generated = True
        
        if 'data_generated' not in st.session_state:
            st.session_state.data_generated = True
        
        if st.session_state.data_generated:
            df = forecaster.generate_sample_data(
                start_date, data_periods, trend_strength, seasonality_strength, noise_level
            )
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
        else:
            st.warning("Please upload a CSV file with 'date' and 'sales' columns.")
            return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Training", "📈 Forecasting", "🎯 Advanced Analytics"])
    
    with tab1:
        st.header("📊 Data Overview & Exploration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>Total Records</h4>
                <h2>{}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>Average Sales</h4>
                <h2>${:,.0f}</h2>
            </div>
            """.format(df['sales'].mean()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>Max Sales</h4>
                <h2>${:,.0f}</h2>
            </div>
            """.format(df['sales'].max()), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>Sales Std</h4>
                <h2>${:,.0f}</h2>
            </div>
            """.format(df['sales'].std()), unsafe_allow_html=True)
        
        # Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['sales'],
            mode='lines',
            name='Sales',
            line=dict(color='#667eea', width=2)
        ))
        fig.update_layout(
            title="Sales Time Series",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution and seasonality analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Sales distribution
            fig_hist = px.histogram(df, x='sales', nbins=30, title="Sales Distribution")
            fig_hist.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Day of week analysis
            dow_sales = df.groupby('day_of_week')['sales'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig_dow = px.bar(x=days, y=dow_sales.values, title="Average Sales by Day of Week")
            fig_dow.update_traces(marker_color='#764ba2')
            st.plotly_chart(fig_dow, use_container_width=True)
    
    with tab2:
        st.header("🤖 Model Training & Evaluation")
        
        # Prepare data for modeling
        df_features = forecaster.create_features(df)
        X, y = forecaster.prepare_data(df_features)
        
        # Train-test split
        split_idx = int(len(df_features) * (1 - test_size/100))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        trained_models = forecaster.train_models(X_train, y_train)
        
        # Evaluate models
        model_results = {}
        predictions = {}
        
        for name, model in trained_models.items():
            y_pred = forecaster.predict(name, model, X_test)
            predictions[name] = y_pred
            model_results[name] = forecaster.calculate_metrics(y_test, y_pred)
        
        # Display results
        st.subheader("Model Performance Comparison")
        
        metrics_df = pd.DataFrame(model_results).T
        st.dataframe(metrics_df.round(2), use_container_width=True)
        
        # Prediction vs Actual plot
        fig_pred = go.Figure()
        
        test_dates = df['date'][split_idx:].values
        
        fig_pred.add_trace(go.Scatter(
            x=test_dates, 
            y=y_test,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#2E86AB', width=3)
        ))
        
        colors = ['#A23B72', '#F18F01']
        for i, (name, y_pred) in enumerate(predictions.items()):
            fig_pred.add_trace(go.Scatter(
                x=test_dates, 
                y=y_pred,
                mode='lines+markers',
                name=f'{name} Prediction',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig_pred.update_layout(
            title="Model Predictions vs Actual Sales",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with tab3:
        st.header("📈 Sales Forecasting")
        
        # Select best model based on R²
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['R²'])
        st.success(f"🏆 Best performing model: **{best_model_name}** (R² = {model_results[best_model_name]['R²']:.3f})")
        
        # Generate future predictions
        last_date = df['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
        
        # Create future features (simplified)
        future_df = pd.DataFrame({
            'date': future_dates,
            'day_of_week': future_dates.dayofweek,
            'day_of_month': future_dates.day,
            'month': future_dates.month,
            'is_weekend': (future_dates.dayofweek >= 5).astype(int)
        })
        
        # Add trend and cyclical features
        future_df['trend'] = np.arange(len(df), len(df) + forecast_days)
        future_df['sin_day'] = np.sin(2 * np.pi * future_df['day_of_month'] / 30)
        future_df['cos_day'] = np.cos(2 * np.pi * future_df['day_of_month'] / 30)
        future_df['sin_week'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['cos_week'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        
        # Use last known values for lag features
        last_sales = df['sales'].iloc[-1]
        last_ma_7 = df['sales'].tail(7).mean()
        last_ma_30 = df['sales'].tail(30).mean()
        
        future_df['sales_lag_1'] = last_sales
        future_df['sales_lag_7'] = last_sales
        future_df['sales_ma_7'] = last_ma_7
        future_df['sales_ma_30'] = last_ma_30
        
        # Make predictions
        best_model = trained_models[best_model_name]
        future_X = future_df.drop(['date'], axis=1)
        future_predictions = forecaster.predict(best_model_name, best_model, future_X)
        
        # Create forecast visualization
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=df['date'], 
            y=df['sales'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Future predictions
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#F18F01', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Add confidence intervals (simplified)
        prediction_std = np.std(y_test - predictions[best_model_name])
        upper_bound = future_predictions + 1.96 * prediction_std
        lower_bound = future_predictions - 1.96 * prediction_std
        
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(241, 143, 1, 0.2)'
        ))
        
        fig_forecast.update_layout(
            title=f"Sales Forecast - Next {forecast_days} Days",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Forecast Period", f"{forecast_days} days")
        with col2:
            st.metric("Predicted Avg Sales", f"${future_predictions.mean():,.0f}")
        with col3:
            st.metric("Total Forecast", f"${future_predictions.sum():,.0f}")
        with col4:
            growth_rate = ((future_predictions.mean() / df['sales'].tail(30).mean()) - 1) * 100
            st.metric("Growth vs Last 30d", f"{growth_rate:+.1f}%")
        
        # Download forecast
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': future_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
        
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.header("🎯 Advanced Analytics")
        
        # Feature importance (for Random Forest)
        if 'Random Forest' in trained_models:
            rf_model = trained_models['Random Forest']
            feature_names = X.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig_imp = px.bar(
                importance_df.head(10), 
                x='importance', 
                y='feature',
                orientation='h',
                title="Top 10 Feature Importance (Random Forest)"
            )
            fig_imp.update_traces(marker_color='#667eea')
            st.plotly_chart(fig_imp, use_container_width=True)
        
        # Seasonal decomposition visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly trends
            df['month_name'] = df['date'].dt.month_name()
            monthly_avg = df.groupby('month_name')['sales'].mean().reindex([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])
            
            fig_monthly = px.line(
                x=monthly_avg.index, 
                y=monthly_avg.values,
                title="Average Sales by Month"
            )
            fig_monthly.update_traces(line_color='#764ba2', line_width=3, marker_size=8)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            # Sales volatility over time
            df['sales_volatility'] = df['sales'].rolling(window=30).std()
            fig_vol = px.line(
                df, 
                x='date', 
                y='sales_volatility',
                title="Sales Volatility (30-day Rolling Std)"
            )
            fig_vol.update_traces(line_color='#A23B72', line_width=2)
            st.plotly_chart(fig_vol, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation Analysis")
        corr_matrix = df_features.select_dtypes(include=[np.number]).corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

if __name__ == "__main__":
    main()