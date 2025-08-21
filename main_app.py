import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from model import AdvancedSalesForecaster
from utils import create_custom_css, create_metric_card, create_forecast_visualization

# Set page config
st.set_page_config(
    page_title="🚀 Advanced Sales Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
create_custom_css()

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
    
    df = None
    
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
            try:
                df = pd.read_csv(uploaded_file)
                df['date'] = pd.to_datetime(df['date'])
                # Add required columns if not present
                if 'day_of_week' not in df.columns:
                    df['day_of_week'] = df['date'].dt.dayofweek
                if 'day_of_month' not in df.columns:
                    df['day_of_month'] = df['date'].dt.day
                if 'month' not in df.columns:
                    df['month'] = df['date'].dt.month
                if 'is_weekend' not in df.columns:
                    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
        else:
            st.warning("Please upload a CSV file with 'date' and 'sales' columns.")
            return
    
    if df is None:
        st.error("No data available. Please generate sample data or upload a CSV file.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Forecasting", 
        "🎯 Advanced Analytics"
    ])
    
    # TAB 1: Data Overview
    with tab1:
        data_overview_tab(df)
    
    # TAB 2: Model Training
    with tab2:
        model_training_tab(df, forecaster, test_size)
    
    # TAB 3: Forecasting
    with tab3:
        forecasting_tab(df, forecaster, forecast_days)
    
    # TAB 4: Advanced Analytics
    with tab4:
        advanced_analytics_tab(df)

def data_overview_tab(df):
    st.header("📊 Data Overview & Exploration")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Total Records", len(df))
    with col2:
        create_metric_card("Average Sales", f"${df['sales'].mean():,.0f}")
    with col3:
        create_metric_card("Max Sales", f"${df['sales'].max():,.0f}")
    with col4:
        create_metric_card("Sales Std", f"${df['sales'].std():,.0f}")

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

def model_training_tab(df, forecaster, test_size):
    st.header("🤖 Model Training & Evaluation")
    
    # Prepare data for modeling
    df_features = forecaster.create_features(df)
    if len(df_features) == 0:
        st.error("Not enough data after feature creation. Please use more data points.")
        return
        
    X, y = forecaster.prepare_data(df_features)
    
    # Train-test split
    split_idx = int(len(df_features) * (1 - test_size/100))
    if split_idx <= 0 or split_idx >= len(df_features):
        st.error("Invalid train-test split. Please adjust the test size.")
        return
        
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        st.error("Insufficient data for train-test split. Please use more data points or adjust test size.")
        return
    
    # Train models
    trained_models = forecaster.train_models(X_train, y_train)
    
    # Evaluate models
    model_results = {}
    predictions = {}
    
    for name, model in trained_models.items():
        try:
            y_pred = forecaster.predict(name, model, X_test)
            predictions[name] = y_pred
            model_results[name] = forecaster.calculate_metrics(y_test, y_pred)
        except Exception as e:
            st.error(f"Error with {name} model: {str(e)}")
            continue
    
    if not model_results:
        st.error("No models could be trained successfully.")
        return
    
    # Store in session state for other tabs
    st.session_state.trained_models = trained_models
    st.session_state.model_results = model_results
    st.session_state.predictions = predictions
    st.session_state.X = X
    st.session_state.y = y
    st.session_state.y_test = y_test
    st.session_state.df_features = df_features
    st.session_state.split_idx = split_idx
    
    # Display results
    st.subheader("Model Performance Comparison")
    
    metrics_df = pd.DataFrame(model_results).T
    st.dataframe(metrics_df.round(2), use_container_width=True)
    
    # Prediction vs Actual plot
    fig_pred = go.Figure()
    
    test_dates = df['date'].iloc[split_idx : split_idx + len(X_test)].values
    
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

def forecasting_tab(df, forecaster, forecast_days):
    st.header("📈 Sales Forecasting")
    
    if 'trained_models' not in st.session_state or 'model_results' not in st.session_state:
        st.error("Please train models first in the Model Training tab.")
        return
    
    trained_models = st.session_state.trained_models
    model_results = st.session_state.model_results
    predictions = st.session_state.predictions
    y_test = st.session_state.y_test
    
    # Select best model based on R²
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['R²'])
    st.success(f"🏆 Best performing model: **{best_model_name}** (R² = {model_results[best_model_name]['R²']:.3f})")
    
    # Generate future predictions
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    # Create future features
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
    prediction_std = np.std(y_test - predictions[best_model_name]) if best_model_name in predictions else None
    fig_forecast = create_forecast_visualization(df, future_dates, future_predictions, prediction_std, forecast_days)
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
    })
    
    if prediction_std:
        lower_bound = np.maximum(future_predictions - 1.96 * prediction_std, 0)
        upper_bound = future_predictions + 1.96 * prediction_std
        forecast_df['lower_bound'] = lower_bound
        forecast_df['upper_bound'] = upper_bound
    
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def advanced_analytics_tab(df):
    st.header("🎯 Advanced Analytics")
    
    if 'trained_models' not in st.session_state:
        st.error("Please train models first in the Model Training tab.")
        return
    
    trained_models = st.session_state.trained_models
    X = st.session_state.X
    df_features = st.session_state.df_features
    
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
    
    # Seasonal analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly trends
        df_copy = df.copy()
        df_copy['month_name'] = df_copy['date'].dt.month_name()
        monthly_avg = df_copy.groupby('month_name')['sales'].mean().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]).dropna()
        
        fig_monthly = px.line(
            x=monthly_avg.index, 
            y=monthly_avg.values,
            title="Average Sales by Month"
        )
        fig_monthly.update_traces(line_color='#764ba2', line_width=3, marker_size=8)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        # Sales volatility over time
        df_vol = df.copy()
        df_vol['sales_volatility'] = df_vol['sales'].rolling(window=30).std()
        fig_vol = px.line(
            df_vol, 
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