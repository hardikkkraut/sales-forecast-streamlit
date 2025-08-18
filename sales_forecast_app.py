import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Sales Forecast Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("🎛️ Controls")

# Generate sample data function
@st.cache_data
def generate_sample_data():
    """Generate realistic sales data with trends and seasonality"""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    n_months = len(dates)
    
    # Base trend
    trend = np.linspace(10000, 25000, n_months)
    
    # Seasonal pattern (higher sales in Nov-Dec, lower in Jan-Feb)
    seasonal = 3000 * np.sin(2 * np.pi * np.arange(n_months) / 12 + np.pi/2)
    
    # Random noise
    noise = np.random.normal(0, 1500, n_months)
    
    # Combine components
    sales = trend + seasonal + noise
    sales = np.maximum(sales, 1000)  # Ensure positive values
    
    df = pd.DataFrame({
        'Date': dates,
        'Sales': sales.round(2),
        'Month': dates.month,
        'Year': dates.year,
        'Quarter': dates.quarter
    })
    
    # Add additional features
    df['Sales_Lag1'] = df['Sales'].shift(1)
    df['Sales_Lag3'] = df['Sales'].shift(3)
    df['Sales_MA3'] = df['Sales'].rolling(window=3).mean()
    df['Sales_MA6'] = df['Sales'].rolling(window=6).mean()
    
    return df

# Data source selection
data_source = st.sidebar.selectbox(
    "Select Data Source:",
    ["Use Sample Data", "Upload CSV File"]
)

# Load data
if data_source == "Use Sample Data":
    df = generate_sample_data()
    st.sidebar.success("✅ Sample data loaded successfully!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            df = generate_sample_data()
    else:
        df = generate_sample_data()
        st.sidebar.info("Using sample data. Upload a CSV to use your own data.")

# Forecast parameters
st.sidebar.subheader("🔮 Forecast Settings")
forecast_months = st.sidebar.slider("Months to Forecast:", 1, 24, 6)
model_type = st.sidebar.selectbox(
    "Select Model:",
    ["Linear Regression", "Random Forest", "Moving Average", "Exponential Smoothing"]
)

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualization", "🔮 Forecast", "📋 Model Performance"])

with tab1:
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=len(df),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Date Range",
            value=f"{df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Average Monthly Sales",
            value=f"${df['Sales'].mean():,.0f}",
            delta=f"{((df['Sales'].iloc[-6:].mean() / df['Sales'].iloc[-12:-6].mean() - 1) * 100):+.1f}%"
        )
    
    with col4:
        st.metric(
            label="Total Sales",
            value=f"${df['Sales'].sum():,.0f}",
            delta=None
        )
    
    st.subheader("📋 Raw Data")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

with tab2:
    st.header("Data Visualization")
    
    # Time series plot
    fig_ts = px.line(df, x='Date', y='Sales', 
                     title='Sales Over Time',
                     labels={'Sales': 'Sales ($)', 'Date': 'Date'})
    fig_ts.update_layout(height=400)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly seasonality
        monthly_avg = df.groupby('Month')['Sales'].mean().reset_index()
        fig_month = px.bar(monthly_avg, x='Month', y='Sales',
                          title='Average Sales by Month')
        st.plotly_chart(fig_month, use_container_width=True)
    
    with col2:
        # Yearly trend
        yearly_total = df.groupby('Year')['Sales'].sum().reset_index()
        fig_year = px.line(yearly_total, x='Year', y='Sales',
                          title='Total Sales by Year', markers=True)
        st.plotly_chart(fig_year, use_container_width=True)
    
    # Sales distribution
    fig_dist = px.histogram(df, x='Sales', nbins=30,
                           title='Sales Distribution')
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.header("Sales Forecast")
    
    # Prepare data for modeling
    def prepare_features(data):
        """Prepare features for modeling"""
        features_df = data.copy()
        features_df['Month_sin'] = np.sin(2 * np.pi * features_df['Month'] / 12)
        features_df['Month_cos'] = np.cos(2 * np.pi * features_df['Month'] / 12)
        features_df['Time_index'] = range(len(features_df))
        
        feature_cols = ['Month_sin', 'Month_cos', 'Time_index']
        if 'Sales_Lag1' in features_df.columns:
            feature_cols.extend(['Sales_Lag1', 'Sales_MA3'])
        
        return features_df[feature_cols].fillna(method='bfill')
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size].copy()
    test_data = df[train_size:].copy()
    
    # Prepare features
    X_train = prepare_features(train_data)
    y_train = train_data['Sales'].values
    X_test = prepare_features(test_data)
    y_test = test_data['Sales'].values
    
    # Model training and prediction
    if model_type == "Linear Regression":
        model = LinearRegression()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        test_pred = model.predict(X_test_scaled)
        
        # Future predictions
        future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), 
                                   periods=forecast_months, freq='M')
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
        future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
        future_df['Time_index'] = range(len(df), len(df) + forecast_months)
        future_df['Sales_Lag1'] = df['Sales'].iloc[-1]
        future_df['Sales_MA3'] = df['Sales'].iloc[-3:].mean()
        
        X_future = prepare_features(future_df)
        X_future_scaled = scaler.transform(X_future)
        future_pred = model.predict(X_future_scaled)
        
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        
        # Future predictions
        future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), 
                                   periods=forecast_months, freq='M')
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Month_sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
        future_df['Month_cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
        future_df['Time_index'] = range(len(df), len(df) + forecast_months)
        future_df['Sales_Lag1'] = df['Sales'].iloc[-1]
        future_df['Sales_MA3'] = df['Sales'].iloc[-3:].mean()
        
        X_future = prepare_features(future_df)
        future_pred = model.predict(X_future)
        
    elif model_type == "Moving Average":
        window = min(6, len(train_data))
        test_pred = []
        for i in range(len(test_data)):
            if i == 0:
                pred = train_data['Sales'].iloc[-window:].mean()
            else:
                recent_data = list(train_data['Sales'].iloc[-window:]) + test_pred[:i]
                pred = np.mean(recent_data[-window:])
            test_pred.append(pred)
        
        # Future predictions
        future_pred = []
        recent_sales = list(df['Sales'].iloc[-window:])
        for i in range(forecast_months):
            pred = np.mean(recent_sales[-window:])
            future_pred.append(pred)
            recent_sales.append(pred)
        
        future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), 
                                   periods=forecast_months, freq='M')
        
    else:  # Exponential Smoothing
        alpha = 0.3
        test_pred = []
        last_smooth = train_data['Sales'].mean()
        
        for i in range(len(test_data)):
            if i == 0:
                pred = last_smooth
            else:
                last_smooth = alpha * test_data['Sales'].iloc[i-1] + (1-alpha) * last_smooth
                pred = last_smooth
            test_pred.append(pred)
        
        # Future predictions
        future_pred = [last_smooth] * forecast_months
        future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), 
                                   periods=forecast_months, freq='M')
    
    # Create forecast visualization
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df['Date'], y=df['Sales'],
        mode='lines', name='Historical Sales',
        line=dict(color='blue', width=2)
    ))
    
    # Test predictions
    fig_forecast.add_trace(go.Scatter(
        x=test_data['Date'], y=test_pred,
        mode='lines', name='Test Predictions',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Future predictions
    fig_forecast.add_trace(go.Scatter(
        x=future_dates, y=future_pred,
        mode='lines+markers', name='Future Forecast',
        line=dict(color='red', width=3),
        marker=dict(size=6)
    ))
    
    fig_forecast.update_layout(
        title=f'Sales Forecast - {model_type}',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast summary
    st.subheader("📊 Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Average Monthly Sales",
            value=f"${np.mean(future_pred):,.0f}",
            delta=f"{((np.mean(future_pred) / df['Sales'].mean() - 1) * 100):+.1f}%"
        )
    
    with col2:
        st.metric(
            label="Total Predicted Sales",
            value=f"${np.sum(future_pred):,.0f}",
            delta=None
        )
    
    with col3:
        growth_rate = ((future_pred[-1] / df['Sales'].iloc[-1] - 1) * 100) / forecast_months * 12
        st.metric(
            label="Annualized Growth Rate",
            value=f"{growth_rate:+.1f}%",
            delta=None
        )
    
    # Detailed forecast table
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Sales': [f"${x:,.0f}" for x in future_pred]
    })
    
    st.subheader("🗓️ Detailed Forecast")
    st.dataframe(forecast_df, use_container_width=True)

with tab4:
    st.header("Model Performance")
    
    if len(test_data) > 0:
        # Calculate metrics
        mae = mean_absolute_error(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, test_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Absolute Error", f"${mae:,.0f}")
        with col2:
            st.metric("Root Mean Squared Error", f"${rmse:,.0f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        with col4:
            mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
            st.metric("MAPE", f"{mape:.1f}%")
        
        # Prediction vs Actual plot
        fig_performance = go.Figure()
        fig_performance.add_trace(go.Scatter(
            x=test_data['Date'], y=y_test,
            mode='lines+markers', name='Actual',
            line=dict(color='blue', width=2)
        ))
        fig_performance.add_trace(go.Scatter(
            x=test_data['Date'], y=test_pred,
            mode='lines+markers', name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_performance.update_layout(
            title='Actual vs Predicted Sales (Test Set)',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            height=400
        )
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Residuals plot
        residuals = y_test - test_pred
        fig_residuals = px.scatter(x=test_pred, y=residuals,
                                 title='Residuals Plot',
                                 labels={'x': 'Predicted Sales', 'y': 'Residuals'})
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)
        
    else:
        st.warning("No test data available for performance evaluation. Consider using more historical data.")

# Footer
st.markdown("---")
st.markdown("**Sales Forecast Dashboard** - Built with Streamlit and Plotly")
st.markdown("💡 **Tips:** Use the sidebar to adjust forecast parameters and try different models for comparison.")