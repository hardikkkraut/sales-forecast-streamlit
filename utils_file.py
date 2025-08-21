import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

def create_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
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
            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            color: white;
        }
        .metric-card h4 {
            color: #f0f0f0;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        .metric-card h2 {
            color: #ffffff;
            font-weight: 700;
            margin: 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1e1e2f;
            color: #bbbbbb;
            border-radius: 8px 8px 0px 0px;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #6a11cb !important;
            color: white !important;
            font-weight: bold;
            border-bottom: 3px solid #2575fc;
        }
    </style>
    """, unsafe_allow_html=True)

def create_metric_card(title, value):
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

def create_forecast_visualization(df, future_dates, future_predictions, prediction_std=None, forecast_days=30):
    """
    Create a forecast visualization with confidence intervals
    
    Args:
        df: Historical data DataFrame
        future_dates: Future date range
        future_predictions: Predicted values
        prediction_std: Standard deviation for confidence intervals
        forecast_days: Number of days to forecast
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_dates, 
        y=future_predictions,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#F18F01', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add confidence intervals if available
    if prediction_std is not None:
        upper_bound = future_predictions + 1.96 * prediction_std
        lower_bound = np.maximum(future_predictions - 1.96 * prediction_std, 0)
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(241, 143, 1, 0.2)'
        ))
    
    fig.update_layout(
        title=f"Sales Forecast - Next {forecast_days} Days",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=500,
        showlegend=True
    )
    
    return fig

def load_and_validate_csv(uploaded_file):
    """
    Load and validate CSV file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame or None if invalid
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check required columns
        if 'date' not in df.columns or 'sales' not in df.columns:
            st.error("CSV must contain 'date' and 'sales' columns")
            return None
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Add missing columns if not present
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
        if 'day_of_month' not in df.columns:
            df['day_of_month'] = df['date'].dt.day
        if 'month' not in df.columns:
            df['month'] = df['date'].dt.month
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
        
        return df
        
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        return None

def create_time_series_plot(df, title="Sales Time Series"):
    """Create a time series plot of sales data"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'], 
        y=df['sales'],
        mode='lines',
        name='Sales',
        line=dict(color='#667eea', width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=400,
        showlegend=True
    )
    return fig

def create_distribution_plot(df, column='sales', title="Sales Distribution"):
    """Create a histogram of sales distribution"""
    fig = px.histogram(df, x=column, nbins=30, title=title)
    fig.update_traces(marker_color='#667eea')
    return fig

def create_seasonal_analysis(df):
    """Create seasonal analysis plots"""
    # Day of week analysis
    dow_sales = df.groupby('day_of_week')['sales'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig_dow = px.bar(
        x=days, 
        y=dow_sales.values, 
        title="Average Sales by Day of Week"
    )
    fig_dow.update_traces(marker_color='#764ba2')
    
    # Monthly analysis
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
    
    return fig_dow, fig_monthly

def create_volatility_plot(df):
    """Create sales volatility plot"""
    df_vol = df.copy()
    df_vol['sales_volatility'] = df_vol['sales'].rolling(window=30).std()
    
    fig = px.line(
        df_vol, 
        x='date', 
        y='sales_volatility',
        title="Sales Volatility (30-day Rolling Std)"
    )
    fig.update_traces(line_color='#A23B72', line_width=2)
    return fig

def create_correlation_heatmap(df_features):
    """Create correlation heatmap for features"""
    corr_matrix = df_features.select_dtypes(include=[np.number]).corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    return fig

def create_feature_importance_plot(importance_df, top_n=10):
    """Create feature importance bar plot"""
    fig = px.bar(
        importance_df.head(top_n), 
        x='importance', 
        y='feature',
        orientation='h',
        title=f"Top {top_n} Feature Importance"
    )
    fig.update_traces(marker_color='#667eea')
    return fig

def calculate_growth_metrics(current_data, forecast_data):
    """Calculate various growth metrics"""
    current_avg = current_data.mean()
    forecast_avg = forecast_data.mean()
    
    growth_rate = ((forecast_avg / current_avg) - 1) * 100
    total_current = current_data.sum()
    total_forecast = forecast_data.sum()
    
    return {
        'growth_rate': growth_rate,
        'current_avg': current_avg,
        'forecast_avg': forecast_avg,
        'total_current': total_current,
        'total_forecast': total_forecast
    }

def export_forecast_to_csv(dates, predictions, confidence_intervals=None):
    """Export forecast data to CSV format"""
    forecast_df = pd.DataFrame({
        'date': dates,
        'predicted_sales': predictions,
    })
    
    if confidence_intervals is not None:
        forecast_df['lower_bound'] = confidence_intervals[0]
        forecast_df['upper_bound'] = confidence_intervals[1]
    
    return forecast_df.to_csv(index=False)

def format_currency(value):
    """Format number as currency"""
    return f"${value:,.0f}"

def format_percentage(value, decimals=1):
    """Format number as percentage"""
    return f"{value:+.{decimals}f}%"

def validate_forecast_parameters(forecast_days, test_size, data_length):
    """Validate forecasting parameters"""
    errors = []
    
    if forecast_days < 1 or forecast_days > 365:
        errors.append("Forecast days must be between 1 and 365")
    
    if test_size < 5 or test_size > 50:
        errors.append("Test size must be between 5% and 50%")
    
    if data_length < 50:
        errors.append("Need at least 50 data points for reliable forecasting")
    
    train_size = int(data_length * (1 - test_size/100))
    if train_size < 30:
        errors.append("Training set too small. Reduce test size or increase data points")
    
    return errors

def create_model_comparison_plot(y_test, predictions, test_dates):
    """Create model comparison plot"""
    fig = go.Figure()
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=test_dates, 
        y=y_test,
        mode='lines+markers',
        name='Actual',
        line=dict(color='#2E86AB', width=3)
    ))
    
    # Model predictions
    colors = ['#A23B72', '#F18F01', '#C73E1D']
    for i, (name, y_pred) in enumerate(predictions.items()):
        fig.add_trace(go.Scatter(
            x=test_dates, 
            y=y_pred,
            mode='lines+markers',
            name=f'{name} Prediction',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Sales",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        height=500,
        showlegend=True
    )
    
    return fig