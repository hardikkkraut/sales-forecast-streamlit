# 🚀 Advanced Sales Forecaster

A comprehensive sales forecasting dashboard built with Streamlit that provides multi-model predictions, advanced analytics, and interactive visualizations for business intelligence.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📖 Overview

The Advanced Sales Forecaster is a powerful web application that enables businesses to:

- **Generate or upload sales data** for analysis
- **Train multiple machine learning models** (Random Forest, Linear Regression)
- **Create accurate sales forecasts** with confidence intervals
- **Perform advanced analytics** including feature importance and correlation analysis
- **Export predictions** in CSV format for business reporting

## ✨ Features

### 🔍 Data Overview & Exploration
- Interactive time series visualization
- Sales distribution analysis
- Day-of-week and monthly trend analysis
- Key performance metrics dashboard

### 🤖 Multi-Model Training
- **Random Forest Regressor**: Handles non-linear patterns and feature interactions
- **Linear Regression**: Provides baseline predictions with feature scaling
- Comprehensive model performance comparison (MAE, MSE, RMSE, R², MAPE)
- Visual comparison of predicted vs actual values

### 📈 Intelligent Forecasting
- Automatic best model selection based on performance metrics
- Configurable forecast periods (7-90 days)
- Confidence interval calculations
- Growth rate analysis compared to historical data

### 🎯 Advanced Analytics
- Feature importance analysis for model interpretability
- Seasonal decomposition and trend analysis
- Sales volatility tracking
- Correlation heatmaps for feature relationships

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/advanced-sales-forecaster.git
   cd advanced-sales-forecaster
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to `http://localhost:8501`

## 📊 Usage Guide

### Getting Started

1. **Configure Parameters**: Use the sidebar to adjust data generation and forecasting parameters
2. **Data Source**: Choose between generating sample data or uploading your own CSV file
3. **Navigate Tabs**: Explore the four main sections of the application

### Data Requirements

If uploading your own data, ensure your CSV file contains:
- `date`: Date column (YYYY-MM-DD format)
- `sales`: Numerical sales values

### Workflow

1. **Data Overview** → Explore and understand your sales data
2. **Model Training** → Train and evaluate machine learning models
3. **Forecasting** → Generate predictions for future periods
4. **Advanced Analytics** → Gain insights through feature analysis

## 🔧 Configuration Options

### Data Parameters
- **Historical Data Points**: Number of data points to generate (100-1000)
- **Trend Strength**: Controls the overall trend direction (0.0-3.0)
- **Seasonality Strength**: Adjusts seasonal variations (0.0-1.0)
- **Noise Level**: Adds realistic data noise (0.0-0.5)

### Forecast Parameters
- **Days to Forecast**: Prediction horizon (7-90 days)
- **Test Set Size**: Percentage of data for model validation (10-40%)

## 📁 Project Structure

```
advanced-sales-forecaster/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── data/                 # Sample data files (optional)
    └── sample_sales.csv
```

## 🧮 Technical Details

### Machine Learning Models

**Random Forest Regressor**
- Ensemble method using multiple decision trees
- Handles non-linear relationships and feature interactions
- Provides feature importance rankings
- Robust to outliers and noise

**Linear Regression**
- Linear relationship modeling with feature scaling
- Fast training and prediction
- Interpretable coefficients
- Good baseline model

### Feature Engineering

The application automatically creates advanced features:
- **Lag Features**: Previous day and week sales
- **Moving Averages**: 7-day and 30-day rolling means
- **Trend Components**: Linear trend modeling
- **Cyclical Features**: Sine/cosine transformations for seasonality
- **Calendar Features**: Day of week, month, weekend indicators

### Performance Metrics

- **MAE (Mean Absolute Error)**: Average prediction error magnitude
- **MSE (Mean Squared Error)**: Squared differences penalty
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals
- **R² (Coefficient of Determination)**: Proportion of variance explained
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric

## 🎨 Customization

### Styling
The application uses custom CSS for modern styling with:
- Gradient backgrounds and professional color schemes
- Interactive tab styling with hover effects
- Responsive metric cards with gradient backgrounds
- Custom Plotly chart themes

### Adding New Models
To add additional forecasting models:

1. Import the model class
2. Add to the `models` dictionary in `AdvancedSalesForecaster.__init__()`
3. Handle any special preprocessing in the `train_models()` method
4. Update the `predict()` method if needed

## 📈 Sample Output

The application generates:
- **Interactive Charts**: Time series, distributions, forecasts
- **Performance Tables**: Model comparison metrics
- **CSV Downloads**: Forecast results with confidence intervals
- **Analytics Dashboard**: Feature importance and correlation analysis

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- [ ] Additional ML models (ARIMA, Prophet, LSTM)
- [ ] Real-time data integration
- [ ] Multi-variate forecasting
- [ ] A/B testing framework
- [ ] API endpoints for model serving
- [ ] Database connectivity
- [ ] User authentication and data persistence

## 🐛 Known Issues

- Large datasets (>10,000 points) may experience slower performance
- CSV uploads require specific column naming convention
- Confidence intervals are simplified estimates

