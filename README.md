# 🚀 Advanced Sales Forecaster

A comprehensive sales forecasting application built with Streamlit that provides multi-model predictions, advanced analytics, and interactive visualizations.

## 📁 Project Structure

```
sales-forecaster/
│
├── app.py                 # Main Streamlit application
├── model.py              # Machine learning models and training functions
├── utils.py              # Helper functions for visualization and data processing
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
│
├── data/                # Data directory
│   └── sample_sales.csv # Sample dataset for testing
│
└── forecasts/           # Output directory (created automatically)
    └── *.csv           # Generated forecast files
```

## 🌟 Features

### 📊 Data Overview
- Interactive time series visualization
- Sales distribution analysis
- Seasonal patterns (daily, weekly, monthly)
- Key statistical metrics

### 🤖 Model Training
- **Random Forest Regressor**: Tree-based ensemble model
- **Linear Regression**: Linear relationship modeling
- Automatic feature engineering
- Model performance comparison
- Cross-validation metrics (MAE, MSE, RMSE, R², MAPE)

### 📈 Forecasting
- Multi-day forecasting (7-90 days)
- Confidence intervals
- Best model auto-selection
- Interactive forecast visualization
- Downloadable predictions

### 🎯 Advanced Analytics
- Feature importance analysis
- Seasonal decomposition
- Sales volatility tracking
- Correlation analysis
- Custom business metrics

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd sales-forecaster

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### 3. Using the App

1. **Configure Parameters**: Use the sidebar to adjust data generation and forecasting parameters
2. **Choose Data Source**: Either generate sample data or upload your own CSV
3. **Explore Data**: View data overview and patterns in the first tab
4. **Train Models**: Run model training and compare performance
5. **Generate Forecasts**: Create future predictions with confidence intervals
6. **Analyze Results**: Dive deep with advanced analytics and feature importance

## 📊 Data Requirements

### For CSV Upload
Your CSV file should contain at minimum:
- `date`: Date column (YYYY-MM-DD format)
- `sales`: Sales values (numeric)

Example:
```csv
date,sales
2023-01-01,1250.50
2023-01-02,1180.75
2023-01-03,1320.25
```

Optional columns (auto-generated if missing):
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `day_of_month`: Day of month (1-31)
- `month`: Month (1-12)
- `is_weekend`: Weekend indicator (0/1)

## 🔧 Customization

### Adding New Models

To add a new model, modify `model.py`:

```python
# In the __init__ method of AdvancedSalesForecaster
self.models['Your Model Name'] = YourModelClass()

# Add any special handling in train_models() and predict() methods
```

### Custom Features

Add new features in the `create_features()` method:

```python
def create_features(self, df):
    # Existing features...
    
    # Add your custom features
    df['your_feature'] = your_calculation
    
    return df.dropna()
```

### Styling

Modify the CSS in `utils.py` `create_custom_css()` function to change the app appearance.

## 📈 Model Details

### Feature Engineering
- **Lag Features**: sales_lag_1, sales_lag_7
- **Moving Averages**: 7-day and 30-day rolling means
- **Trend**: Linear trend component
- **Cyclical**: Sine/cosine transformations for temporal patterns
- **Calendar**: Day of week, month, weekend indicators

### Model Algorithms

1. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Provides feature importance
   - Robust to outliers

2. **Linear Regression**
   - Linear relationship modeling
   - Feature scaling applied
   - Fast training and prediction
   - Good baseline model

## 🎛️ Configuration Options

### Sidebar Parameters

- **Historical Data Points**: 100-1000 (for generated data)
- **Trend Strength**: 0.0-3.0
- **Seasonality Strength**: 0.0-1.0
- **Noise Level**: 0.0-0.5
- **Forecast Days**: 7-90
- **Test Set Size**: 10-40%

## 📁 Output Files

Forecasts are downloadable as CSV files containing:
- `date`: Future dates
- `predicted_sales`: Predicted values
- `lower_bound`: Lower confidence interval (if available)
- `upper_bound`: Upper confidence interval (if available)

## 🐛 Troubleshooting

### Common Issues

1. **"Not enough data after feature creation"**
   - Solution: Increase historical data points or use a larger dataset

2. **"Invalid train-test split"**
   - Solution: Adjust test size percentage or use more data

3. **Model training errors**
   - Solution: Check data quality and ensure no missing values

4. **CSV upload issues**
   - Solution: Ensure proper date format and required columns

### Performance Tips

- Use 200+ data points for better accuracy
- Keep test size between 15-25%
- Monitor model R² scores (>0.7 is good)
- Validate forecasts against business knowledge

## 🔄 Updates and Maintenance

### Adding New Features
1. Update model training in `model.py`
2. Add visualization functions in `utils.py`
3. Update the main app interface in `app.py`
4. Test with sample data

### Dependencies
- Keep dependencies updated for security
- Test after major version updates
- Pin versions for production deployment

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Validate your data format
3. Review error messages in the app
4. Check console output when running locally

---

## 🏆 Best Practices

1. **Data Quality**: Ensure clean, consistent data
2. **Model Selection**: Compare multiple models
3. **Validation**: Use appropriate test set sizes
4. **Interpretation**: Consider business context
5. **Monitoring**: Track forecast accuracy over time

Happy forecasting! 🎯