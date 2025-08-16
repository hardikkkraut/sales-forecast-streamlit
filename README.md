# 📈 Sales Forecast App

A machine learning project to **predict future sales using past data**.  
This project demonstrates data preprocessing, feature engineering (time-based lags, rolling averages), training a regression model, and evaluating it against a naive baseline.  
Optionally, it includes a **Streamlit web app** for interactive forecasting.

---

## 🚀 Features
- Exploratory Data Analysis (EDA) with sales trends visualization.
- Feature engineering: date parts, lags, rolling averages, categorical encoding.
- Ridge Regression baseline model (easy to extend with Random Forest / XGBoost).
- Metrics: MAE & RMSE, benchmarked against a naive forecast.
- Streamlit app for uploading CSVs and visualizing predictions.
- Clean, modular structure for easy extension.

---

## 📂 Project Structure
```
sales-forecast-streamlit/
├─ data/                # sample CSV (replace with your data)
├─ notebooks/           # for EDA experiments (optional)
├─ src/
│  ├─ make_features.py  # feature engineering
│  ├─ train.py          # training script
│  └─ infer.py          # (optional) inference utilities
├─ app.py               # Streamlit app
├─ requirements.txt     # dependencies
└─ README.md            # project documentation
```

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/sales-forecast-streamlit.git
cd sales-forecast-streamlit
pip install -r requirements.txt
```

---

## 📊 Usage

### 1) Train Model
```bash
python src/train.py
```
This will preprocess data, train a Ridge regression model, save the model (`model.joblib`), and save the feature list (`features_used.csv`).

### 2) Run Streamlit App (Optional)
```bash
streamlit run app.py
```

Upload your own sales CSV (must include at least `date` and `sales` columns). The app will display predictions and comparison charts.

---

## 📈 Example Results
- Time-series sales trends plotted with Matplotlib.
- Actual vs Predicted sales chart for the test set.
- MAE & RMSE values reported for model evaluation.

---

## 🔮 Next Steps
- Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Extend to Random Forest, XGBoost, or Neural Networks.
- Add holiday/seasonality features for better forecasts.
- Deploy the Streamlit app to Streamlit Community Cloud.

---

## 📝 License
This project is open-source and available under the MIT License.

---

## 👨‍💻 Author
Developed by [Hardik Raut ](https://www.linkedin.com/in/your-profile/)  
Feel free to connect and share feedback! 🚀
