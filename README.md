# Stock Price Prediction with LSTM Models

This repository contains a Flask web application for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The application fetches historical stock data from Yahoo Finance API (`yfinance`), preprocesses the data by adding moving average features, normalizes it using Min-Max scaling, and trains LSTM models with different hyperparameters.

## Key Features

- **Data Preprocessing:** Includes adding moving average features to enhance model performance.
- **Model Training:** Utilizes LSTM models with varying architectures (layers, hidden dimensions, dropout) to predict stock prices.
- **Hyperparameter Tuning:** Grid search over predefined parameters to optimize model performance.
- **Web Application:** Built using Flask to allow users to input a stock ticker symbol and a date, then receive predictions for future stock prices.
- **Visualization:** Generates plots using `matplotlib` to visualize predicted prices and backtesting results.
- **Backtesting Strategy:** Evaluates a simple trading strategy based on predicted signals.

## Technologies Used

- Python
- Flask
- PyTorch (for LSTM models)
- `yfinance` (Yahoo Finance API)
- `matplotlib` (for plotting)
- `sklearn` (for data preprocessing)

## How to Use

1. Clone the repo
2. Create a virtualenv with `py -m venv virtualenv`
3. `virtualenv\Scripts\activate`
4. `pip install json5 flask numpy matplotlib yfinance scikit-learn torch`
5. run `py app.py`
6. Navigate to http://localhost:5001 in your web browser.
7. Enter a stock ticker symbol and a recent date to get predictions and backtesting results.

## Screeshots
![image](https://github.com/ChellaVigneshKP/stock-prediction/assets/97314418/964acfc0-5e28-4cee-b7df-682af6996665)
![image](https://github.com/ChellaVigneshKP/stock-prediction/assets/97314418/95f5e097-733f-4941-b4e6-cfc4eae74d35)


