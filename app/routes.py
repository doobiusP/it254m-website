import yfinance as yf
from flask import Blueprint, render_template, request, jsonify
import matplotlib.pyplot as plt
import torch
from app.model_utils import load_or_train_model
from app.data_utils import add_moving_average_features, normalize_data, prepare_sequences
from app.plot_utils import plot_predictions, plot_backtesting

from sklearn.model_selection import train_test_split
import numpy as np

app_routes = Blueprint("app_routes", __name__)

BEST_MODEL_PATH = 'best_model.pth'
BEST_PARAMS_PATH = 'best_params.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ticker_data(ticker_symbol : str) -> yf.Ticker:
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        
        df = ticker_data.history(period='1d')
        if df.empty:
            raise ValueError(f"No data found for ticker symbol: {ticker_symbol}")
        
        return ticker_data
    except ValueError as e:
        return jsonify({"error": str(e), "message": "Invalid ticker symbol provided."}), 400
    
    except Exception as e:
        return jsonify({"error": str(e), "message": "Failed to retrieve ticker data. Please try again later."}), 500


@app_routes.route('/')
def home():

    symbols = ['AAPL', 'TSLA', 'MSFT']


    stock_data = {}

    for symbol in symbols:
        data = yf.download(symbol, period="6mo", interval="1d")  
        stock_data[symbol] = data
    def generate_stock_chart(data, symbol):
    
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Close'], label=f'{symbol} Closing Price')
        plt.title(f'{symbol} Stock Price (Last 6 Months)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.xticks(rotation=45)
        chart_filename = f"static/{symbol.lower()}_chart.png"
        plt.savefig(chart_filename)
        plt.close()

    for symbol, data in stock_data.items():
        generate_stock_chart(data, symbol)
    
    return render_template('home.html', 
                           apple_stock_data=stock_data['AAPL'],
                           tesla_stock_data=stock_data['TSLA'],
                           microsoft_stock_data=stock_data['MSFT'])

@app_routes.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app_routes.route('/predict', methods=['POST'])
def predict():
    ticker_symbol = request.form['ticker']
    recent_update_date = request.form['date']

    ticker_data = get_ticker_data(ticker_symbol)
    
    if isinstance(ticker_data, tuple):
        print("INTERNAL MESSAGE: Failed to find the Ticker within Yahoo! Finance API")
        return ticker_data
    
    df = ticker_data.history(period='1d', start='2010-01-01', end=recent_update_date)
    df = add_moving_average_features(df)
    data = df[['Close', 'MA_5', 'MA_10', 'MA_20']].values
    scaler, scaled_data = normalize_data(data)

    sequence_length = 60
    x, y = prepare_sequences(scaled_data, sequence_length)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    x_train, x_test = torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    model = load_or_train_model(x_train, y_train, x_test, y_test, BEST_MODEL_PATH, BEST_PARAMS_PATH, device)


    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test.to(device))

    predictions_reshaped = np.zeros((test_outputs.shape[0], 4))
    predictions_reshaped[:, 0] = test_outputs.cpu().numpy().reshape(-1) 

    predictions_rescaled = scaler.inverse_transform(predictions_reshaped)[:, 0] 

    plot_path = plot_predictions(df, predictions_rescaled)
    plot_path1 = plot_backtesting(df, predictions_rescaled)

    return render_template('prediction_result.html', plot_path=plot_path, plot_path1=plot_path1)
