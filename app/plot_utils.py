import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive backend
matplotlib.use("Agg")

def plot_predictions(df : pd.DataFrame, predictions_rescaled : np.ndarray) -> str:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(predictions_rescaled):], df['Close'].values[-len(predictions_rescaled):], label='Actual Prices')
    plt.plot(df.index[-len(predictions_rescaled):], predictions_rescaled, label='Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plot_path = 'static/prediction_plot.png'
    plt.savefig(plot_path, format='png')
    plt.close() 
    return plot_path

def plot_backtesting(df : pd.DataFrame, predictions_rescaled : np.ndarray) -> str:
    signals = np.diff(predictions_rescaled) > 0
    signals = np.insert(signals, 0, False)

    actual = df['Close'].values[-len(predictions_rescaled):]
    daily_returns = np.insert(np.diff(actual) / actual[:-1], 0, 0)
    strategy_returns = signals[:-1] * daily_returns[1:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(daily_returns), label='Stock Returns (Buy and Hold)')
    plt.plot(np.cumsum(strategy_returns), label='Strategy Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    
    plot_path = 'static/backtesting_plot.png'
    plt.savefig(plot_path, format='png')
    plt.close()  # Close the figure to free memory
    return plot_path
