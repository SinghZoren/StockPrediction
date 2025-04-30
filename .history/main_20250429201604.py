# main.py
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from linear_regression import LinearRegression
from utils import fetch_stock_data, prepare_data, calculate_metrics, create_stock_plot

class StockPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Predictor")
        self.root.geometry("1200x800")
        
        # Create frames
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(fill='x')
        
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(fill='both', expand=True)
        
        self.result_frame = ttk.Frame(root, padding="10")
        self.result_frame.pack(fill='x')
        
        # Create controls
        self.setup_controls()
        
        # Popular stocks list
        self.popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM']
        
    def setup_controls(self):
        # Stock input
        ttk.Label(self.control_frame, text="Stock Symbol:").pack(side='left', padx=5)
        self.stock_var = tk.StringVar()
        self.stock_entry = ttk.Entry(self.control_frame, textvariable=self.stock_var)
        self.stock_entry.pack(side='left', padx=5)
        
        # Popular stocks dropdown
        ttk.Label(self.control_frame, text="Popular Stocks:").pack(side='left', padx=5)
        self.popular_var = tk.StringVar()
        self.popular_dropdown = ttk.Combobox(self.control_frame, textvariable=self.popular_var, 
                                           values=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM'])
        self.popular_dropdown.pack(side='left', padx=5)
        self.popular_dropdown.bind('<<ComboboxSelected>>', self.on_stock_select)
        
        # Predict button
        self.predict_button = ttk.Button(self.control_frame, text="Predict", command=self.predict)
        self.predict_button.pack(side='left', padx=5)
        
        # Results labels
        self.prediction_label = ttk.Label(self.result_frame, text="")
        self.prediction_label.pack(side='left', padx=5)
        
        self.actual_label = ttk.Label(self.result_frame, text="")
        self.actual_label.pack(side='left', padx=5)
        
    def on_stock_select(self, event):
        self.stock_var.set(self.popular_var.get())
        
    def predict(self):
        ticker = self.stock_var.get().upper()
        if not ticker:
            self.prediction_label.config(text="Please enter a stock symbol")
            return
            
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        
        if stock_data is None or stock_data.empty:
            self.prediction_label.config(text="Error fetching stock data")
            return
            
        # Prepare data
        closing_prices = stock_data['Close'].values
        dates = stock_data.index
        normalized_prices = (closing_prices - np.mean(closing_prices)) / np.std(closing_prices)
        
        X, y = prepare_data(normalized_prices)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape data
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_2d, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_2d)
        test_pred = model.predict(X_test_2d)
        
        # Calculate metrics
        r2_score = calculate_metrics(y_test, test_pred)
        
        # Create plot and get accuracy
        accuracy = create_stock_plot(self.plot_frame, y_train, train_pred, y_test, test_pred, ticker, r2_score, dates)
        
        # Make future prediction
        last_sequence = normalized_prices[-30:].reshape(1, -1)
        next_day_pred = model.predict(last_sequence)
        # Fix the scalar conversion
        std_price = float(np.std(closing_prices))
        mean_price = float(np.mean(closing_prices))
        predicted_price = float(next_day_pred[0]) * std_price + mean_price
        last_actual_price = float(closing_prices[-1])
        
        # Update labels with accuracy
        self.prediction_label.config(
            text=f"Predicted next day price: ${predicted_price:.2f}")
        self.actual_label.config(
            text=f"Last actual price: ${last_actual_price:.2f} | Model Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorApp(root)
    root.mainloop()