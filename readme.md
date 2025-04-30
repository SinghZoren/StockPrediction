# README.md

# Stock Price Predictor

A simple stock price prediction application using Linear Regression and historical stock data.

## Features
- Interactive GUI for stock selection
- Real-time stock data fetching using yfinance
- Linear Regression model for price prediction
- Visual representation of actual vs predicted prices
- R-squared score calculation for model evaluation

## Requirements
- Python 3.x
- Required packages: numpy, matplotlib, yfinance, pandas, tkinter

## Installation
1. Clone this repository
2. Install required packages:
   ```
   pip install numpy matplotlib yfinance pandas
   ```

## Usage
1. Run main.py:
   ```
   python main.py
   ```
2. Enter a stock symbol or select from popular stocks
3. Click "Predict" to see the results

## File Structure
- main.py: Main application file with GUI
- linear_regression.py: Linear Regression model implementation
- utils.py: Utility functions for data processing and visualization
- README.md: Project documentation

## Notes
- The prediction is based on historical closing prices only
- Past performance does not guarantee future results