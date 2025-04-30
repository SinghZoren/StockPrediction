import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None
    
def prepare_data(data, lookback=30):
    x = []
    y = []
    for i in range(len(data) - lookback):
        x.append(data[i:(i+lookback)])
        y.append(data[i+lookback])

    return np.array(x), np.array(y)

def calculate_metrics(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return r2_score

def create_stock_plot(canvas_widget, y_train, train_pred, y_test, test_pred, ticker, r2_score, dates):
    # Clear previous plot
    for widget in canvas_widget.winfo_children():
        widget.destroy()

    # Create new figure with more width for legend
    fig = Figure(figsize=(18, 10))
    
    # Create two subplots with more space between them
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot all historical data in top subplot
    train_dates = dates[:len(y_train)]
    test_dates = dates[len(y_train):len(y_train) + len(y_test)]
    
    # Plot training and testing data with a vertical line separating them
    ax1.plot(train_dates, y_train, label='Training Data (Actual)', alpha=0.7, color='blue', linewidth=2)
    ax1.plot(train_dates, train_pred, label='Training Data (Predicted)', alpha=0.7, color='lightblue', linewidth=2)
    ax1.plot(test_dates, y_test, label='Testing Data (Actual)', alpha=0.7, color='green', linewidth=2)
    ax1.plot(test_dates, test_pred, label='Testing Data (Predicted)', alpha=0.7, color='lightgreen', linewidth=2)
    
    # Add vertical line to separate training and testing data
    if len(train_dates) > 0 and len(test_dates) > 0:
        separation_date = test_dates[0]
        ax1.axvline(x=separation_date, color='red', linestyle='--', alpha=0.5, label='Train/Test Split')
    
    # Format the title with accuracy percentage
    accuracy_percentage = r2_score * 100
    main_title = (f'{ticker} Stock Price Prediction - Full History\n'
                f'Model Accuracy: {accuracy_percentage:.2f}%\n'
                f'Training Period: {len(train_dates)} days | Testing Period: {len(test_dates)} days')
    ax1.set_title(main_title, pad=20, fontsize=12, fontweight='bold')
    
    # Format axes for full history plot
    ax1.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Normalized Price', fontsize=10)
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Create custom legend with explanation
    legend_elements = [
        plt.Line2D([0], [0], color='blue', label='Training Data (Actual)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='lightblue', label='Training Data (Predicted)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='green', label='Testing Data (Actual)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='lightgreen', label='Testing Data (Predicted)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='red', label='Train/Test Split', linestyle='--', alpha=0.5)
    ]
    ax1.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9,
               title='Full History Explanation:\n' +
                     '• Blue lines show historical training data\n' +
                     '• Green lines show recent testing data\n' +
                     '• Lighter colors are model predictions\n' +
                     '• Red line marks where training ends\n' +
                     '• Closer predictions = better accuracy',
               title_fontsize=9)

    # For the recent view, only show testing data and the last week of training
    last_week_mask = slice(-7, None)  # Last 7 days of training
    recent_train_dates = train_dates[last_week_mask] if len(train_dates) > 7 else train_dates
    recent_train_actual = y_train[last_week_mask] if len(y_train) > 7 else y_train
    recent_train_pred = train_pred[last_week_mask] if len(train_pred) > 7 else train_pred

    # Plot recent data
    if len(recent_train_dates) > 0:
        ax2.plot(recent_train_dates, recent_train_actual, label='Last Week Training (Actual)', 
                alpha=0.7, color='blue', linewidth=2)
        ax2.plot(recent_train_dates, recent_train_pred, label='Last Week Training (Predicted)', 
                alpha=0.7, color='lightblue', linewidth=2)
    
    ax2.plot(test_dates, y_test, label='Testing Data (Actual)', 
            alpha=0.7, color='green', linewidth=2)
    ax2.plot(test_dates, test_pred, label='Testing Data (Predicted)', 
            alpha=0.7, color='lightgreen', linewidth=2)
    
    # Add vertical line for train/test split in recent view
    if len(test_dates) > 0:
        ax2.axvline(x=test_dates[0], color='red', linestyle='--', alpha=0.5, label='Train/Test Split')
    
    detail_title = ('Recent Performance Detail View\n'
                   'Showing last week of training and all testing data')
    ax2.set_title(detail_title, pad=20, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Normalized Price', fontsize=10)
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Create custom legend for detail view
    detail_legend_elements = [
        plt.Line2D([0], [0], color='blue', label='Last Week Training (Actual)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='lightblue', label='Last Week Training (Predicted)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='green', label='Testing Data (Actual)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='lightgreen', label='Testing Data (Predicted)', alpha=0.7, linewidth=2),
        plt.Line2D([0], [0], color='red', label='Train/Test Split', linestyle='--', alpha=0.5)
    ]
    ax2.legend(handles=detail_legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9,
               title='Recent View Explanation:\n' +
                     '• Shows most recent predictions\n' +
                     '• Compare actual vs predicted\n' +
                     '• Closer lines = better predictions\n' +
                     '• Use this to gauge current\n  model performance',
               title_fontsize=9)

    # Add model performance text box
    performance_text = (
        f"Model Summary:\n"
        f"• Overall Accuracy: {accuracy_percentage:.2f}%\n"
        f"• Training Data: {len(train_dates)} days\n"
        f"• Testing Data: {len(test_dates)} days\n"
        f"• Prediction Type: Next-day price\n"
        f"• Model: Linear Regression"
    )
    
    # Position the performance text box in a better location
    fig.text(0.02, 0.02, performance_text, fontsize=10,
             bbox=dict(facecolor='wheat', alpha=0.5, edgecolor='gray', pad=5),
             transform=fig.transFigure)

    # Adjust layout
    fig.autofmt_xdate()  # Rotate date labels
    
    # Adjust subplot spacing to accommodate legends and text boxes
    plt.subplots_adjust(right=0.85, bottom=0.15, hspace=0.4, left=0.1)
    
    # Create canvas and display
    canvas = FigureCanvasTkAgg(fig, master=canvas_widget)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    
    return accuracy_percentage

