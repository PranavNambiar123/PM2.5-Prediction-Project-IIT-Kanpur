import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from load_data import load_all_localities_data
from datetime import datetime, timedelta
import seaborn as sns
import os

class NextDayPredictor:
    def __init__(self, locality='Airport'):
        """Initialize predictor for a specific locality"""
        self.locality = locality
        self.model = None
        self.feature_cols = None
        self.scaler = None
        
        # Create output directory for predictions
        os.makedirs('predictions', exist_ok=True)
    
    def prepare_data(self, df):
        """Prepare features for prediction"""
        # Use all numeric columns except PM2.5 as features
        self.feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col != 'PM2.5']
        return df[self.feature_cols].values, df['PM2.5'].values
    
    def train_model(self, train_data):
        """Train the model on historical data"""
        X, y = self.prepare_data(train_data)
        self.model = LinearRegression()
        self.model.fit(X, y)
        return self.model
    
    def predict_next_day(self, current_data):
        """Predict PM2.5 for the next 24 hours"""
        X = current_data[self.feature_cols].values
        return self.model.predict(X)
    
    def evaluate_predictions(self, test_data):
        """Evaluate predictions against actual values"""
        # Get predictions
        X_test = test_data[self.feature_cols].values
        y_pred = self.model.predict(X_test)
        y_true = test_data['PM2.5'].values
        
        # Calculate error metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        
        return {
            'predictions': y_pred,
            'actual': y_true,
            'mae': mae,
            'rmse': rmse
        }
    
    def plot_predictions(self, results, date):
        """Plot predictions vs actual values"""
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot actual vs predicted values
        hours = np.arange(24)
        plt.plot(hours, results['actual'], 'b-', label='Actual', marker='o')
        plt.plot(hours, results['predictions'], 'r--', label='Predicted', marker='x')
        
        # Add error bands (standard deviation of error)
        error = results['predictions'] - results['actual']
        std_error = np.std(error)
        plt.fill_between(hours, 
                        results['predictions'] - std_error,
                        results['predictions'] + std_error,
                        alpha=0.2, color='red',
                        label='Prediction Uncertainty')
        
        # Customize plot
        plt.title(f'PM2.5 Predictions vs Actual Values for {self.locality}\n{date.strftime("%Y-%m-%d")}')
        plt.xlabel('Hour of Day')
        plt.ylabel('PM2.5 Concentration')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add error metrics as text
        plt.text(0.02, 0.98, 
                f'MAE: {results["mae"]:.2f}\nRMSE: {results["rmse"]:.2f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Save plot
        plt.savefig(f'predictions/next_day_prediction_{self.locality}_{date.strftime("%Y%m%d")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(error, kde=True)
        plt.title(f'Prediction Error Distribution for {self.locality}\n{date.strftime("%Y-%m-%d")}')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Count')
        plt.savefig(f'predictions/error_distribution_{self.locality}_{date.strftime("%Y%m%d")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Load data
    print("Loading data...")
    data_dict = load_all_localities_data()
    
    # Choose a locality with good data quality (Airport has shown good completeness)
    locality = 'Airport'
    df = data_dict[locality]
    
    # Clean data - drop rows with any NaN values
    print("Cleaning data...")
    df = df.dropna()
    
    # Initialize predictor
    predictor = NextDayPredictor(locality)
    
    # Choose a date for prediction (let's use a date in the middle of our dataset)
    all_dates = df.index.date
    unique_dates = np.unique(all_dates)
    
    # Find a date with complete 24-hour data
    prediction_date = None
    for date in unique_dates[len(unique_dates)//2:]:
        day_data = df[df.index.date == date]
        if len(day_data) == 24:  # Complete day of hourly data
            prediction_date = date
            break
    
    if prediction_date is None:
        print("Could not find a complete day of data!")
        return
    
    print(f"\nPredicting PM2.5 levels for {prediction_date}")
    
    # Split data into training (before prediction date) and testing (prediction date)
    train_data = df[df.index.date < prediction_date]
    test_data = df[df.index.date == prediction_date]
    
    if len(test_data) < 24:
        print(f"Warning: Only {len(test_data)} hours of data available for test date")
    
    # Train model
    print("Training model...")
    predictor.train_model(train_data)
    
    # Make and evaluate predictions
    print("Making predictions...")
    results = predictor.evaluate_predictions(test_data)
    
    # Plot results
    print("Generating plots...")
    predictor.plot_predictions(results, prediction_date)
    
    print(f"\nPrediction Results for {prediction_date}:")
    print(f"Mean Absolute Error: {results['mae']:.2f}")
    print(f"Root Mean Square Error: {results['rmse']:.2f}")
    print("\nPlots have been saved in the 'predictions' directory!")

if __name__ == "__main__":
    main()
