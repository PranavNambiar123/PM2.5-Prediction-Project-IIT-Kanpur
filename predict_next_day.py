import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from load_data import load_all_localities_data
from datetime import datetime, timedelta
import seaborn as sns

class NextDayPredictor:
    def __init__(self, locality='Airport'):
        """Initialize predictor for a specific locality"""
        self.locality = locality
        self.model = None
        self.feature_cols = None
        self.scaler = None
        
        os.makedirs('predictions', exist_ok=True)
    
    def prepare_data(self, df):
        """Prepare features for prediction"""
        df = df.dropna()
        
        self.feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in ['PM2.5', 'From Date']]
        
        if not self.feature_cols:
            raise ValueError("No valid feature columns found in the data")
            
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
        X_test = test_data[self.feature_cols].values
        y_pred = self.model.predict(X_test)
        y_true = test_data['PM2.5'].values
        
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
        plt.figure(figsize=(15, 10))
        
        hours = np.arange(24)
        plt.plot(hours, results['actual'], 'b-', label='Actual', marker='o')
        plt.plot(hours, results['predictions'], 'r--', label='Predicted', marker='x')
        
        error = results['predictions'] - results['actual']
        std_error = np.std(error)
        plt.fill_between(hours, 
                        results['predictions'] - std_error,
                        results['predictions'] + std_error,
                        alpha=0.2, color='red',
                        label='Prediction Uncertainty')
        
        plt.title(f'PM2.5 Predictions vs Actual Values for {self.locality}\n{date.strftime("%Y-%m-%d")}')
        plt.xlabel('Hour of Day')
        plt.ylabel('PM2.5 Concentration')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.text(0.02, 0.98, 
                f'MAE: {results["mae"]:.2f}\nRMSE: {results["rmse"]:.2f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.savefig(f'predictions/next_day_prediction_{self.locality}_{date.strftime("%Y%m%d")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(error, kde=True)
        plt.title(f'Prediction Error Distribution for {self.locality}\n{date.strftime("%Y-%m-%d")}')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Count')
        plt.savefig(f'predictions/error_distribution_{self.locality}_{date.strftime("%Y%m%d")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

def load_cleaned_data():
    """Load all cleaned datasets for Airport"""
    cleaned_data = {}
    
    csv_files = glob.glob(os.path.join('cleaned_data', 'Airport_*.csv'))
    
    for file_path in csv_files:
        method = os.path.basename(file_path).replace('Airport_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path, parse_dates=True)
            df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
            df = df.iloc[:, 1:]
            
            cleaned_data[method] = df
            print(f"Successfully loaded {method} data with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {method} data: {str(e)}")
    
    return cleaned_data

def evaluate_cleaning_methods():
    """Evaluate predictions using different cleaning methods"""
    print("Loading cleaned datasets...")
    cleaned_datasets = load_cleaned_data()
    
    all_results = {}
    
    for method, df in cleaned_datasets.items():
        print(f"\nEvaluating {method} cleaning method...")
        try:
            predictor = NextDayPredictor('Airport')
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            all_dates = df.index.date
            unique_dates = np.unique(all_dates)
            
            prediction_date = None
            for date in unique_dates[len(unique_dates)//2:]:
                day_data = df[df.index.date == date]
                if len(day_data) == 24:
                    prediction_date = date
                    break
            
            if prediction_date is None:
                print(f"Skipping {method} - Could not find a complete day of data")
                continue
            
            print(f"Predicting PM2.5 levels for {prediction_date}")
            
            train_data = df[df.index.date < prediction_date].copy()
            test_data = df[df.index.date == prediction_date].copy()
            
            if len(test_data) < 24:
                print(f"Warning: Only {len(test_data)} hours of data available for test date")
                continue
            
            predictor.train_model(train_data)
            results = predictor.evaluate_predictions(test_data)
            
            all_results[method] = results
            
            predictor.plot_predictions(results, prediction_date)
            plt.savefig(os.path.join('predictions', f'predictions_{method}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error evaluating {method}: {str(e)}")
            continue
    
    return all_results

def main():
    results = evaluate_cleaning_methods()
    if results:
        plt.figure(figsize=(15, 10))
        hours = np.arange(24)
        
        for method, data in results.items():
            results = data
            plt.plot(hours, results['predictions'], '--', label=f'{method} (MAE: {results["mae"]:.2f})', alpha=0.7)
        
        plt.plot(hours, results['actual'], 'k-', label='Actual', linewidth=2)
        
        plt.title(f'Comparison of PM2.5 Predictions Across Cleaning Methods')
        plt.xlabel('Hour of Day')
        plt.ylabel('PM2.5 Concentration')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('predictions/cleaning_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        methods = list(results.keys())
        mae_values = [results[m]['mae'] for m in methods]
        rmse_values = [results[m]['rmse'] for m in methods]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(methods))
        width = 0.35
        
        plt.bar(x - width/2, mae_values, width, label='MAE')
        plt.bar(x + width/2, rmse_values, width, label='RMSE')
        
        plt.xlabel('Cleaning Method')
        plt.ylabel('Error')
        plt.title('Comparison of Error Metrics Across Cleaning Methods')
        plt.xticks(x, methods, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions/error_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nSummary of Results:")
        print("=" * 50)
        for method in methods:
            results = results[method]
            print(f"\n{method}:")
            print(f"  MAE:  {results['mae']:.2f}")
            print(f"  RMSE: {results['rmse']:.2f}")

if __name__ == "__main__":
    main()
