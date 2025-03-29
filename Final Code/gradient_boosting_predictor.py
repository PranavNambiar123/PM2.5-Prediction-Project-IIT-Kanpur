import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import time
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradient_boosting_training.log'),
        logging.StreamHandler()
    ]
)

class GradientBoostingPredictor:
    def __init__(self, locality='Airport', sequence_length=24, prediction_horizon=24):
        """
        Initialize Gradient Boosting predictor for a specific locality
        Args:
            locality (str): Name of the locality to predict for
            sequence_length (int): Number of past time steps to use for prediction
            prediction_horizon (int): Number of future time steps to predict
        """
        self.locality = locality
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.models = []  # List to store models for each prediction step
        self.feature_scalers = {}
        
        logging.info(f"Initializing Gradient Boosting Predictor for {locality}")
        logging.info(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}")
        
        # Create directories for saving models and predictions
        os.makedirs('models', exist_ok=True)
        os.makedirs('predictions', exist_ok=True)

    def load_cleaned_data(self):
        """
        Load cleaned data from the cleaned_data directory
        Returns:
            pd.DataFrame: Cleaned data for the locality
        """
        # Look for files matching the locality name and spatial average method
        cleaned_files = glob(os.path.join('cleaned_data', f'{self.locality}_spatial_average.csv'))
        
        if not cleaned_files:
            raise ValueError(f"No spatial average cleaned data found for locality {self.locality}")
        
        data_path = cleaned_files[0]
        logging.info(f"Loading cleaned data from {data_path}")
        
        # Load the data
        df = pd.read_csv(data_path)
        
        # Convert 'From Date' to datetime index
        df['From Date'] = pd.to_datetime(df['From Date'])
        df.set_index('From Date', inplace=True)
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        # Select only numeric columns
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'PM2.5']
        
        if not feature_cols:
            raise ValueError(f"No numeric features found in columns: {df.columns}")
        
        # Keep only PM2.5 and feature columns
        df = df[['PM2.5'] + feature_cols]
        
        logging.info(f"Loaded cleaned data with shape: {df.shape}")
        logging.info(f"Features selected: {feature_cols}")
        
        return df

    def prepare_sequences(self, data):
        """
        Prepare sequences for gradient boosting with enhanced feature engineering
        """
        start_time = time.time()
        logging.info(f"Preparing sequences from data with shape: {data.shape}")
        
        # Add enhanced time-based features
        data = data.copy()
        data['hour'] = data.index.hour.astype(float)
        data['day'] = data.index.day.astype(float)
        data['month'] = data.index.month.astype(float)
        data['day_of_week'] = data.index.dayofweek.astype(float)
        data['week_of_year'] = data.index.isocalendar().week.astype(float)
        
        # Add cyclical time features for multiple periods
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Enhanced rolling features
        data['rolling_mean_6h'] = data['PM2.5'].rolling(window=6).mean()
        data['rolling_mean_12h'] = data['PM2.5'].rolling(window=12).mean()
        data['rolling_mean_24h'] = data['PM2.5'].rolling(window=24).mean()
        data['rolling_std_6h'] = data['PM2.5'].rolling(window=6).std()
        data['rolling_std_12h'] = data['PM2.5'].rolling(window=12).std()
        data['rolling_max_6h'] = data['PM2.5'].rolling(window=6).max()
        data['rolling_min_6h'] = data['PM2.5'].rolling(window=6).min()
        
        # Add lag features
        data['lag_1h'] = data['PM2.5'].shift(1)
        data['lag_3h'] = data['PM2.5'].shift(3)
        data['lag_6h'] = data['PM2.5'].shift(6)
        
        # Add rate of change features
        data['rate_of_change_1h'] = data['PM2.5'].diff(1)
        data['rate_of_change_3h'] = data['PM2.5'].diff(3)
        
        # Fill NaN values from rolling calculations
        data = data.bfill().ffill()  # Forward fill after backward fill to handle any remaining NaNs
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != 'PM2.5']
        X_raw = data[feature_cols].values
        y_raw = data['PM2.5'].values
        
        # Scale features
        if 'features' not in self.feature_scalers:
            self.feature_scalers['features'] = StandardScaler()
            X_scaled = self.feature_scalers['features'].fit_transform(X_raw)
        else:
            X_scaled = self.feature_scalers['features'].transform(X_raw)
            
        if 'PM2.5' not in self.feature_scalers:
            self.feature_scalers['PM2.5'] = StandardScaler()
            y_scaled = self.feature_scalers['PM2.5'].fit_transform(y_raw.reshape(-1, 1))
        else:
            y_scaled = self.feature_scalers['PM2.5'].transform(y_raw.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        total_sequences = len(data) - self.sequence_length - self.prediction_horizon + 1
        logging.info(f"Creating {total_sequences} sequences...")
        
        for i in range(total_sequences):
            if i % 1000 == 0:
                logging.debug(f"Processing sequence {i}/{total_sequences}")
            
            # Input sequence includes all features
            X.append(X_scaled[i:(i + self.sequence_length)])
            
            # Target sequence is only PM2.5
            y.append(y_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        # Reshape X to 2D array for XGBoost (flatten the sequence)
        X = np.array(X).reshape(len(X), -1)
        y = np.array(y)
        
        processing_time = time.time() - start_time
        logging.info(f"Sequence preparation completed. X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Sequence preparation took {processing_time:.2f} seconds")
        
        return X, y

    def train_model(self, train_data, validation_split=0.2):
        """
        Train the XGBoost models (one for each prediction horizon)
        Args:
            train_data (pd.DataFrame): Training data
            validation_split (float): Fraction of data to use for validation
        """
        start_time = time.time()
        logging.info(f"Starting model training with {len(train_data)} samples")
        
        X, y = self.prepare_sequences(train_data)
        
        # Split into train and validation sets
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train separate models for each prediction step
        self.models = []
        for step in range(self.prediction_horizon):
            logging.info(f"Training model for step {step + 1}/{self.prediction_horizon}")
            
            # Create and train XGBoost model with optimized hyperparameters
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.9,
                colsample_bytree=0.9,
                colsample_bylevel=0.9,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                tree_method='hist',  # Faster histogram-based algorithm
                booster='gbtree'
            )
            
            # Train model
            model.fit(
                X_train, y_train[:, step],
                eval_set=[(X_val, y_val[:, step])],
                verbose=False
            )
            
            self.models.append(model)
            
            # Save model
            model.save_model(f'models/xgb_{self.locality}_step_{step}.json')
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        
    def predict_next_days(self, current_data):
        """
        Predict PM2.5 for the next days
        Args:
            current_data (pd.DataFrame): Current data to base prediction on
        Returns:
            np.array: Predicted PM2.5 values
        """
        logging.info(f"Making predictions for {self.locality}")
        X, _ = self.prepare_sequences(current_data)
        
        # Take the last sequence for prediction
        last_sequence = X[-1:]
        
        # Make predictions for each step
        predictions = []
        for step, model in enumerate(self.models):
            pred = model.predict(last_sequence)
            predictions.append(pred[0])
        
        predictions = np.array(predictions).reshape(-1, 1)
        
        # Inverse transform predictions
        predictions = self.feature_scalers['PM2.5'].inverse_transform(predictions)
        
        logging.debug(f"Predicted values: {predictions.flatten()}")
        
        return predictions.flatten()

    def evaluate_predictions(self, test_data):
        """
        Evaluate predictions against actual values
        Args:
            test_data (pd.DataFrame): Test data
        Returns:
            dict: Dictionary containing evaluation metrics and predictions
        """
        logging.info(f"Evaluating predictions for {self.locality}")
        start_time = time.time()
        
        X_test, y_test = self.prepare_sequences(test_data)
        
        # Make predictions for each step
        predictions = []
        for step, model in enumerate(self.models):
            pred = model.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions).T
        
        # Reshape y_test to 2D
        y_test_2d = y_test.reshape(-1, self.prediction_horizon)
        
        # Inverse transform predictions and actual values
        predictions = self.feature_scalers['PM2.5'].inverse_transform(predictions)
        actual = self.feature_scalers['PM2.5'].inverse_transform(y_test_2d)
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predictions))
        rmse = np.sqrt(np.mean((actual - predictions)**2))
        
        evaluation_time = time.time() - start_time
        logging.info(f"Evaluation completed in {evaluation_time:.2f} seconds")
        logging.info(f"Evaluation metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return {
            'predictions': predictions,
            'actual': actual,
            'mae': mae,
            'rmse': rmse
        }

    def plot_predictions(self, results, date):
        """
        Plot predictions vs actual values
        Args:
            results (dict): Dictionary containing predictions and actual values
            date (datetime): Date of prediction
        """
        logging.info(f"Generating prediction plots for {self.locality}")
        
        plt.figure(figsize=(15, 10))
        
        hours = np.arange(self.prediction_horizon)
        plt.plot(hours, results['actual'][0], 'b-', label='Actual', marker='o')
        plt.plot(hours, results['predictions'][0], 'r--', label='Predicted', marker='x')
        
        # Add confidence intervals
        error = results['predictions'] - results['actual']
        std_error = np.std(error, axis=0)
        plt.fill_between(
            hours,
            results['predictions'][0] - std_error,
            results['predictions'][0] + std_error,
            alpha=0.2, color='red',
            label='Prediction Uncertainty'
        )
        
        plt.title(f'PM2.5 Predictions vs Actual Values for {self.locality} (XGBoost)')
        plt.xlabel('Hours Ahead')
        plt.ylabel('PM2.5 Concentration')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add metrics text
        plt.text(0.02, 0.98,
                f'MAE: {results["mae"]:.2f}\nRMSE: {results["rmse"]:.2f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plot_path = f'predictions/xgboost_prediction_{self.locality}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved prediction plot to {plot_path}")
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        error_flat = error.flatten()
        sns.histplot(error_flat, kde=True)
        plt.title(f'Prediction Error Distribution for {self.locality} (XGBoost)')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Count')
        
        error_plot_path = f'predictions/xgboost_error_distribution_{self.locality}.png'
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved error distribution plot to {error_plot_path}")

def main():
    """
    Main function to demonstrate XGBoost predictor usage
    """
    start_time = time.time()
    logging.info("Starting XGBoost predictor main execution")
    
    csv_files = glob(os.path.join('cleaned_data', '*.csv'))
    localities = []
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        locality, method = filename.replace('.csv', '').split('_', 1)
        localities.append(locality)
    
    final_results = []
    
    for locality in localities:
        try:
            # Create predictor
            predictor = GradientBoostingPredictor(
                locality=locality,
                sequence_length=24,  # Use 24 hours of historical data
                prediction_horizon=6  # Predict next 6 hours
            )
            
            # Load and prepare data
            df = predictor.load_cleaned_data()
            
            # Split data into train and test
            train_size = int(len(df) * 0.8)
            train_data = df[:train_size]
            test_data = df[train_size:]
            
            # Train model
            predictor.train_model(
                train_data,
                validation_split=0.2
            )
            
            # Evaluate on test data
            results = predictor.evaluate_predictions(test_data)
            
            # Plot results
            predictor.plot_predictions(results, test_data.index[0])
            
            logging.info("\nFinal Evaluation Results:")
            logging.info(f"MAE: {results['mae']:.2f}")
            logging.info(f"RMSE: {results['rmse']:.2f}")
            final_results.append(results)
        
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise
    
    total_execution_time = time.time() - start_time
    logging.info(f"Total execution time: {total_execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
