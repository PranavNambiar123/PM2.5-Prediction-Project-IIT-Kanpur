import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import time
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_model.log'),
        logging.StreamHandler()
    ]
)

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator"""
        self.results = {}
        self.models = {}
        
    def load_cleaned_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all cleaned datasets
        Returns: Dictionary with structure {cleaning_method: {locality: dataframe}}
        """
        cleaned_data = {}
        
        csv_files = glob(os.path.join('cleaned_data', '*.csv'))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            locality, method = filename.replace('.csv', '').split('_', 1)
            
            df = pd.read_csv(file_path)
            if 'Unnamed: 0' in df.columns:
                df.set_index(pd.to_datetime(df['Unnamed: 0']), inplace=True)
                df.drop('Unnamed: 0', axis=1, inplace=True)
            
            if method not in cleaned_data:
                cleaned_data[method] = {}
            
            cleaned_data[method][locality] = df
            
        return cleaned_data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for modeling
        """
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'PM2.5']
        
        if not feature_cols:
            raise ValueError(f"No numeric features found in columns: {df.columns}")
            
        X = df[feature_cols].values
        y = df['PM2.5'].values
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets (70-15-15)
        """
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_locality(self, df: pd.DataFrame, locality: str, method: str) -> Dict:
        """
        Evaluate a single locality's data
        """
        try:
            df = df.dropna()
            
            if len(df) < 100:  # arbitrary minimum size
                print(f"Skipping {locality} due to insufficient data (only {len(df)} samples)")
                return None
                
            X, y = self.prepare_features(df)
            
            if X.shape[1] == 0:
                print(f"Skipping {locality} due to no valid features")
                return None
                
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            if method not in self.models:
                self.models[method] = {}
            self.models[method][locality] = model
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_r2': r2_score(y_val, y_val_pred),
                'test_r2': r2_score(y_test, y_test_pred)
            }
            
            return metrics
        except Exception as e:
            print(f"Error processing {locality}: {str(e)}")
            return None
    
    def evaluate_all_methods(self):
        """
        Evaluate all cleaning methods and localities
        """
        cleaned_data = self.load_cleaned_data()
        
        for method, localities_data in cleaned_data.items():
            print(f"\nEvaluating {method} method...")
            method_results = {}
            
            for locality, df in localities_data.items():
                print(f"Processing {locality}...")
                metrics = self.evaluate_locality(df, locality, method)
                if metrics is not None:
                    method_results[locality] = metrics
            
            if method_results:
                self.results[method] = method_results
    
    def plot_comparison(self):
        """
        Plot comparison of different cleaning methods
        """
        if not self.results:
            print("No results to plot!")
            return
            
        os.makedirs('model_results', exist_ok=True)
        
        methods = list(self.results.keys())
        metrics = ['test_rmse', 'test_mae', 'test_r2']
        metric_names = ['RMSE', 'MAE', 'R²']
        
        for metric, metric_name in zip(metrics, metric_names):
            plt.figure(figsize=(12, 6))
            data = []
            labels = []
            
            for method in methods:
                values = [self.results[method][locality][metric] 
                         for locality in self.results[method].keys()]
                if values:
                    data.append(values)
                    labels.extend([method] * len(values))
            
            if data:
                plt.boxplot(data, labels=methods)
                plt.title(f'Comparison of {metric_name} Across Cleaning Methods')
                plt.ylabel(metric_name)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'model_results/comparison_{metric}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        results_df = pd.DataFrame()
        for method in methods:
            method_data = pd.DataFrame(self.results[method]).T
            method_data['method'] = method
            results_df = pd.concat([results_df, method_data])
        
        results_df.to_csv('model_results/all_results.csv')
        print("\nResults have been saved to 'model_results' directory!")


class LSTMPredictor:
    def __init__(self, locality='Airport', sequence_length=24, prediction_horizon=24):
        """
        Initialize LSTM predictor for a specific locality
        Args:
            locality (str): Name of the locality to predict for
            sequence_length (int): Number of past time steps to use for prediction
            prediction_horizon (int): Number of future time steps to predict
        """
        self.locality = locality
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.feature_scalers = {}
        
        logging.info(f"Initializing LSTM Predictor for {locality}")
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
        
        # Convert the index to datetime if it exists
        if 'Unnamed: 0' in df.columns:
            df.set_index(pd.to_datetime(df['Unnamed: 0']), inplace=True)
            df.drop('Unnamed: 0', axis=1, inplace=True)
        
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
        Prepare sequences for LSTM training
        Args:
            data (pd.DataFrame): Input dataframe with features
        Returns:
            X (np.array): Sequence inputs
            y (np.array): Target outputs
        """
        start_time = time.time()
        logging.info(f"Preparing sequences from data with shape: {data.shape}")
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != 'PM2.5']
        X_raw = data[feature_cols].values
        y_raw = data['PM2.5'].values
        
        # Scale features
        if 'features' not in self.feature_scalers:
            self.feature_scalers['features'] = MinMaxScaler(feature_range=(-1, 1))
            X_scaled = self.feature_scalers['features'].fit_transform(X_raw)
        else:
            X_scaled = self.feature_scalers['features'].transform(X_raw)
            
        # Scale target separately
        if 'PM2.5' not in self.feature_scalers:
            self.feature_scalers['PM2.5'] = MinMaxScaler(feature_range=(-1, 1))
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
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, self.prediction_horizon)
        
        processing_time = time.time() - start_time
        logging.info(f"Sequence preparation completed. X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Sequence preparation took {processing_time:.2f} seconds")
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        Args:
            input_shape (tuple): Shape of input sequences (sequence_length, n_features)
        """
        logging.info(f"Building LSTM model with input shape: {input_shape}")
        
        # Use smaller network initially to prevent overfitting
        self.model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(self.prediction_horizon)
        ])
        
        # Use a smaller learning rate and gradient clipping
        optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logging.info("Model architecture:")
        self.model.summary(print_fn=logging.info)
        
    def train_model(self, train_data, validation_split=0.2, epochs=100, batch_size=32):
        """
        Train the LSTM model
        Args:
            train_data (pd.DataFrame): Training data
            validation_split (float): Fraction of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        start_time = time.time()
        logging.info(f"Starting model training with {len(train_data)} samples")
        logging.info(f"Training parameters - epochs: {epochs}, batch_size: {batch_size}, validation_split: {validation_split}")
        
        X, y = self.prepare_sequences(train_data)
        
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            f'models/lstm_{self.locality}.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        custom_callback = CustomCallback()
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, custom_callback],
            verbose=0  # Disable default verbose output since we're using custom callback
        )
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        logging.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        logging.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
    
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
        
        # Make prediction
        start_time = time.time()
        scaled_prediction = self.model.predict(last_sequence)
        prediction_time = time.time() - start_time
        
        # Inverse transform the prediction
        prediction = self.feature_scalers['PM2.5'].inverse_transform(scaled_prediction.reshape(-1, 1))
        
        logging.info(f"Prediction completed in {prediction_time:.2f} seconds")
        logging.debug(f"Predicted values: {prediction.flatten()}")
        
        return prediction.flatten()
    
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
        
        # Make predictions
        scaled_predictions = self.model.predict(X_test)
        
        # Reshape predictions and actual values for inverse transform
        predictions = self.feature_scalers['PM2.5'].inverse_transform(scaled_predictions.reshape(-1, 1)).reshape(-1, self.prediction_horizon)
        actual = self.feature_scalers['PM2.5'].inverse_transform(y_test.reshape(-1, 1)).reshape(-1, self.prediction_horizon)
        
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
        
        plt.title(f'PM2.5 Predictions vs Actual Values for {self.locality}\n{date.strftime("%Y-%m-%d")}')
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
        
        plot_path = f'predictions/lstm_prediction_{self.locality}_{date.strftime("%Y%m%d")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved prediction plot to {plot_path}")
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        error_flat = error.flatten()
        sns.histplot(error_flat, kde=True)
        plt.title(f'Prediction Error Distribution for {self.locality}\n{date.strftime("%Y-%m-%d")}')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Count')
        
        error_plot_path = f'predictions/lstm_error_distribution_{self.locality}_{date.strftime("%Y%m%d")}.png'
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved error distribution plot to {error_plot_path}")

def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_methods()
    evaluator.plot_comparison()

if __name__ == "__main__":
    main()
