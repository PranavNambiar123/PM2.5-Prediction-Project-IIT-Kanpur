import os
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Bidirectional, Add, BatchNormalization
from tensorflow.keras.layers import Layer, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import time
import warnings
from glob import glob
import argparse
import traceback
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_training.log'),
        logging.StreamHandler()
    ]
)

# Custom attention layer implementation
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        # score shape: (batch_size, time_steps, 1)
        score = tf.matmul(inputs, self.W)
        
        # attention_weights shape: (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape: (batch_size, features)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Log training metrics after each epoch"""
        logging.info(
            f"Epoch {epoch + 1} - loss: {logs['loss']:.4f} - mae: {logs['mae']:.4f} - "
            f"val_loss: {logs['val_loss']:.4f} - val_mae: {logs['val_mae']:.4f}"
        )

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
        Load cleaned data for the locality
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
        Prepare sequences for LSTM training with improved normalization
        """
        start_time = time.time()
        logging.info(f"Preparing sequences from data with shape: {data.shape}")
        
        # Add time-based features
        data = data.copy()
        data['hour'] = data.index.hour.astype(float)
        data['day'] = data.index.day.astype(float)
        data['month'] = data.index.month.astype(float)
        data['dayofweek'] = data.index.dayofweek.astype(float)
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(float)
        
        # Add cyclical time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Essential rolling features
        data['rolling_mean_6h'] = data['PM2.5'].rolling(window=6).mean()
        data['rolling_mean_12h'] = data['PM2.5'].rolling(window=12).mean()
        data['rolling_mean_24h'] = data['PM2.5'].rolling(window=24).mean()
        data['rolling_std_12h'] = data['PM2.5'].rolling(window=12).std()
        
        # Essential lag features
        data['lag_1h'] = data['PM2.5'].shift(1)
        data['lag_3h'] = data['PM2.5'].shift(3)
        data['lag_6h'] = data['PM2.5'].shift(6)
        data['lag_12h'] = data['PM2.5'].shift(12)
        
        # Rate of change features
        data['rate_of_change_1h'] = data['PM2.5'].diff(1)
        data['rate_of_change_6h'] = data['PM2.5'].diff(6)
        
        # Fill NaN values - use forward fill first, then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != 'PM2.5']
        X_raw = data[feature_cols].values
        y_raw = data['PM2.5'].values
        
        # Normalize features using robust scaling for each feature
        if 'features' not in self.feature_scalers:
            self.feature_scalers['features'] = {}
            for i in range(X_raw.shape[1]):
                # Get feature column
                feature_col = X_raw[:, i]
                
                # Calculate median and IQR for robust scaling
                median = np.median(feature_col)
                q1 = np.percentile(feature_col, 25)
                q3 = np.percentile(feature_col, 75)
                iqr = q3 - q1
                
                # Handle zero IQR
                if iqr == 0:
                    iqr = 1.0
                
                # Store scaling parameters
                self.feature_scalers['features'][i] = {
                    'median': median,
                    'iqr': iqr
                }
        
        # Apply robust scaling to features
        X_scaled = np.zeros_like(X_raw)
        for i in range(X_raw.shape[1]):
            median = self.feature_scalers['features'][i]['median']
            iqr = self.feature_scalers['features'][i]['iqr']
            X_scaled[:, i] = (X_raw[:, i] - median) / iqr
        
        # Apply log transformation to PM2.5 target to handle skewness
        # Add a small constant to avoid log(0)
        y_log = np.log1p(y_raw)
        
        # Normalize target
        if 'target' not in self.feature_scalers:
            self.feature_scalers['target'] = {
                'mean': np.mean(y_log),
                'std': np.std(y_log)
            }
        
        y_scaled = (y_log - self.feature_scalers['target']['mean']) / self.feature_scalers['target']['std']
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        logging.info(f"Creating {len(X_scaled) - self.sequence_length - self.prediction_horizon + 1} sequences...")
        
        for i in range(len(X_scaled) - self.sequence_length - self.prediction_horizon + 1):
            X_sequences.append(X_scaled[i:i+self.sequence_length])
            y_sequences.append(y_scaled[i+self.sequence_length:i+self.sequence_length+self.prediction_horizon])
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        logging.info(f"Sequence preparation completed. X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Sequence preparation took {time.time() - start_time:.2f} seconds")
        
        return X, y
    
    def inverse_transform_predictions(self, y_pred_scaled):
        """
        Inverse transform scaled predictions back to original scale
        """
        # Inverse normalize
        y_pred_log = y_pred_scaled * self.feature_scalers['target']['std'] + self.feature_scalers['target']['mean']
        
        # Inverse log transform
        y_pred = np.expm1(y_pred_log)
        
        return y_pred
    
    def build_model(self, input_shape):
        """
        Build stacked LSTM model with residual connections
        Args:
            input_shape (tuple): Shape of input sequences (sequence_length, n_features)
        """
        logging.info(f"Building stacked LSTM model with residual connections, input shape: {input_shape}")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer with batch normalization
        x = BatchNormalization()(inputs)
        lstm1 = LSTM(128, return_sequences=True, recurrent_dropout=0.1)(x)
        lstm1 = Dropout(0.3)(lstm1)
        
        # Second LSTM layer with residual connection
        x = BatchNormalization()(lstm1)
        lstm2 = LSTM(128, return_sequences=True, recurrent_dropout=0.1)(x)
        lstm2 = Dropout(0.3)(lstm2)
        
        # Residual connection (requires same shape)
        lstm2_with_residual = Add()([lstm1, lstm2])
        
        # Third LSTM layer
        x = BatchNormalization()(lstm2_with_residual)
        lstm3 = LSTM(64, return_sequences=False, recurrent_dropout=0.1)(x)
        lstm3 = Dropout(0.3)(lstm3)
        
        # Dense layers with batch normalization and dropout
        x = BatchNormalization()(lstm3)
        dense1 = Dense(64, activation='relu')(x)
        dense1 = Dropout(0.3)(dense1)
        
        x = BatchNormalization()(dense1)
        dense2 = Dense(32, activation='relu')(x)
        dense2 = Dropout(0.2)(dense2)
        
        # Output layer
        x = BatchNormalization()(dense2)
        outputs = Dense(self.prediction_horizon)(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Use Adam optimizer with learning rate
        optimizer = Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logging.info("Model architecture:")
        self.model.summary(print_fn=logging.info)
    
    def train_model(self, train_data, validation_split=0.2, epochs=150, batch_size=32):
        """
        Train the LSTM model with advanced training strategy
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
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        model_checkpoint = ModelCheckpoint(
            f'models/lstm_{self.locality}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        # Add reduce learning rate on plateau with more patience
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=0
        )
        
        # Learning rate scheduler for warm-up and decay
        def lr_scheduler(epoch, lr):
            if epoch < 5:  # Warm-up phase
                return lr * (1.0 + 0.1 * epoch)
            else:  # Decay phase
                return lr * 0.99
        
        lr_schedule = LearningRateScheduler(lr_scheduler, verbose=0)
        
        custom_callback = CustomCallback()
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr, lr_schedule, custom_callback],
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
        prediction = self.inverse_transform_predictions(scaled_prediction)
        
        logging.info(f"Prediction completed in {prediction_time:.2f} seconds")
        logging.debug(f"Predicted values: {prediction.flatten()}")
        
        return prediction.flatten()
    
    def evaluate_predictions(self, test_data):
        """
        Evaluate model predictions on test data
        Args:
            test_data (pd.DataFrame): Test data
        Returns:
            dict: Dictionary with evaluation metrics
        """
        logging.info(f"Evaluating predictions for {self.locality}")
        start_time = time.time()
        
        # Prepare sequences
        X, y_true_scaled = self.prepare_sequences(test_data)
        
        # Make predictions
        y_pred_scaled = self.model.predict(X)
        
        # Inverse transform predictions and true values
        y_pred = self.inverse_transform_predictions(y_pred_scaled)
        y_true = self.inverse_transform_predictions(y_true_scaled)
        
        # Calculate metrics
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        
        logging.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        logging.info(f"Evaluation metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # Generate plots
        self.visualize_results({'predictions': y_pred, 'actual': y_true}, test_data)
        
        return {
            'predictions': y_pred,
            'actual': y_true,
            'mae': mae,
            'rmse': rmse
        }
    
    def visualize_results(self, results, test_data):
        """
        Visualize prediction results
        Args:
            results (dict): Dictionary with prediction results
            test_data (pd.DataFrame): Test data
        """
        # Create directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(results['actual'], label='Actual')
        plt.plot(results['predictions'], label='Predicted')
        plt.title(f'PM2.5 Prediction Results for {self.locality}')
        plt.xlabel('Time Steps')
        plt.ylabel('PM2.5')
        plt.legend()
        plt.savefig(f'plots/{self.locality}_predictions.png')
        plt.close()
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        error = results['predictions'] - results['actual']
        error_flat = error.flatten()
        sns.histplot(error_flat, kde=True)
        plt.title(f'Prediction Error Distribution for {self.locality}')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Frequency')
        plt.savefig(f'plots/{self.locality}_error_distribution.png')
        plt.close()
    
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
        
        plt.title(f'PM2.5 Predictions vs Actual Values for {self.locality}')
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
        
        plot_path = f'predictions/lstm_prediction_{self.locality}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved prediction plot to {plot_path}")
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        error_flat = (results['predictions'] - results['actual']).flatten()
        sns.histplot(error_flat, kde=True)
        plt.title(f'Prediction Error Distribution for {self.locality}')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Count')
        
        error_plot_path = f'predictions/lstm_error_distribution_{self.locality}.png'
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved error distribution plot to {error_plot_path}")

def main():
    """
    Main function to run LSTM predictor
    """
    start_time = time.time()
    logging.info("Starting LSTM predictor main execution")
    
    csv_files = glob(os.path.join('cleaned_data', '*.csv'))
    localities = []
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if '_spatial_average' in filename:
            locality = filename.split('_spatial_average')[0]
            localities.append(locality)
    
    final_results = []
    
    for locality in localities:
        try:
            # Create predictor with adjusted parameters
            predictor = LSTMPredictor(
                locality=locality,
                sequence_length=24,  # Use 24 hours of historical data
                prediction_horizon=6  # Keep 6-hour prediction
            )
            
            # Load and prepare data
            df = predictor.load_cleaned_data()
            
            # Split data into train and test
            train_size = int(len(df) * 0.8)
            train_data = df[:train_size]
            test_data = df[train_size:]
            
            # Train with adjusted parameters
            history = predictor.train_model(
                train_data,
                epochs=150,  # Increase epochs
                batch_size=32,  # Increase batch size
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
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM model for PM2.5 prediction')
    parser.add_argument('--locality', type=str, default='Worli', help='Locality for prediction')
    parser.add_argument('--sequence_length', type=int, default=24, help='Sequence length for LSTM')
    parser.add_argument('--prediction_horizon', type=int, default=6, help='Prediction horizon')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # Create and train LSTM model
        predictor = LSTMPredictor(
            locality=args.locality,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon
        )
        
        # Load data
        data = predictor.load_cleaned_data()
        
        # Split data into train and test sets (80% train, 20% test)
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        # Train model
        if os.path.exists(f'models/lstm_{args.locality}.keras'):
            logging.info(f"Loading existing model for {args.locality}")
            predictor.model = tf.keras.models.load_model(f'models/lstm_{args.locality}.keras', 
                                        custom_objects={'AttentionLayer': AttentionLayer})
        else:
            # Train with adjusted parameters
            history = predictor.train_model(
                train_data,
                epochs=150,  # Increase epochs
                batch_size=32,  # Adjust batch size
                validation_split=0.2
            )
        
        # Evaluate on test data
        results = predictor.evaluate_predictions(test_data)
        
        # Print final results
        logging.info("Final Evaluation Results:")
        logging.info(f"MAE: {results['mae']:.2f}")
        logging.info(f"RMSE: {results['rmse']:.2f}")
        
        # Calculate total execution time
        logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        traceback.print_exc()