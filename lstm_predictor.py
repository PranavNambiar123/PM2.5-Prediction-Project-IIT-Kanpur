import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import logging
import time
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lstm_training.log'),
        logging.StreamHandler()
    ]
)

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
        Prepare sequences for LSTM training with enhanced feature engineering
        """
        start_time = time.time()
        logging.info(f"Preparing sequences from data with shape: {data.shape}")
        
        # Add time-based features
        data = data.copy()
        data['hour'] = data.index.hour.astype(float)
        data['day'] = data.index.day.astype(float)
        data['month'] = data.index.month.astype(float)
        
        # Add cyclical time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
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
        data = data.fillna(method='bfill')
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col != 'PM2.5']
        X_raw = data[feature_cols].values
        y_raw = data['PM2.5'].values
        
        # Use StandardScaler instead of MinMaxScaler
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
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, self.prediction_horizon)
        
        processing_time = time.time() - start_time
        logging.info(f"Sequence preparation completed. X shape: {X.shape}, y shape: {y.shape}")
        logging.info(f"Sequence preparation took {processing_time:.2f} seconds")
        
        return X, y
    
    def build_model(self, input_shape):
        """
        Build enhanced LSTM model with attention mechanism
        Args:
            input_shape (tuple): Shape of input sequences (sequence_length, n_features)
        """
        logging.info(f"Building LSTM model with attention, input shape: {input_shape}")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layer with return_sequences=True to get output for each timestep
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        
        # Self-attention mechanism
        attention = Attention()([lstm_out, lstm_out])
        
        # Concatenate LSTM output with attention output
        concat = Concatenate()([lstm_out, attention])
        
        # Final LSTM layer
        final_lstm = LSTM(32)(concat)
        
        # Dense layers with dropout for regularization
        x = Dense(64, activation='relu')(final_lstm)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # Output layer
        outputs = Dense(self.prediction_horizon)(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Use Adam optimizer with learning rate schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True)
        
        optimizer = Adam(learning_rate=lr_schedule)
        
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
        error_flat = error.flatten()
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
    Main function to demonstrate LSTM predictor usage
    """
    start_time = time.time()
    logging.info("Starting LSTM predictor main execution")
    
    csv_files = glob(os.path.join('cleaned_data', '*.csv'))
    localities = []
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        locality, method = filename.replace('.csv', '').split('_', 1)
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
                epochs=100,  # Reduce epochs but monitor convergence
                batch_size=64,  # Increase batch size
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