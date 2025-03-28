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
import json
from geopy.distance import geodesic
import matplotlib
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lstm_spatial.log"),
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
        self.location_coordinates = self.load_location_coordinates()
        self.nearby_localities = self.find_nearby_localities()
        
        logging.info(f"Initializing LSTM Predictor for {locality}")
        logging.info(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}")
        
        # Create directories for saving models and predictions
        os.makedirs('models', exist_ok=True)
        os.makedirs('predictions', exist_ok=True)

    def load_location_coordinates(self):
        """
        Load location coordinates for all localities
        Returns:
            dict: Dictionary mapping locality names to coordinates
        """
        try:
            with open('location_coordinates.json', 'r') as f:
                coordinates = json.load(f)
            logging.info(f"Loaded coordinates for {len(coordinates)} localities")
            return coordinates
        except Exception as e:
            logging.error(f"Error loading location coordinates: {str(e)}")
            return {}
            
    def calculate_distance(self, locality1, locality2):
        """
        Calculate distance between two localities
        Args:
            locality1 (str): Name of first locality
            locality2 (str): Name of second locality
        Returns:
            float: Distance in kilometers
        """
        if locality1 not in self.location_coordinates or locality2 not in self.location_coordinates:
            return float('inf')
            
        coord1 = self.location_coordinates[locality1]
        coord2 = self.location_coordinates[locality2]
        
        return geodesic(coord1, coord2).kilometers
        
    def calculate_weight(self, distance, decay_factor=0.5):
        """
        Calculate weight based on distance using exponential decay
        Args:
            distance (float): Distance in kilometers
            decay_factor (float): Decay factor for exponential weighting
        Returns:
            float: Weight
        """
        return np.exp(-decay_factor * distance)
        
    def find_nearby_localities(self, max_distance=10.0):
        """
        Find nearby localities within a certain distance
        Args:
            max_distance (float): Maximum distance in kilometers
        Returns:
            list: List of tuples (locality, distance, weight)
        """
        if self.locality not in self.location_coordinates:
            logging.warning(f"No coordinates found for {self.locality}")
            return []
            
        nearby = []
        for other_locality in self.location_coordinates:
            if other_locality == self.locality:
                continue
                
            distance = self.calculate_distance(self.locality, other_locality)
            if distance <= max_distance:
                weight = self.calculate_weight(distance)
                nearby.append((other_locality, distance, weight))
                
        nearby.sort(key=lambda x: x[1])  # Sort by distance
        logging.info(f"Found {len(nearby)} nearby localities for {self.locality}")
        for locality, distance, weight in nearby:
            logging.info(f"  - {locality}: {distance:.2f} km, weight: {weight:.4f}")
            
        return nearby

    def load_cleaned_data(self):
        """
        Load cleaned data for the locality from the spatial_cleaned_data directory
        Returns:
            pd.DataFrame: Cleaned data for the locality
        """
        # Look for files matching the locality name in the spatial_cleaned_data directory
        cleaned_files = glob(os.path.join('spatial_cleaned_data', f'{self.locality}_spatial_weighted.csv'))
        
        if not cleaned_files:
            # Fall back to the original cleaned_data directory if spatial data not found
            logging.warning(f"No spatial weighted data found for locality {self.locality}, checking regular cleaned data")
            cleaned_files = glob(os.path.join('cleaned_data', f'{self.locality}_spatial_average.csv'))
            
            if not cleaned_files:
                raise ValueError(f"No cleaned data found for locality {self.locality}")
        
        data_path = cleaned_files[0]
        logging.info(f"Loading cleaned data from {data_path}")
        
        # Load the data
        df = pd.read_csv(data_path)
        
        # Convert date column to datetime index
        date_col = None
        for col in ['From Date', 'date', 'Date', 'timestamp', 'Timestamp']:
            if col in df.columns:
                date_col = col
                break
                
        if date_col is None:
            # If no date column is found, assume the first column is the index
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Instead of dropping rows with NaN values, use spatial data to fill them
        if self.nearby_localities and 'PM2.5' in df.columns:
            logging.info(f"Using spatial data from nearby localities to fill any remaining missing values")
            df = self.fill_missing_with_spatial_data(df)
        else:
            # If no nearby localities or PM2.5 not in columns, fall back to dropping NaNs
            if df.isna().any().any():
                logging.warning(f"Data still contains NaN values. Filling with forward and backward fill.")
                df = df.fillna(method='ffill').fillna(method='bfill')
        
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
        
    def load_locality_data(self, locality):
        """
        Load data for a specific locality from the spatial_cleaned_data directory
        Args:
            locality (str): Name of the locality
        Returns:
            pd.DataFrame: Data for the locality
        """
        # First try to load from spatial_cleaned_data
        cleaned_files = glob(os.path.join('spatial_cleaned_data', f'{locality}_spatial_weighted.csv'))
        
        if not cleaned_files:
            # Fall back to the original cleaned_data directory
            cleaned_files = glob(os.path.join('cleaned_data', f'{locality}_spatial_average.csv'))
            
            if not cleaned_files:
                logging.warning(f"No data found for locality {locality}")
                return None
        
        data_path = cleaned_files[0]
        
        try:
            # Load the data
            df = pd.read_csv(data_path)
            
            # Convert date column to datetime index
            date_col = None
            for col in ['From Date', 'date', 'Date', 'timestamp', 'Timestamp']:
                if col in df.columns:
                    date_col = col
                    break
                    
            if date_col is None:
                # If no date column is found, assume the first column is the index
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            else:
                df[date_col] = pd.to_datetime(df[date_col])
                df.set_index(date_col, inplace=True)
            
            return df
        except Exception as e:
            logging.error(f"Error loading data for locality {locality}: {str(e)}")
            return None

    def fill_missing_with_spatial_data(self, df):
        """
        Fill missing values using spatial data from nearby localities
        Args:
            df (pd.DataFrame): DataFrame with missing values
        Returns:
            pd.DataFrame: DataFrame with filled values
        """
        # Make a copy to avoid modifying the original
        df_filled = df.copy()
        
        # Find rows with missing PM2.5 values
        missing_mask = df_filled['PM2.5'].isna()
        missing_dates = df_filled.index[missing_mask]
        
        if len(missing_dates) == 0:
            logging.info("No missing PM2.5 values found")
            return df_filled
            
        logging.info(f"Found {len(missing_dates)} timestamps with missing PM2.5 values")
        
        # Load data from nearby localities
        nearby_data = {}
        for locality, distance, weight in self.nearby_localities:
            locality_df = self.load_locality_data(locality)
            if locality_df is not None and 'PM2.5' in locality_df.columns:
                nearby_data[locality] = (locality_df, weight)
                
        if not nearby_data:
            logging.warning("No usable data from nearby localities")
            return df_filled.dropna()
            
        # Fill missing values with weighted averages from nearby localities
        filled_count = 0
        
        for date in missing_dates:
            weighted_values = []
            total_weight = 0
            
            for locality, (locality_df, weight) in nearby_data.items():
                if date in locality_df.index and not pd.isna(locality_df.loc[date, 'PM2.5']):
                    value = locality_df.loc[date, 'PM2.5']
                    weighted_values.append(value * weight)
                    total_weight += weight
                    
                    # Add locality-specific weighted features
                    col_name = f'PM2.5_{locality}_weighted'
                    df_filled.loc[date, col_name] = value * weight
            
            if total_weight > 0:
                # Calculate weighted average
                weighted_avg = sum(weighted_values) / total_weight
                df_filled.loc[date, 'PM2.5'] = weighted_avg
                df_filled.loc[date, 'PM2.5_spatial_avg'] = weighted_avg
                filled_count += 1
                
        # Fill any remaining missing values using time-based methods
        df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
        
        logging.info(f"Filled {filled_count} missing values using spatial data")
        logging.info(f"Final data shape after filling: {df_filled.shape}")
        
        return df_filled
        
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
        
        # Add spatial features if available
        spatial_features = [col for col in data.columns if col.startswith('PM2.5_') and col != 'PM2.5']
        if spatial_features:
            logging.info(f"Found {len(spatial_features)} spatial features: {spatial_features}")
            
            # Create aggregate spatial features
            if 'PM2.5_spatial_avg' in data.columns:
                # Calculate difference between local and spatial average
                data['spatial_diff'] = data['PM2.5'] - data['PM2.5_spatial_avg']
                
                # Calculate ratio between local and spatial average (capped to avoid extreme values)
                data['spatial_ratio'] = data['PM2.5'] / data['PM2.5_spatial_avg'].replace(0, np.nan)
                data['spatial_ratio'] = data['spatial_ratio'].clip(0.1, 10).fillna(1.0)
                
                # Add lagged spatial features
                data['spatial_avg_lag_1h'] = data['PM2.5_spatial_avg'].shift(1)
                data['spatial_avg_lag_3h'] = data['PM2.5_spatial_avg'].shift(3)
                
                # Add rate of change for spatial average
                data['spatial_avg_change_1h'] = data['PM2.5_spatial_avg'].diff(1)
        
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
    
    def build_model(self, input_shape, output_size=None):
        """
        Build stacked LSTM model with residual connections
        Args:
            input_shape (tuple): Shape of input sequences (sequence_length, n_features)
            output_size (int): Size of output layer (prediction horizon), defaults to self.prediction_horizon
        """
        if output_size is None:
            output_size = self.prediction_horizon
            
        logging.info(f"Building stacked LSTM model with residual connections, input shape: {input_shape}, output size: {output_size}")
        
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
        outputs = Dense(output_size)(x)
        
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
        
        try:
            # Prepare sequences
            X, y_true_scaled = self.prepare_sequences(test_data)
            
            # Make predictions with encoding error handling
            try:
                # Set environment variables to handle encoding issues
                os.environ['PYTHONIOENCODING'] = 'utf-8'
                
                # Disable verbose output from TensorFlow to avoid encoding issues
                y_pred_scaled = self.model.predict(X, verbose=0)
            except UnicodeEncodeError:
                logging.warning("Encountered encoding issue during prediction, trying alternative approach")
                # Alternative approach with redirected stdout
                import sys
                original_stdout = sys.stdout
                try:
                    # Redirect stdout to avoid encoding issues
                    sys.stdout = open(os.devnull, 'w')
                    y_pred_scaled = self.model.predict(X, verbose=0)
                finally:
                    sys.stdout = original_stdout
            
            # Handle prediction horizon mismatch
            model_output_size = y_pred_scaled.shape[1]
            if model_output_size != y_true_scaled.shape[1]:
                logging.warning(f"Prediction horizon mismatch: model outputs {model_output_size}, but true values have {y_true_scaled.shape[1]} steps")
                # Use the smaller of the two to avoid dimension mismatch
                min_horizon = min(model_output_size, y_true_scaled.shape[1])
                y_pred_scaled = y_pred_scaled[:, :min_horizon]
                y_true_scaled = y_true_scaled[:, :min_horizon]
                logging.info(f"Adjusted to use {min_horizon} steps for evaluation")
            
            # Inverse transform predictions and true values
            y_pred = self.inverse_transform_predictions(y_pred_scaled)
            y_true = self.inverse_transform_predictions(y_true_scaled)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            
            logging.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
            
            # Use try-except to handle potential encoding issues in logging
            try:
                logging.info(f"Evaluation metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            except UnicodeEncodeError:
                # Fallback to a simpler message if encoding issues occur
                logging.info(f"Evaluation metrics calculated successfully")
            
            # Generate plots with error handling
            try:
                self.visualize_results({'predictions': y_pred, 'actual': y_true}, test_data)
            except Exception as e:
                logging.error(f"Error generating visualization: {str(e)}")
                # Continue execution even if visualization fails
            
            return {
                'predictions': y_pred,
                'actual': y_true,
                'mae': mae,
                'rmse': rmse
            }
        except Exception as e:
            logging.error(f"Error in evaluate_predictions: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    
    def visualize_results(self, results, test_data):
        """
        Visualize prediction results
        Args:
            results (dict): Dictionary with prediction results
            test_data (pd.DataFrame): Test data
        """
        try:
            # Create directory for plots if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Set a non-interactive backend to avoid display issues
            matplotlib.use('Agg')
            
            # Plot actual vs predicted
            plt.figure(figsize=(12, 6))
            plt.plot(results['actual'], label='Actual')
            plt.plot(results['predictions'], label='Predicted')
            
            # Use try-except for title setting to handle potential encoding issues
            try:
                plt.title(f'PM2.5 Prediction Results for {self.locality}')
            except UnicodeEncodeError:
                plt.title('PM2.5 Prediction Results')
                
            plt.xlabel('Time Steps')
            plt.ylabel('PM2.5')
            plt.legend()
            
            # Save figure with error handling
            try:
                plt.savefig(f'plots/distance_{self.locality}_predictions.png')
            except Exception as e:
                logging.error(f"Error saving prediction plot: {str(e)}")
                # Use a sanitized filename as fallback
                safe_locality = self.locality.encode('ascii', errors='replace').decode('ascii')
                plt.savefig(f'plots/distance_{safe_locality}_predictions.png')
            
            plt.close()
            
            # Plot error distribution
            plt.figure(figsize=(10, 6))
            error = results['predictions'] - results['actual']
            error_flat = error.flatten()
            sns.histplot(error_flat, kde=True)
            
            # Use try-except for title setting to handle potential encoding issues
            try:
                plt.title(f'Prediction Error Distribution for {self.locality}')
            except UnicodeEncodeError:
                plt.title('Prediction Error Distribution')
            
            plt.xlabel('Prediction Error (Predicted - Actual)')
            plt.ylabel('Frequency')
            
            # Save figure with error handling
            try:
                plt.savefig(f'plots/distance_{self.locality}_error_distribution.png')
            except Exception as e:
                logging.error(f"Error saving error distribution plot: {str(e)}")
                # Use a sanitized filename as fallback
                safe_locality = self.locality.encode('ascii', errors='replace').decode('ascii')
                plt.savefig(f'plots/distance_{safe_locality}_error_distribution.png')
            
            plt.close()
            
            # Log success message with error handling
            try:
                logging.info(f"Plots saved to plots/distance_{self.locality}_predictions.png and plots/distance_{self.locality}_error_distribution.png")
            except UnicodeEncodeError:
                logging.info("Plots saved successfully")
                
        except Exception as e:
            logging.error(f"Error in visualize_results: {str(e)}")
            logging.error(traceback.format_exc())
            
    def plot_predictions(self, results, date):
        """
        Plot predictions vs actual values
        Args:
            results (dict): Dictionary containing predictions and actual values
            date (datetime): Date of prediction
        """
        try:
            logging.info(f"Generating prediction plots for {self.locality}")
            
            # Set a non-interactive backend to avoid display issues
            matplotlib.use('Agg')
            
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
            
            # Use try-except for title setting to handle potential encoding issues
            try:
                plt.title(f'PM2.5 Predictions vs Actual Values for {self.locality}')
            except UnicodeEncodeError:
                plt.title('PM2.5 Predictions vs Actual Values')
                
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
            
            # Format date for filename
            date_str = date.strftime('%Y%m%d')
            
            # Save figure with error handling
            try:
                plt.savefig(f'plots/{self.locality}_{date_str}_prediction.png')
                logging.info(f"Prediction plot saved to plots/{self.locality}_{date_str}_prediction.png")
            except Exception as e:
                logging.error(f"Error saving prediction plot: {str(e)}")
                # Use a sanitized filename as fallback
                safe_locality = self.locality.encode('ascii', errors='replace').decode('ascii')
                plt.savefig(f'plots/{safe_locality}_{date_str}_prediction.png')
                logging.info(f"Prediction plot saved with sanitized filename")
                
            plt.close()
        except Exception as e:
            logging.error(f"Error in plot_predictions: {str(e)}")
            logging.error(traceback.format_exc())
            
def main():
    """
    Main function to run LSTM predictor
    """
    start_time = time.time()
    logging.info("Starting LSTM predictor main execution")
    
    # Check for spatially cleaned data first
    csv_files = glob(os.path.join('spatial_cleaned_data', '*.csv'))
    localities = []
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        if '_spatial_weighted' in filename:
            locality = filename.split('_spatial_weighted')[0]
            localities.append(locality)
    
    # If no spatially cleaned data found, fall back to regular cleaned data
    if not localities:
        logging.warning("No spatially cleaned data found, falling back to regular cleaned data")
        csv_files = glob(os.path.join('cleaned_data', '*.csv'))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            if '_spatial_average' in filename:
                locality = filename.split('_spatial_average')[0]
                localities.append(locality)
    
    logging.info(f"Found {len(localities)} localities: {localities}")
    
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("lstm_spatial.log"),
            logging.StreamHandler()
        ]
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LSTM model for PM2.5 prediction with spatial data')
    parser.add_argument('--locality', type=str, default='Airport', help='Locality to predict for')
    parser.add_argument('--sequence_length', type=int, default=24, help='Number of past time steps to use')
    parser.add_argument('--prediction_horizon', type=int, default=24, help='Number of future time steps to predict')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], default='train', 
                        help='Mode to run in (train, evaluate, or predict)')
    parser.add_argument('--decay_factor', type=float, default=0.5, 
                        help='Decay factor for spatial weighting (higher means faster decay with distance)')
    parser.add_argument('--max_distance', type=float, default=10.0, 
                        help='Maximum distance (km) to consider for nearby localities')
    parser.add_argument('--use_spatial_data', action='store_true',
                        help='Use spatially cleaned data from spatial_cleaned_data directory')
    
    args = parser.parse_args()
    
    # Create LSTM predictor
    predictor = LSTMPredictor(
        locality=args.locality,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon
    )
    
    # Log spatial data information
    if args.use_spatial_data:
        logging.info("Using spatially cleaned data from spatial_cleaned_data directory")
    logging.info(f"Using spatial data with decay factor {args.decay_factor} and max distance {args.max_distance} km")
    
    # Run in specified mode
    if args.mode == 'train':
        logging.info("Running in training mode")
        # Load data and split into train/test
        data = predictor.load_cleaned_data()
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        predictor.train_model(train_data, epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == 'evaluate':
        logging.info("Running in evaluation mode")
        # Load data and use test portion
        data = predictor.load_cleaned_data()
        train_size = int(len(data) * 0.8)
        test_data = data[train_size:]
        predictor.evaluate_predictions(test_data)
    elif args.mode == 'predict':
        logging.info("Running in prediction mode")
        # Load most recent data for prediction
        data = predictor.load_cleaned_data()
        predictor.predict_next_days(data)
    else:
        logging.error(f"Unknown mode: {args.mode}")
        sys.exit(1)