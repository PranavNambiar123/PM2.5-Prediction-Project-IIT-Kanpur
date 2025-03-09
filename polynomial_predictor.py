import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from glob import glob
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil
import gc
import warnings
warnings.filterwarnings('ignore')

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('polynomial_training.log', mode='w'),
        logging.StreamHandler()
    ]
)

class MonitoredElasticNetCV(ElasticNetCV):
    """Custom ElasticNetCV with progress monitoring"""
    def __init__(self, *args, horizon_idx=None, **kwargs):
        self.horizon_idx = horizon_idx
        super().__init__(*args, **kwargs)
    
    def _fit_path(self, X, y, alphas, l1_ratio, eps, n_alphas, cv, max_iter, tol):
        self.iter_count = getattr(self, 'iter_count', 0)
        self.start_time = getattr(self, 'start_time', time.time())
        self.last_log_time = getattr(self, 'last_log_time', time.time())
        
        horizon_str = f" (Horizon t+{self.horizon_idx+1})" if self.horizon_idx is not None else ""
        
        # Log start of training
        if self.iter_count == 0:
            logging.info(f"Starting training{horizon_str} - Alphas: {len(alphas)}, L1 ratios: {len(l1_ratio)}, Shape: {X.shape}")
        
        # Track convergence
        best_score = float('inf')
        n_not_improved = 0
        
        for i, (alpha, l1) in enumerate(zip(alphas, l1_ratio)):
            current_time = time.time()
            self.iter_count += 1
            
            # Log progress and check convergence
            if self.iter_count % 10 == 0:
                elapsed = current_time - self.start_time
                iter_time = current_time - self.last_log_time
                
                # Get current score if available
                current_score = None
                if hasattr(self, 'mse_path_') and self.mse_path_ is not None:
                    current_score = np.mean(self.mse_path_[-1])
                    if current_score < best_score:
                        best_score = current_score
                        n_not_improved = 0
                    else:
                        n_not_improved += 1
                
                msg = [
                    f"Horizon {self.horizon_idx+1}" if self.horizon_idx is not None else "Training",
                    f"Iter: {self.iter_count}",
                    f"Alpha: {i+1}/{len(alphas)}",
                    f"α={alpha:.6f}",
                    f"L1={l1:.3f}"
                ]
                
                if current_score is not None:
                    msg.extend([
                        f"MSE: {current_score:.4f}",
                        f"Best: {best_score:.4f}"
                    ])
                
                msg.extend([
                    f"Time: {iter_time:.1f}s",
                    f"Total: {elapsed:.1f}s"
                ])
                
                logging.info(" | ".join(msg))
                self.last_log_time = current_time
                
                # Early stopping if no improvement for a while
                if n_not_improved >= 5 and i > len(alphas) // 2:
                    logging.info(f"Early stopping{horizon_str} - No improvement for {n_not_improved} iterations")
                    break
        
        return super()._fit_path(X, y, alphas, l1_ratio, eps, n_alphas, cv, max_iter, tol)

class PolynomialPredictor:
    def __init__(self, locality='Airport', sequence_length=24, prediction_horizon=24, degree=2, batch_size=64):  # Very small batch size to minimize memory usage
        self.locality = locality
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.degree = degree
        self.batch_size = batch_size
        self.model = None
        self.feature_scalers = {}
        self.feature_names = None
        
        logging.info(f"Initializing Polynomial Predictor for {locality}")
        logging.info(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}")
        logging.info(f"Polynomial degree: {degree}, Batch size: {batch_size}")
        
        os.makedirs('models', exist_ok=True)
        os.makedirs('predictions', exist_ok=True)

    def load_cleaned_data(self):
        cleaned_files = glob(os.path.join('cleaned_data', f'{self.locality}_spatial_average.csv'))
        
        if not cleaned_files:
            raise ValueError(f"No spatial average cleaned data found for locality {self.locality}")
        
        data_path = cleaned_files[0]
        logging.info(f"Loading cleaned data from {data_path}")
        
        # Read data in chunks to reduce memory usage
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=1000):
            chunks.append(chunk)
            cleanup_memory()
        
        df = pd.concat(chunks)
        del chunks
        cleanup_memory()
        
        # Convert to datetime and set index
        df['From Date'] = pd.to_datetime(df['From Date'])
        df.set_index('From Date', inplace=True)
        
        # Sample data to reduce size (take every 3rd row)
        df = df.iloc[::3].copy()
        
        # Drop NA and select features
        df = df.dropna()
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'PM2.5']
        
        if not feature_cols:
            raise ValueError(f"No numeric features found in columns: {df.columns}")
        
        df = df[['PM2.5'] + feature_cols]
        self.feature_names = feature_cols
        
        logging.info(f"Loaded and sampled data with shape: {df.shape}")
        return df

    def prepare_sequences(self, data, is_training=True):
        start_time = time.time()
        logging.info(f"Preparing sequences for {'training' if is_training else 'prediction'}")
        logging.info(f"Input data shape: {data.shape}")
        
        # Process in chunks to reduce memory usage
        chunk_size = 1000
        n_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        processed_chunks = []
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data))
            
            chunk = data[start_idx:end_idx].copy()
            
            # Minimal feature set
            chunk_pm25 = chunk['PM2.5'].values
            
            # Time features (minimal set)
            chunk['hour_sin'] = np.sin(2 * np.pi * chunk.index.hour / 24)
            
            # Simple moving average only
            chunk['rolling_mean_6h'] = pd.Series(chunk_pm25).rolling(
                window=6, min_periods=1).mean().values
            
            # Clean up
            chunk = chunk.fillna(method='ffill').fillna(0)
            chunk['PM2.5'] = chunk['PM2.5'].clip(lower=0.1)
            
            processed_chunks.append(chunk)
            del chunk
            gc.collect()
        
        # Combine chunks
        data = pd.concat(processed_chunks)
        del processed_chunks
        gc.collect()
        
        feature_cols = [col for col in data.columns if col != 'PM2.5']
        X_raw = data[feature_cols].values
        y_raw = data['PM2.5'].values
        
        if is_training or 'features' not in self.feature_scalers:
            self.feature_scalers['features'] = StandardScaler().fit(X_raw)
        X_scaled = self.feature_scalers['features'].transform(X_raw)
        
        if is_training or 'PM2.5' not in self.feature_scalers:
            self.feature_scalers['PM2.5'] = StandardScaler().fit(y_raw.reshape(-1, 1))
        y_scaled = self.feature_scalers['PM2.5'].transform(y_raw.reshape(-1, 1))
        
        # Create sequences efficiently using numpy operations
        n_samples = len(data) - self.sequence_length - self.prediction_horizon + 1
        n_features = X_scaled.shape[1]
        
        X = np.zeros((n_samples, self.sequence_length * n_features))
        y = np.zeros((n_samples, self.prediction_horizon))
        
        for i in range(n_samples):
            X[i] = X_scaled[i:i + self.sequence_length].flatten()
            y[i] = y_scaled[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon].flatten()
        
        processing_time = time.time() - start_time
        logging.info(f"Sequence preparation completed in {processing_time:.2f}s")
        logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y

    def train_horizon_model(self, X, y, horizon_idx):
        """Train model for a specific prediction horizon"""
        start_time = time.time()
        
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('regressor', MonitoredElasticNetCV(
                cv=5,
                n_alphas=20,
                random_state=42,
                max_iter=10000,
                tol=1e-2,
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                selection='random',
                eps=1e-2,
                n_jobs=1,
                positive=True,
                verbose=True,
                horizon_idx=horizon_idx  # Pass horizon index for monitoring
            ))
        ])
        
        # Train on batches to handle large datasets
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        logging.info(f"Training on {n_batches} batches with batch size {self.batch_size}")
        
        for batch in range(n_batches):
            batch_start_time = time.time()
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx, horizon_idx]
            
            if batch == 0:
                logging.info(f"Fitting initial model on first batch ({end_idx-start_idx} samples)")
                model.fit(X_batch, y_batch)
            else:
                logging.info(f"Updating model with batch {batch+1}/{n_batches} ({end_idx-start_idx} samples)")
                model.named_steps['regressor'].fit(
                    model.named_steps['poly'].transform(X_batch),
                    y_batch
                )
            
            batch_time = time.time() - batch_start_time
            logging.info(f"Batch {batch+1}/{n_batches} completed in {batch_time:.2f}s")
        
        # Log feature selection info
        n_features = len(model.named_steps['regressor'].coef_)
        n_selected = np.sum(model.named_steps['regressor'].coef_ != 0)
        best_alpha = model.named_steps['regressor'].alpha_
        logging.info(f"Model for horizon t+{horizon_idx+1} - Selected {n_selected}/{n_features} features")
        logging.info(f"Best alpha: {best_alpha:.6f}")
        
        total_time = time.time() - start_time
        logging.info(f"Horizon t+{horizon_idx+1} training completed in {total_time:.2f}s")
        
        return model

    def train_model(self, train_data):
        start_time = time.time()
        
        # Process data in smaller chunks
        chunk_size = len(train_data) // 4  # Split into 4 chunks
        models = [None] * self.prediction_horizon
        
        for chunk_start in range(0, len(train_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(train_data))
            logging.info(f"Processing chunk {chunk_start//chunk_size + 1}/4")
            
            X, y = self.prepare_sequences(train_data[chunk_start:chunk_end])
            
            for i in tqdm(range(self.prediction_horizon), desc="Training models"):
                if not monitor_resources():
                    logging.warning("Resource usage too high, pausing for 60 seconds...")
                    time.sleep(60)
                    cleanup_memory()
                
                # Train or update model
                if models[i] is None:
                    models[i] = self.train_horizon_model(X, y, i)
                else:
                    # Update existing model with new chunk
                    models[i].named_steps['regressor'].fit(
                        models[i].named_steps['poly'].transform(
                            models[i].named_steps['scaler'].transform(X)
                        ),
                        y[:, i]
                    )
                
                cleanup_memory()
            
            # Clear chunk data
            del X, y
            cleanup_memory()
            time.sleep(10)  # Cool down between chunks
        
        self.model = models
        
        # Calculate metrics
        train_predictions = np.column_stack([model.predict(X) for model in self.model])
        train_predictions = self.feature_scalers['PM2.5'].inverse_transform(train_predictions)
        train_actual = self.feature_scalers['PM2.5'].inverse_transform(y)
        
        train_mae = np.mean(np.abs(train_actual - train_predictions))
        train_rmse = np.sqrt(np.mean((train_actual - train_predictions)**2))
        
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds")
        logging.info(f"Training MAE: {train_mae:.2f}")
        logging.info(f"Training RMSE: {train_rmse:.2f}")
        
        return {'mae': train_mae, 'rmse': train_rmse}

    def predict_next_days(self, current_data):
        X, _ = self.prepare_sequences(current_data, is_training=False)
        last_sequence = X[-1:]
        
        scaled_predictions = np.column_stack([
            model.predict(last_sequence) for model in self.model
        ])
        predictions = self.feature_scalers['PM2.5'].inverse_transform(scaled_predictions)
        
        return predictions.flatten()

    def evaluate_predictions(self, test_data):
        X_test, y_test = self.prepare_sequences(test_data, is_training=False)
        
        scaled_predictions = np.column_stack([
            model.predict(X_test) for model in self.model
        ])
        
        predictions = self.feature_scalers['PM2.5'].inverse_transform(scaled_predictions)
        actual = self.feature_scalers['PM2.5'].inverse_transform(y_test)
        
        mae = np.mean(np.abs(actual - predictions))
        rmse = np.sqrt(np.mean((actual - predictions)**2))
        
        logging.info(f"Evaluation metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return {
            'predictions': predictions,
            'actual': actual,
            'mae': mae,
            'rmse': rmse
        }

    def plot_predictions(self, results, date):
        plt.figure(figsize=(15, 10))
        
        hours = np.arange(self.prediction_horizon)
        plt.plot(hours, results['actual'][0], 'b-', label='Actual', marker='o')
        plt.plot(hours, results['predictions'][0], 'r--', label='Predicted', marker='x')
        
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
        
        plt.text(0.02, 0.98,
                f'MAE: {results["mae"]:.2f}\nRMSE: {results["rmse"]:.2f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plot_path = f'predictions/polynomial_prediction_{self.locality}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 6))
        error_flat = error.flatten()
        sns.histplot(error_flat, kde=True)
        plt.title(f'Prediction Error Distribution for {self.locality}')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Count')
        
        error_plot_path = f'predictions/polynomial_error_distribution_{self.locality}.png'
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def monitor_resources():
    """Monitor system resources and return True if they're within safe limits."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > 80 or memory_percent > 80:
        logging.warning(f"Resource usage high - CPU: {cpu_percent}%, Memory: {memory_percent}%")
        return False
    return True

def cleanup_memory():
    """Force garbage collection and clear memory caches."""
    gc.collect()
    _ = psutil.Process().memory_info()

def main():
    start_time = time.time()
    logging.info("Starting Polynomial predictor main execution")
    logging.info("="*80)
    
    # Process one locality at a time to minimize memory usage
    csv_files = sorted(glob(os.path.join('cleaned_data', '*_spatial_average.csv')))
    localities = [os.path.basename(f).split('_')[0] for f in csv_files]
    
    # Start with just one locality for testing
    test_localities = localities[:1]  # Process only first locality
    logging.info(f"Testing with locality: {test_localities[0]}")
    logging.info("="*80)
    
    def process_locality(locality):
        try:
            if not monitor_resources():
                logging.warning(f"Skipping {locality} due to high resource usage")
                return None
                
            predictor = PolynomialPredictor(
                locality=locality,
                sequence_length=24,
                prediction_horizon=6,
                degree=2,  # Reduced from higher degrees
                batch_size=32  # Very small batch size to minimize memory spikes
            )
            
            df = predictor.load_cleaned_data()
            train_size = int(len(df) * 0.8)
            train_data = df[:train_size]
            test_data = df[train_size:]
            
            # Clear original dataframe to free memory
            del df
            cleanup_memory()
            
            history = predictor.train_model(train_data)
            cleanup_memory()
            
            results = predictor.evaluate_predictions(test_data)
            predictor.plot_predictions(results, test_data.index[0])
            
            # Clear predictor to free memory
            del predictor
            cleanup_memory()
            
            return results
            
        except Exception as e:
            logging.error(f"Error processing {locality}: {str(e)}")
            return None
    
    # Process test localities first
    results = []
    for locality in tqdm(test_localities, desc="Processing test localities"):
        if not monitor_resources():
            logging.warning("Resource usage too high, waiting 30 seconds...")
            time.sleep(30)
            cleanup_memory()
        
        result = process_locality(locality)
        if result is not None:
            results.append(result)
            
            # Log intermediate results after each locality
            logging.info(f"\nResults for {locality}:")
            logging.info(f"MAE: {result['mae']:.2f}")
            logging.info(f"RMSE: {result['rmse']:.2f}")
        
        # Aggressive cleanup between localities
        cleanup_memory()
        gc.collect()
        
        # Additional pause to let system cool down
        time.sleep(5)
    
    # Filter out None results from errors
    final_results = [r for r in results if r is not None]
    
    if final_results:
        avg_mae = np.mean([r['mae'] for r in final_results])
        avg_rmse = np.mean([r['rmse'] for r in final_results])
        logging.info(f"\nAverage Results across all localities:")
        logging.info(f"Average MAE: {avg_mae:.2f}")
        logging.info(f"Average RMSE: {avg_rmse:.2f}")
    
    total_execution_time = time.time() - start_time
    logging.info(f"Total execution time: {total_execution_time:.2f} seconds")

if __name__ == "__main__":
    main()