import os
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
from lstm_distance import LSTMPredictor, AttentionLayer
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_comparison.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def predict_and_compare(locality, sequence_length=24, prediction_horizon=5, test_samples=10):
    """
    Make predictions for specific time periods in the test set and compare with actual values
    
    Args:
        locality (str): Name of the locality to predict for
        sequence_length (int): Number of past time steps to use for prediction
        prediction_horizon (int): Number of future time steps to predict
        test_samples (int): Number of test samples to evaluate
        
    Returns:
        dict: Dictionary with prediction results and comparison metrics
    """
    logging.info(f"Running prediction and comparison for {locality}")
    logging.info(f"Parameters: sequence_length={sequence_length}, prediction_horizon={prediction_horizon}, test_samples={test_samples}")
    
    # Initialize predictor
    predictor = LSTMPredictor(
        locality=locality,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Load data
    data = predictor.load_cleaned_data()
    if data is None or len(data) == 0:
        raise ValueError(f"No data available for {locality}")
    
    logging.info(f"Loaded data with shape: {data.shape}")
    
    # Split data into train and test
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Check for model file
    model_path = f'models/lstm_{locality}_ph{prediction_horizon}.keras'
    if not os.path.exists(model_path):
        # Try alternative model path with different prediction horizon
        alt_model_paths = [f'models/lstm_{locality}_ph{h}.keras' for h in [6, 12, 24]]
        found_model = False
        
        for alt_path in alt_model_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                found_model = True
                logging.info(f"Found alternative model at {alt_path}")
                break
        
        if not found_model:
            logging.warning(f"No model found for {locality}, training new model")
            
            # Prepare sequences
            X_train, y_train = predictor.prepare_sequences(train_data)
            
            # Build and train model
            input_shape = (X_train.shape[1], X_train.shape[2])
            predictor.build_model(input_shape, output_size=prediction_horizon)
            
            predictor.train_model(
                train_data,
                epochs=100,
                batch_size=32,
                validation_split=0.2
            )
            
            # Save model
            if predictor.model is not None:
                predictor.model.save(model_path)
                logging.info(f"Model saved to {model_path}")
    
    # Load model
    logging.info(f"Loading model from {model_path}")
    try:
        # Set environment variables to handle encoding issues
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        predictor.model = tf.keras.models.load_model(
            model_path, 
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        logging.error(traceback.format_exc())
        raise
    
    # Prepare test sequences
    X_test, y_test_scaled = predictor.prepare_sequences(test_data)
    
    # Select random test samples
    if test_samples > len(X_test):
        test_samples = len(X_test)
        logging.warning(f"Adjusted test_samples to {test_samples} due to limited data")
    
    # Select evenly spaced samples from the test set
    sample_indices = np.linspace(0, len(X_test) - 1, test_samples, dtype=int)
    
    # Make predictions for each sample
    all_predictions = []
    all_actual = []
    
    for i, idx in enumerate(sample_indices):
        logging.info(f"Processing test sample {i+1}/{test_samples}")
        
        # Get input sequence
        input_sequence = X_test[idx:idx+1]
        
        # Get actual values
        actual_values = predictor.inverse_transform_predictions(y_test_scaled[idx:idx+1])
        
        # Make prediction
        try:
            # Disable verbose output from TensorFlow to avoid encoding issues
            predicted_scaled = predictor.model.predict(input_sequence, verbose=0)
            predicted_values = predictor.inverse_transform_predictions(predicted_scaled)
            
            all_predictions.append(predicted_values[0])
            all_actual.append(actual_values[0])
            
            logging.info(f"Sample {i+1} - Predicted: {predicted_values[0]}, Actual: {actual_values[0]}")
            
        except Exception as e:
            logging.error(f"Error making prediction for sample {i+1}: {str(e)}")
            continue
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actual = np.array(all_actual)
    
    # Calculate metrics for each horizon
    metrics = {}
    for h in range(prediction_horizon):
        horizon_pred = all_predictions[:, h]
        horizon_actual = all_actual[:, h]
        
        mae = mean_absolute_error(horizon_actual, horizon_pred)
        rmse = np.sqrt(mean_squared_error(horizon_actual, horizon_pred))
        r2 = r2_score(horizon_actual, horizon_pred)
        
        metrics[h+1] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        logging.info(f"Horizon {h+1} hour - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # Calculate overall metrics
    overall_mae = mean_absolute_error(all_actual.flatten(), all_predictions.flatten())
    overall_rmse = np.sqrt(mean_squared_error(all_actual.flatten(), all_predictions.flatten()))
    overall_r2 = r2_score(all_actual.flatten(), all_predictions.flatten())
    
    logging.info(f"Overall - MAE: {overall_mae:.2f}, RMSE: {overall_rmse:.2f}, R²: {overall_r2:.4f}")
    
    # Visualize results
    visualize_comparison(all_predictions, all_actual, metrics, locality, prediction_horizon)
    
    return {
        'predictions': all_predictions,
        'actual': all_actual,
        'metrics': metrics,
        'overall_metrics': {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'r2': overall_r2
        }
    }

def visualize_comparison(predictions, actual, metrics, locality, prediction_horizon):
    """
    Visualize comparison between predictions and actual values
    
    Args:
        predictions (np.array): Predicted values
        actual (np.array): Actual values
        metrics (dict): Metrics for each horizon
        locality (str): Name of the locality
        prediction_horizon (int): Number of future time steps predicted
    """
    # Create directory for saving plots
    os.makedirs('comparison_plots', exist_ok=True)
    
    # Plot metrics by horizon
    plt.figure(figsize=(12, 8))
    
    horizons = list(metrics.keys())
    mae_values = [metrics[h]['mae'] for h in horizons]
    rmse_values = [metrics[h]['rmse'] for h in horizons]
    r2_values = [metrics[h]['r2'] for h in horizons]
    
    plt.subplot(3, 1, 1)
    plt.plot(horizons, mae_values, marker='o', label='MAE')
    plt.title(f'Mean Absolute Error by Prediction Horizon - {locality}')
    plt.xlabel('Prediction Horizon (hours)')
    plt.ylabel('MAE')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(horizons, rmse_values, marker='s', label='RMSE', color='orange')
    plt.title(f'Root Mean Square Error by Prediction Horizon - {locality}')
    plt.xlabel('Prediction Horizon (hours)')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(horizons, r2_values, marker='^', label='R²', color='green')
    plt.title(f'R² Score by Prediction Horizon - {locality}')
    plt.xlabel('Prediction Horizon (hours)')
    plt.ylabel('R²')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'comparison_plots/{locality}_metrics_by_horizon.png')
    logging.info(f"Metrics plot saved to comparison_plots/{locality}_metrics_by_horizon.png")
    
    # Plot sample-by-sample comparison for each horizon
    for h in range(prediction_horizon):
        plt.figure(figsize=(14, 10))
        
        # Get predictions and actual values for this horizon
        horizon_pred = predictions[:, h]
        horizon_actual = actual[:, h]
        
        # Scatter plot
        plt.subplot(2, 1, 1)
        plt.scatter(horizon_actual, horizon_pred, alpha=0.7)
        
        # Add perfect prediction line
        max_val = max(np.max(horizon_actual), np.max(horizon_pred))
        min_val = min(np.min(horizon_actual), np.min(horizon_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        plt.title(f'{h+1}-hour Prediction vs Actual - {locality}')
        plt.xlabel('Actual PM2.5')
        plt.ylabel('Predicted PM2.5')
        plt.grid(True)
        
        # Add metrics to the plot
        mae = metrics[h+1]['mae']
        rmse = metrics[h+1]['rmse']
        r2 = metrics[h+1]['r2']
        
        plt.text(0.05, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}',
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Line plot
        plt.subplot(2, 1, 2)
        x = np.arange(len(horizon_actual))
        plt.plot(x, horizon_actual, 'b-', label='Actual', alpha=0.7)
        plt.plot(x, horizon_pred, 'r-', label='Predicted', alpha=0.7)
        
        plt.title(f'{h+1}-hour Prediction and Actual Values - {locality}')
        plt.xlabel('Sample Index')
        plt.ylabel('PM2.5')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'comparison_plots/{locality}_{h+1}hour_prediction.png')
        logging.info(f"{h+1}-hour prediction plot saved to comparison_plots/{locality}_{h+1}hour_prediction.png")
    
    # Create a combined plot for all horizons
    plt.figure(figsize=(15, 10))
    
    # Sample a subset of predictions for clarity
    sample_size = min(20, len(predictions))
    sample_indices = np.linspace(0, len(predictions) - 1, sample_size, dtype=int)
    
    for h in range(prediction_horizon):
        plt.subplot(prediction_horizon, 1, h+1)
        
        horizon_pred = predictions[sample_indices, h]
        horizon_actual = actual[sample_indices, h]
        
        x = np.arange(len(horizon_actual))
        plt.plot(x, horizon_actual, 'b-', label='Actual', alpha=0.7)
        plt.plot(x, horizon_pred, 'r-', label='Predicted', alpha=0.7)
        
        plt.title(f'{h+1}-hour Prediction - MAE: {metrics[h+1]["mae"]:.2f}, RMSE: {metrics[h+1]["rmse"]:.2f}')
        plt.ylabel('PM2.5')
        plt.grid(True)
        
        if h == 0:
            plt.legend()
        
        if h == prediction_horizon - 1:
            plt.xlabel('Sample Index')
    
    plt.tight_layout()
    plt.savefig(f'comparison_plots/{locality}_all_horizons.png')
    logging.info(f"Combined plot saved to comparison_plots/{locality}_all_horizons.png")

def main():
    parser = argparse.ArgumentParser(description='Predict and compare PM2.5 values')
    parser.add_argument('--locality', type=str, default='Airport', help='Locality to predict for')
    parser.add_argument('--sequence_length', type=int, default=24, help='Number of past time steps to use')
    parser.add_argument('--prediction_horizon', type=int, default=5, help='Number of future time steps to predict')
    parser.add_argument('--test_samples', type=int, default=50, help='Number of test samples to evaluate')
    
    args = parser.parse_args()
    
    try:
        results = predict_and_compare(
            locality=args.locality,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            test_samples=args.test_samples
        )
        
        # Print summary table
        print("\nPrediction Comparison Summary:")
        print("=" * 60)
        print(f"Locality: {args.locality}")
        print("-" * 60)
        print(f"{'Horizon (hours)':<15} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
        print("-" * 60)
        
        for h in range(1, args.prediction_horizon + 1):
            print(f"{h:<15} {results['metrics'][h]['mae']:<10.2f} {results['metrics'][h]['rmse']:<10.2f} {results['metrics'][h]['r2']:<10.4f}")
        
        print("-" * 60)
        print(f"{'Overall':<15} {results['overall_metrics']['mae']:<10.2f} {results['overall_metrics']['rmse']:<10.2f} {results['overall_metrics']['r2']:<10.4f}")
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Error in prediction and comparison: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
