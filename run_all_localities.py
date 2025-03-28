import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import time
import json
from datetime import datetime
import traceback
import tensorflow as tf
from lstm_distance import LSTMPredictor, AttentionLayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("all_localities_prediction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def get_all_localities():
    """
    Get a list of all localities from the coordinates file
    
    Returns:
        list: List of locality names
    """
    try:
        # Try to load coordinates file
        coordinates_file = 'coordinates.json'
        if os.path.exists(coordinates_file):
            with open(coordinates_file, 'r') as f:
                coordinates = json.load(f)
            return list(coordinates.keys())
        
        # If coordinates file doesn't exist, try to find spatial data files
        spatial_data_dir = 'spatial_cleaned_data'
        if os.path.exists(spatial_data_dir):
            locality_files = [f for f in os.listdir(spatial_data_dir) if f.endswith('_spatial_weighted.csv')]
            localities = [f.split('_spatial_weighted.csv')[0] for f in locality_files]
            return localities
        
        # If neither exists, return a default list of localities
        return ['Airport', 'Andheri', 'Bandra', 'Borivali', 'Chembur', 'Colaba', 'Dadar', 
                'Dharavi', 'Juhu', 'Kurla', 'Malad', 'Mazagaon', 'Powai', 'Sion', 
                'Vileparle', 'Worli']
    
    except Exception as e:
        logging.error(f"Error getting localities: {str(e)}")
        return ['Airport', 'Bandra', 'Chembur', 'Dadar', 'Kurla']

def run_prediction_for_locality(locality, sequence_length=24, prediction_horizon=5, 
                               decay_factor=0.5, max_distance=10.0, generate_actual=True):
    """
    Run prediction for a specific locality
    
    Args:
        locality (str): Name of the locality to predict for
        sequence_length (int): Number of past time steps to use
        prediction_horizon (int): Number of future time steps to predict
        decay_factor (float): Decay factor for spatial weighting
        max_distance (float): Maximum distance to consider for nearby localities
        generate_actual (bool): Whether to generate simulated actual values
        
    Returns:
        tuple: (prediction_file, actual_file) paths to the generated files
    """
    logging.info(f"Running prediction for {locality}")
    logging.info(f"Parameters: sequence_length={sequence_length}, prediction_horizon={prediction_horizon}")
    
    try:
        # Initialize predictor
        predictor = LSTMPredictor(
            locality=locality,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        # Load data
        data = predictor.load_cleaned_data()
        if data is None or len(data) == 0:
            logging.warning(f"No data available for {locality}, skipping")
            return None, None
        
        logging.info(f"Loaded data with shape: {data.shape}")
        
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
                # Split data for training
                train_size = int(len(data) * 0.8)
                train_data = data[:train_size]
                
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
                    os.makedirs('models', exist_ok=True)
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
            logging.error(f"Error loading model for {locality}: {str(e)}")
            logging.error(traceback.format_exc())
            return None, None
        
        # Make predictions
        try:
            # Get the most recent data for prediction
            recent_data = data.iloc[-sequence_length:]
            logging.info(f"Using most recent {len(recent_data)} data points for prediction")
            
            # Make predictions
            predictions = predictor.predict_next_days(data)
            
            # Get the current timestamp
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create time points for future predictions
            future_times = []
            for i in range(prediction_horizon):
                future_time = datetime.now() + pd.Timedelta(hours=i+1)
                future_times.append(future_time.strftime('%Y-%m-%d %H:%M:%S'))
            
            # Create a DataFrame with predictions
            pred_df = pd.DataFrame({
                'Timestamp': future_times,
                'PM2.5_Prediction': predictions
            })
            
            # Save predictions to CSV
            os.makedirs('predictions', exist_ok=True)
            pred_file = f'predictions/{locality}_predictions_{current_time}.csv'
            pred_df.to_csv(pred_file, index=False)
            logging.info(f"Predictions saved to {pred_file}")
            
            # Generate simulated actual values if requested
            actual_file = None
            if generate_actual:
                # Create simulated actual values (for demonstration)
                # In a real scenario, these would be filled in later with actual measurements
                
                # Use test data to simulate actual values
                train_size = int(len(data) * 0.8)
                test_data = data[train_size:]
                
                # If test data is available, use it to create simulated actual values
                if len(test_data) > prediction_horizon:
                    # Get a random segment from test data
                    start_idx = np.random.randint(0, len(test_data) - prediction_horizon)
                    actual_values = test_data.iloc[start_idx:start_idx+prediction_horizon]['PM2.5'].values
                    
                    # Add some random noise to make it more realistic
                    noise = np.random.normal(0, 2, prediction_horizon)
                    actual_values = actual_values + noise
                    
                    # Ensure no negative values
                    actual_values = np.maximum(actual_values, 0)
                else:
                    # If not enough test data, create random values based on predictions
                    actual_values = predictions + np.random.normal(2, 3, prediction_horizon)
                    actual_values = np.maximum(actual_values, 0)
                
                # Create actual values DataFrame
                actual_df = pd.DataFrame({
                    'Timestamp': future_times,
                    'PM2.5_Actual': actual_values
                })
                
                # Save actual values to CSV
                os.makedirs('actual_values', exist_ok=True)
                actual_file = f'actual_values/{locality}_actual_{current_time}.csv'
                actual_df.to_csv(actual_file, index=False)
                logging.info(f"Simulated actual values saved to {actual_file}")
            
            return pred_file, actual_file
            
        except Exception as e:
            logging.error(f"Error making predictions for {locality}: {str(e)}")
            logging.error(traceback.format_exc())
            return None, None
            
    except Exception as e:
        logging.error(f"Error in prediction for {locality}: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None

def visualize_predictions(prediction_file, actual_file, output_dir='all_localities_plots'):
    """
    Visualize predictions and actual values
    
    Args:
        prediction_file (str): Path to prediction CSV file
        actual_file (str): Path to actual values CSV file
        output_dir (str): Directory to save plots
        
    Returns:
        str: Path to the generated plot
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load prediction data
        pred_df = pd.read_csv(prediction_file)
        pred_df['Timestamp'] = pd.to_datetime(pred_df['Timestamp'])
        
        # Extract locality from filename
        locality = os.path.basename(prediction_file).split('_predictions_')[0]
        
        # Load actual data
        actual_df = pd.read_csv(actual_file)
        actual_df['Timestamp'] = pd.to_datetime(actual_df['Timestamp'])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot predictions
        plt.plot(pred_df['Timestamp'], pred_df['PM2.5_Prediction'], 
                marker='o', linestyle='-', linewidth=2, label='Predicted PM2.5')
        
        # Plot actual values
        plt.plot(actual_df['Timestamp'], actual_df['PM2.5_Actual'], 
                marker='s', linestyle='-', linewidth=2, label='Actual PM2.5')
        
        # Calculate metrics
        merged_df = pd.merge(
            pred_df, actual_df, 
            on='Timestamp', 
            how='inner',
            suffixes=('_Prediction', '_Actual')
        )
        
        if not merged_df.empty:
            mae = np.mean(np.abs(merged_df['PM2.5_Prediction'] - merged_df['PM2.5_Actual']))
            rmse = np.sqrt(np.mean((merged_df['PM2.5_Prediction'] - merged_df['PM2.5_Actual']) ** 2))
            
            # Add metrics to plot
            plt.text(0.02, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set plot title and labels
        plt.title(f'PM2.5 Predictions for {locality} - Next 5 Hours')
        plt.xlabel('Time')
        plt.ylabel('PM2.5')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f'{output_dir}/{locality}_comparison_{current_time}.png'
        plt.savefig(plot_file)
        plt.close()
        
        logging.info(f"Plot saved to {plot_file}")
        
        return plot_file, mae, rmse
        
    except Exception as e:
        logging.error(f"Error visualizing predictions: {str(e)}")
        logging.error(traceback.format_exc())
        return None, None, None

def run_all_localities(sequence_length=24, prediction_horizon=5, decay_factor=0.5, max_distance=10.0):
    """
    Run predictions for all localities
    
    Args:
        sequence_length (int): Number of past time steps to use
        prediction_horizon (int): Number of future time steps to predict
        decay_factor (float): Decay factor for spatial weighting
        max_distance (float): Maximum distance to consider for nearby localities
        
    Returns:
        dict: Results for all localities
    """
    # Get all localities
    localities = get_all_localities()
    logging.info(f"Found {len(localities)} localities: {', '.join(localities)}")
    
    # Create results dictionary
    results = {}
    
    # Create summary DataFrame
    summary_data = []
    
    # Run predictions for each locality
    for i, locality in enumerate(localities):
        logging.info(f"Processing locality {i+1}/{len(localities)}: {locality}")
        
        try:
            # Run prediction
            pred_file, actual_file = run_prediction_for_locality(
                locality=locality,
                sequence_length=sequence_length,
                prediction_horizon=prediction_horizon,
                decay_factor=decay_factor,
                max_distance=max_distance
            )
            
            if pred_file is None or actual_file is None:
                logging.warning(f"Prediction failed for {locality}, skipping visualization")
                results[locality] = {
                    'status': 'failed',
                    'error': 'Prediction or actual file generation failed'
                }
                continue
            
            # Visualize predictions
            plot_file, mae, rmse = visualize_predictions(pred_file, actual_file)
            
            if plot_file is None:
                logging.warning(f"Visualization failed for {locality}")
                results[locality] = {
                    'status': 'partial',
                    'prediction_file': pred_file,
                    'actual_file': actual_file,
                    'error': 'Visualization failed'
                }
            else:
                results[locality] = {
                    'status': 'success',
                    'prediction_file': pred_file,
                    'actual_file': actual_file,
                    'plot_file': plot_file,
                    'mae': mae,
                    'rmse': rmse
                }
                
                # Add to summary data
                summary_data.append({
                    'Locality': locality,
                    'MAE': mae,
                    'RMSE': rmse
                })
            
        except Exception as e:
            logging.error(f"Error processing {locality}: {str(e)}")
            logging.error(traceback.format_exc())
            results[locality] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by MAE
        summary_df = summary_df.sort_values('MAE')
        
        # Save summary
        summary_file = f'all_localities_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        summary_df.to_csv(summary_file, index=False)
        logging.info(f"Summary saved to {summary_file}")
        
        # Create summary plot
        plt.figure(figsize=(14, 8))
        
        # Plot MAE by locality
        plt.subplot(2, 1, 1)
        plt.bar(summary_df['Locality'], summary_df['MAE'], color='skyblue')
        plt.title('Mean Absolute Error by Locality')
        plt.xlabel('Locality')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        # Plot RMSE by locality
        plt.subplot(2, 1, 2)
        plt.bar(summary_df['Locality'], summary_df['RMSE'], color='salmon')
        plt.title('Root Mean Square Error by Locality')
        plt.xlabel('Locality')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        
        plt.tight_layout()
        
        # Save summary plot
        summary_plot = f'all_localities_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(summary_plot)
        plt.close()
        
        logging.info(f"Summary plot saved to {summary_plot}")
        
        # Print summary table
        print("\nPrediction Results Summary:")
        print("=" * 60)
        print(f"{'Locality':<15} {'MAE':<10} {'RMSE':<10} {'Status':<10}")
        print("-" * 60)
        
        for locality in localities:
            if locality in results:
                if results[locality]['status'] == 'success':
                    print(f"{locality:<15} {results[locality]['mae']:<10.2f} {results[locality]['rmse']:<10.2f} {'✓':<10}")
                else:
                    print(f"{locality:<15} {'N/A':<10} {'N/A':<10} {'✗':<10}")
            else:
                print(f"{locality:<15} {'N/A':<10} {'N/A':<10} {'✗':<10}")
        
        print("-" * 60)
        print(f"Average MAE: {summary_df['MAE'].mean():.2f}, Average RMSE: {summary_df['RMSE'].mean():.2f}")
        print("=" * 60)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run predictions for all localities')
    parser.add_argument('--sequence_length', type=int, default=24, 
                        help='Number of past time steps to use')
    parser.add_argument('--prediction_horizon', type=int, default=5, 
                        help='Number of future time steps to predict')
    parser.add_argument('--decay_factor', type=float, default=0.5, 
                        help='Decay factor for spatial weighting')
    parser.add_argument('--max_distance', type=float, default=10.0, 
                        help='Maximum distance to consider for nearby localities')
    
    args = parser.parse_args()
    
    # Run predictions for all localities
    start_time = time.time()
    
    results = run_all_localities(
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        decay_factor=args.decay_factor,
        max_distance=args.max_distance
    )
    
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
