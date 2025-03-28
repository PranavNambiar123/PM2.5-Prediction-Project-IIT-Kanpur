import os
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from lstm_distance import LSTMPredictor, AttentionLayer
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def run_prediction(locality, sequence_length=24, prediction_horizon=5, decay_factor=0.5, max_distance=10.0):
    """
    Run prediction for a specific locality
    
    Args:
        locality (str): Name of the locality to predict for
        sequence_length (int): Number of past time steps to use for prediction
        prediction_horizon (int): Number of future time steps to predict
        decay_factor (float): Decay factor for spatial weighting
        max_distance (float): Maximum distance to consider for nearby localities
    """
    logging.info(f"Running prediction for {locality}")
    logging.info(f"Parameters: sequence_length={sequence_length}, prediction_horizon={prediction_horizon}")
    logging.info(f"Spatial parameters: decay_factor={decay_factor}, max_distance={max_distance}")
    
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
    
    # Make predictions
    try:
        # Get the most recent data for prediction
        recent_data = data.iloc[-sequence_length:]
        logging.info(f"Using most recent {len(recent_data)} data points for prediction")
        
        # Make predictions
        predictions = predictor.predict_next_days(data)
        
        # Get the current timestamp
        current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create time points for future predictions
        future_times = []
        for i in range(prediction_horizon):
            future_time = pd.Timestamp.now() + pd.Timedelta(hours=i+1)
            future_times.append(future_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        # Create a DataFrame with predictions
        pred_df = pd.DataFrame({
            'Timestamp': future_times,
            'PM2.5_Prediction': predictions
        })
        
        # Save predictions to CSV
        os.makedirs('predictions', exist_ok=True)
        pred_file = f'predictions/{locality}_predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        pred_df.to_csv(pred_file, index=False)
        logging.info(f"Predictions saved to {pred_file}")
        
        # Print predictions
        logging.info("\nPredictions for the next hours:")
        logging.info("-" * 50)
        logging.info(f"{'Time':<20} {'PM2.5 Prediction':<15}")
        logging.info("-" * 50)
        
        for i, (time_str, pred) in enumerate(zip(future_times, predictions)):
            logging.info(f"{time_str:<20} {pred:<15.2f}")
        
        logging.info("-" * 50)
        
        # Visualize predictions
        plt.figure(figsize=(12, 6))
        plt.plot(future_times, predictions, marker='o', linestyle='-', linewidth=2)
        plt.title(f'PM2.5 Predictions for {locality} - Next {prediction_horizon} Hours')
        plt.xlabel('Time')
        plt.ylabel('PM2.5')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_file = f'predictions/{locality}_predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_file)
        logging.info(f"Plot saved to {plot_file}")
        
        return pred_df
        
    except Exception as e:
        logging.error(f"Error making predictions: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description='Run PM2.5 predictions for a locality')
    parser.add_argument('--locality', type=str, default='Airport', help='Locality to predict for')
    parser.add_argument('--sequence_length', type=int, default=24, help='Number of past time steps to use')
    parser.add_argument('--prediction_horizon', type=int, default=5, help='Number of future time steps to predict')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='Decay factor for spatial weighting')
    parser.add_argument('--max_distance', type=float, default=10.0, help='Maximum distance for nearby localities')
    
    args = parser.parse_args()
    
    try:
        run_prediction(
            locality=args.locality,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            decay_factor=args.decay_factor,
            max_distance=args.max_distance
        )
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
