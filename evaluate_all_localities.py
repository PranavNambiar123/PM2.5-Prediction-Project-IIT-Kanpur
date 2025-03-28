import os
import sys
import pandas as pd
import numpy as np
import logging
import argparse
import subprocess
import json
import tensorflow as tf
from tqdm import tqdm
from tabulate import tabulate
from lstm_distance import LSTMPredictor, AttentionLayer
import io
import traceback

class ImmediateFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
    def emit(self, record):
        try:
            super().emit(record)
            self.flush()
        except UnicodeEncodeError:
            try:
                msg = self.format(record)
                msg = msg.encode('utf-8', errors='replace').decode('utf-8')
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception as e:
                safe_msg = f"Logging error (encoding issue): {str(e)}"
                try:
                    stream = self.stream
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                except:
                    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        ImmediateFileHandler("lstm_distance.log"),
        ImmediateFileHandler("ev_loc_training.log"),
        logging.StreamHandler()
    ]
)

# Custom print function to capture all terminal output
def custom_print(*args, **kwargs):
    output_stream = io.StringIO()
    print(*args, file=output_stream, **kwargs)
    output = output_stream.getvalue()
    
    try:
        logging.info(output.strip())
    except UnicodeEncodeError:
        safe_output = output.encode('utf-8', errors='replace').decode('utf-8')
        logging.info(safe_output.strip())
    
    print(*args, file=sys.__stdout__, **kwargs)

# Replace the built-in print function with our custom one
print = custom_print

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_available_localities():
    localities = []
    for file in os.listdir('cleaned_data'):
        if file.endswith('_spatial_average.csv'):
            locality = file.replace('_spatial_average.csv', '')
            localities.append(locality)
    return localities

def evaluate_locality(locality, sequence_length=24, prediction_horizon=6, decay_factor=0.5, max_distance=10.0):
    try:
        logging.info(f"Evaluating locality: {locality}")
        
        predictor = LSTMPredictor(
            locality=locality,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        data = predictor.load_cleaned_data()
        if data is None or len(data) == 0:
            raise ValueError(f"No data available for {locality}")
            
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
        
        model_path = f'models/lstm_{locality}_ph{prediction_horizon}.keras'
        if os.path.exists(model_path):
            logging.info(f"Loading existing model for {locality}")
            try:
                predictor.model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects={'AttentionLayer': AttentionLayer}
                )
            except Exception as e:
                logging.error(f"Error loading model for {locality}, will train new one: {str(e)}")
                predictor.model = None
        
        if predictor.model is None:
            logging.info(f"Training new model for {locality}")
            try:
                X_train, y_train = predictor.prepare_sequences(train_data)
                
                input_shape = (X_train.shape[1], X_train.shape[2])
                predictor.build_model(input_shape, output_size=prediction_horizon)
                
                history = predictor.train_model(
                    train_data,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2
                )
                
                if predictor.model is not None:
                    predictor.model.save(model_path)
                    logging.info(f"Model saved to {model_path}")
                else:
                    raise ValueError("Model training failed")
            except Exception as e:
                logging.error(f"Error training model for {locality}: {str(e)}")
                return {
                    'locality': locality,
                    'mae': None,
                    'rmse': None,
                    'mse': None,
                    'error': f"Training error: {str(e)}"
                }
        
        try:
            if predictor.model is None:
                raise ValueError("Model is None, cannot evaluate")
                
            results = predictor.evaluate_predictions(test_data)
            
            predictions = results['predictions']
            actual = results['actual']
            mse = np.mean((predictions - actual) ** 2)
            
            logging.info(f"Final Evaluation Results:")
            logging.info(f"MAE: {results['mae']:.2f}")
            logging.info(f"RMSE: {results['rmse']:.2f}")
            logging.info(f"MSE: {mse:.2f}")
            
            return {
                'locality': locality,
                'mae': results['mae'],
                'rmse': results['rmse'],
                'mse': mse,
                'error': None
            }
        except Exception as e:
            error_msg = f"Error evaluating model for {locality}: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return {
                'locality': locality,
                'mae': None,
                'rmse': None,
                'mse': None,
                'error': f"Evaluation error: {str(e)}"
            }
    except Exception as e:
        error_msg = f"Unexpected error for {locality}: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return {
            'locality': locality,
            'mae': None,
            'rmse': None,
            'mse': None,
            'error': f"Unexpected error: {str(e)}"
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate LSTM model for all localities')
    parser.add_argument('--sequence_length', type=int, default=24, help='Sequence length for LSTM')
    parser.add_argument('--prediction_horizon', type=int, default=6, help='Prediction horizon')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='Decay factor for spatial weighting')
    parser.add_argument('--max_distance', type=float, default=10.0, help='Maximum distance for nearby localities')
    parser.add_argument('--localities', type=str, help='Comma-separated list of localities to evaluate (default: all)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()
    
    if args.localities:
        localities = args.localities.split(',')
    else:
        localities = get_available_localities()
    
    logging.info(f"Evaluating {len(localities)} localities: {', '.join(localities)}")
    logging.info(f"Using sequence_length={args.sequence_length}, prediction_horizon={args.prediction_horizon}")
    
    results = []
    for locality in tqdm(localities, desc="Evaluating localities"):
        result = evaluate_locality(
            locality, 
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            decay_factor=args.decay_factor,
            max_distance=args.max_distance
        )
        results.append(result)
        
        try:
            with open('locality_evaluation_results.json', 'w', encoding="utf-8", errors="replace") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving results to JSON: {str(e)}")
            try:
                with open('locality_evaluation_results.json', 'w', encoding="ascii", errors="replace") as f:
                    json.dump(results, f, indent=2, ensure_ascii=True)
            except Exception as e2:
                logging.error(f"Failed to save results even with ASCII encoding: {str(e2)}")
    
    table_data = []
    for result in results:
        if result['error'] is None:
            table_data.append([
                result['locality'],
                f"{result['mae']:.4f}",
                f"{result['rmse']:.4f}",
                f"{result['mse']:.4f}",
                "Success"
            ])
        else:
            error_msg = result['error']
            try:
                error_msg = error_msg.encode('utf-8', errors='replace').decode('utf-8')
            except:
                error_msg = "Error message contains invalid characters"
                
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
                
            table_data.append([
                result['locality'],
                "N/A",
                "N/A",
                "N/A",
                error_msg
            ])
    
    table_data.sort(key=lambda x: float(x[2]) if x[2] != "N/A" else float('inf'))
    
    headers = ["Locality", "MAE", "RMSE", "MSE", "Status"]
    try:
        table_output = tabulate(table_data, headers=headers, tablefmt="grid")
        logging.info("\nResults (sorted by RMSE):")
        
        for line in table_output.split('\n'):
            try:
                logging.info(line)
            except UnicodeEncodeError:
                safe_line = line.encode('utf-8', errors='replace').decode('utf-8')
                logging.info(safe_line)
                
        print("\nResults (sorted by RMSE):")
        print(table_output)
    except Exception as e:
        logging.error(f"Error generating results table: {str(e)}")
    
    successful_results = [r for r in results if r['error'] is None]
    if successful_results:
        avg_mae = sum(r['mae'] for r in successful_results) / len(successful_results)
        avg_rmse = sum(r['rmse'] for r in successful_results) / len(successful_results)
        avg_mse = sum(r['mse'] for r in successful_results) / len(successful_results)
        
        logging.info(f"\nAverage metrics across {len(successful_results)} localities:")
        logging.info(f"Average MAE: {avg_mae:.4f}")
        logging.info(f"Average RMSE: {avg_rmse:.4f}")
        logging.info(f"Average MSE: {avg_mse:.4f}")
    
    try:
        with open('locality_evaluation_results.json', 'w', encoding="utf-8", errors="replace") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving final results to JSON: {str(e)}")
        try:
            with open('locality_evaluation_results.json', 'w', encoding="ascii", errors="replace") as f:
                json.dump(results, f, indent=2, ensure_ascii=True)
        except Exception as e2:
            logging.error(f"Failed to save final results even with ASCII encoding: {str(e2)}")
    
    logging.info("Evaluation complete")

if __name__ == "__main__":
    main()
