import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("future_prediction_viz.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def visualize_future_predictions(prediction_file, actual_file=None, output_dir='future_prediction_plots'):
    """
    Visualize future predictions and compare with actual values if available
    
    Args:
        prediction_file (str): Path to the CSV file with predictions
        actual_file (str, optional): Path to the CSV file with actual values
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prediction data
    try:
        pred_df = pd.read_csv(prediction_file)
        pred_df['Timestamp'] = pd.to_datetime(pred_df['Timestamp'])
        
        # Extract locality and timestamp from filename
        filename = os.path.basename(prediction_file)
        parts = filename.split('_predictions_')
        locality = parts[0]
        timestamp = parts[1].split('.')[0]
        
        logging.info(f"Loaded predictions for {locality} generated at {timestamp}")
        logging.info(f"Prediction shape: {pred_df.shape}")
        
    except Exception as e:
        logging.error(f"Error loading prediction file: {str(e)}")
        return
    
    # Load actual data if provided
    actual_df = None
    if actual_file:
        try:
            actual_df = pd.read_csv(actual_file)
            actual_df['Timestamp'] = pd.to_datetime(actual_df['Timestamp'])
            logging.info(f"Loaded actual values with shape: {actual_df.shape}")
        except Exception as e:
            logging.error(f"Error loading actual file: {str(e)}")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    plt.plot(pred_df['Timestamp'], pred_df['PM2.5_Prediction'], 
             marker='o', linestyle='-', linewidth=2, label='Predicted PM2.5')
    
    # Plot actual values if available
    if actual_df is not None:
        plt.plot(actual_df['Timestamp'], actual_df['PM2.5_Actual'], 
                marker='s', linestyle='-', linewidth=2, label='Actual PM2.5')
        
        # Calculate metrics if actual values are available
        # Find matching timestamps
        merged_df = pd.merge(
            pred_df, actual_df, 
            on='Timestamp', 
            how='inner',
            suffixes=('_Prediction', '_Actual')
        )
        
        if not merged_df.empty:
            # Calculate metrics
            mae = np.mean(np.abs(merged_df['PM2.5_Prediction'] - merged_df['PM2.5_Actual']))
            rmse = np.sqrt(np.mean((merged_df['PM2.5_Prediction'] - merged_df['PM2.5_Actual']) ** 2))
            
            # Add metrics to plot
            plt.text(0.02, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
                    transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            logging.info(f"Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
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
    plot_file = f'{output_dir}/{locality}_future_prediction_{current_time}.png'
    plt.savefig(plot_file)
    logging.info(f"Plot saved to {plot_file}")
    
    # Display plot information
    print(f"\nFuture Prediction Visualization")
    print("=" * 60)
    print(f"Locality: {locality}")
    print(f"Prediction generated at: {timestamp}")
    print("-" * 60)
    print(f"{'Timestamp':<25} {'Predicted PM2.5':<15}")
    print("-" * 60)
    
    for _, row in pred_df.iterrows():
        print(f"{row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S'):<25} {row['PM2.5_Prediction']:<15.2f}")
    
    print("-" * 60)
    
    if actual_df is not None and not merged_df.empty:
        print(f"Comparison metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    print(f"Visualization saved to: {plot_file}")
    print("=" * 60)

def create_actual_values_template(prediction_file, output_file=None):
    """
    Create a template CSV file for entering actual values
    
    Args:
        prediction_file (str): Path to the prediction CSV file
        output_file (str, optional): Path to save the template
    
    Returns:
        str: Path to the created template file
    """
    try:
        # Load prediction data
        pred_df = pd.read_csv(prediction_file)
        pred_df['Timestamp'] = pd.to_datetime(pred_df['Timestamp'])
        
        # Create template dataframe
        template_df = pd.DataFrame({
            'Timestamp': pred_df['Timestamp'],
            'PM2.5_Actual': [None] * len(pred_df)
        })
        
        # Determine output file name if not provided
        if output_file is None:
            filename = os.path.basename(prediction_file)
            parts = filename.split('_predictions_')
            locality = parts[0]
            timestamp = parts[1].split('.')[0]
            
            output_file = f'actual_values/{locality}_actual_{timestamp}.csv'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save template
        template_df.to_csv(output_file, index=False)
        logging.info(f"Created actual values template at {output_file}")
        
        print(f"\nActual Values Template Created")
        print("=" * 60)
        print(f"Template saved to: {output_file}")
        print("Please fill in the 'PM2.5_Actual' column with actual values once available")
        print("=" * 60)
        
        return output_file
        
    except Exception as e:
        logging.error(f"Error creating template: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualize future PM2.5 predictions')
    parser.add_argument('--prediction_file', type=str, required=True, 
                        help='Path to the prediction CSV file')
    parser.add_argument('--actual_file', type=str, default=None,
                        help='Path to the CSV file with actual values (optional)')
    parser.add_argument('--create_template', action='store_true',
                        help='Create a template for entering actual values')
    parser.add_argument('--output_dir', type=str, default='future_prediction_plots',
                        help='Directory to save the plots')
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_template:
        create_actual_values_template(args.prediction_file)
    
    # Visualize predictions
    visualize_future_predictions(
        prediction_file=args.prediction_file,
        actual_file=args.actual_file,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
