import pandas as pd
import numpy as np
import os
import json
from data_cleaning import DataCleaner
from load_data import load_all_localities_data

def main():
    print("Loading data...")
    data_dict = load_all_localities_data()
    
    # Create a directory for the spatially cleaned data
    os.makedirs('spatial_cleaned_data', exist_ok=True)
    
    # Initialize the data cleaner
    cleaner = DataCleaner(data_dict)
    
    print("\nRunning spatial data cleaning...")
    cleaned_data = cleaner.clean_by_spatial_average()
    
    # Save the cleaned data to a different directory
    print("\nSaving spatially cleaned data...")
    for locality, df in cleaned_data.items():
        output_file = f'spatial_cleaned_data/{locality}_spatial_weighted.csv'
        df.to_csv(output_file)
        print(f"Saved {locality} data to {output_file}")
    
    # Create a summary file with statistics
    print("\nGenerating summary statistics...")
    summary_data = []
    
    for locality, df in cleaned_data.items():
        # Calculate statistics
        stats = {
            'locality': locality,
            'total_rows': len(df),
            'pm25_mean': df['PM2.5'].mean(),
            'pm25_median': df['PM2.5'].median(),
            'pm25_std': df['PM2.5'].std(),
            'pm25_min': df['PM2.5'].min(),
            'pm25_max': df['PM2.5'].max()
        }
        summary_data.append(stats)
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_file = 'spatial_cleaned_data/cleaning_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary statistics to {summary_file}")
    
    print("\nSpatial data cleaning complete!")

if __name__ == "__main__":
    main()
