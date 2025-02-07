import pandas as pd
import numpy as np
from load_data import load_all_localities_data
from typing import Dict, Optional
import matplotlib.pyplot as plt
from scipy import interpolate
import os

class DataCleaner:
    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        """
        Initialize with dictionary of DataFrames for each locality
        """
        self.raw_data = {}
        for locality, df in data_dict.items():
            df = df.copy()
            columns_to_keep = ['PM2.5', 'PM10', 'RH']
            if 'Temp' in df.columns:
                df['AT'] = df['Temp']
            if 'AT' in df.columns:
                columns_to_keep.append('AT')
            df = df[columns_to_keep]
            self.raw_data[locality] = df
        
        self.cleaned_data = None
        self.method = None
        os.makedirs('cleaned_data', exist_ok=True)
    
    def clean_by_statistical_measure(self, measure: str = 'mean'):
        """
        Fill missing values using mean, median, or mode
        Args:
            measure: One of 'mean', 'median', 'mode'
        """
        self.method = f'statistical_{measure}'
        cleaned_dict = {}
        
        for locality, df in self.raw_data.items():
            df_cleaned = df.copy()
            
            for column in df.select_dtypes(include=[np.number]).columns:
                if measure == 'mean':
                    fill_value = df[column].mean()
                elif measure == 'median':
                    fill_value = df[column].median()
                else:
                    fill_value = df[column].mode()[0] if not df[column].mode().empty else np.nan
                
                df_cleaned[column] = df_cleaned[column].fillna(fill_value)
            
            cleaned_dict[locality] = df_cleaned
        
        self.cleaned_data = cleaned_dict
        return cleaned_dict
    
    def clean_by_elimination(self):
        """
        Remove all rows with any missing values
        """
        self.method = 'elimination'
        cleaned_dict = {}
        
        for locality, df in self.raw_data.items():
            df_cleaned = df.dropna()
            cleaned_dict[locality] = df_cleaned
        
        self.cleaned_data = cleaned_dict
        return cleaned_dict
    
    def clean_by_linear_interpolation(self):
        """
        Fill missing values using linear interpolation between non-empty events
        """
        self.method = 'linear_interpolation'
        cleaned_dict = {}
        
        for locality, df in self.raw_data.items():
            df_cleaned = df.copy()
            
            for column in df.select_dtypes(include=[np.number]).columns:
                df_cleaned[column] = df_cleaned[column].interpolate(method='linear')
                df_cleaned[column] = df_cleaned[column].fillna(method='ffill')
                df_cleaned[column] = df_cleaned[column].fillna(method='bfill')
            
            cleaned_dict[locality] = df_cleaned
        
        self.cleaned_data = cleaned_dict
        return cleaned_dict
    
    def clean_by_spatial_average(self):
        """
        Fill missing values using average of surrounding localities
        """
        self.method = 'spatial_average'
        cleaned_dict = {}
        
        for locality, df in self.raw_data.items():
            cleaned_dict[locality] = df.copy()
        
        for locality in self.raw_data.keys():
            df = cleaned_dict[locality]
            
            for column in df.select_dtypes(include=[np.number]).columns:
                missing_idx = df[column].isna()
                
                if missing_idx.any():
                    for idx in df[missing_idx].index:
                        values_at_timestamp = []
                        
                        for other_locality, other_df in self.raw_data.items():
                            if other_locality != locality and idx in other_df.index:
                                val = other_df.loc[idx, column]
                                if pd.notna(val):
                                    values_at_timestamp.append(val)
                        
                        if values_at_timestamp:
                            df.loc[idx, column] = np.mean(values_at_timestamp)
            
            cleaned_dict[locality] = df
        
        self.cleaned_data = cleaned_dict
        return cleaned_dict
    
    def compare_methods(self, locality: str, feature: str = 'PM2.5'):
        """
        Compare different cleaning methods for a specific locality and feature
        """
        plt.figure(figsize=(15, 10))
        
        original = self.raw_data[locality][feature]
        plt.plot(original.index, original.values, 'o', label='Original', alpha=0.3, markersize=2)
        
        self.clean_by_statistical_measure('mean')
        mean_cleaned = self.cleaned_data[locality][feature]
        plt.plot(mean_cleaned.index, mean_cleaned.values, label='Mean', alpha=0.7)
        
        self.clean_by_statistical_measure('median')
        median_cleaned = self.cleaned_data[locality][feature]
        plt.plot(median_cleaned.index, median_cleaned.values, label='Median', alpha=0.7)
        
        self.clean_by_linear_interpolation()
        interp_cleaned = self.cleaned_data[locality][feature]
        plt.plot(interp_cleaned.index, interp_cleaned.values, label='Linear Interpolation', alpha=0.7)
        
        self.clean_by_spatial_average()
        spatial_cleaned = self.cleaned_data[locality][feature]
        plt.plot(spatial_cleaned.index, spatial_cleaned.values, label='Spatial Average', alpha=0.7)
        
        plt.title(f'Comparison of Cleaning Methods for {locality} - {feature}')
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f'cleaned_data/comparison_{locality}_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_cleaned_data(self):
        """
        Save the cleaned data to CSV files
        """
        if self.cleaned_data is None or self.method is None:
            raise ValueError("No cleaned data available. Run a cleaning method first.")
        
        for locality, df in self.cleaned_data.items():
            df.to_csv(f'cleaned_data/{locality}_{self.method}.csv')

def main():
    print("Loading data...")
    data_dict = load_all_localities_data()
    
    cleaner = DataCleaner(data_dict)
    
    print("\nGenerating comparison plots...")
    for locality in data_dict.keys():
        print(f"Processing {locality}...")
        cleaner.compare_methods(locality)
    
    print("\nGenerating cleaned datasets...")
    
    print("1. Mean imputation")
    cleaner.clean_by_statistical_measure('mean')
    cleaner.save_cleaned_data()
    
    print("2. Median imputation")
    cleaner.clean_by_statistical_measure('median')
    cleaner.save_cleaned_data()
    
    print("3. Linear interpolation")
    cleaner.clean_by_linear_interpolation()
    cleaner.save_cleaned_data()
    
    print("4. Spatial average")
    cleaner.clean_by_spatial_average()
    cleaner.save_cleaned_data()
    
    print("\nAll cleaned datasets have been saved in the 'cleaned_data' directory!")

if __name__ == "__main__":
    main()
