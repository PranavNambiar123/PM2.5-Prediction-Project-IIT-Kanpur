import pandas as pd
import numpy as np
from load_data import load_all_localities_data
from typing import Dict, Optional
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import json
import math
from geopy.distance import geodesic

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
        Fill missing values using distance-weighted average of surrounding localities
        Uses exponential decay weighting based on distance between localities
        """
        self.method = 'spatial_average'
        cleaned_dict = {}
        
        try:
            # Load location coordinates
            with open('location_coordinates.json', 'r') as f:
                location_coordinates = json.load(f)
            
            # Function to calculate distance between two localities
            def calculate_distance(loc1, loc2):
                if loc1 in location_coordinates and loc2 in location_coordinates:
                    coord1 = location_coordinates[loc1]
                    coord2 = location_coordinates[loc2]
                    return geodesic((coord1[0], coord1[1]), (coord2[0], coord2[1])).kilometers
                return float('inf')  # Return infinity if coordinates not found
            
            # Function to calculate weight based on distance with exponential decay
            def calculate_weight(distance, decay_factor=0.5):
                return math.exp(-decay_factor * distance)
            
            # Find nearby localities for each locality
            locality_neighbors = {}
            for locality in self.raw_data.keys():
                if locality in location_coordinates:
                    neighbors = []
                    for other_locality in self.raw_data.keys():
                        if other_locality != locality and other_locality in location_coordinates:
                            distance = calculate_distance(locality, other_locality)
                            if distance < 10:  # Consider localities within 10km
                                weight = calculate_weight(distance)
                                neighbors.append((other_locality, distance, weight))
                    
                    # Sort by distance
                    neighbors.sort(key=lambda x: x[1])
                    locality_neighbors[locality] = neighbors
                    
                    print(f"Found {len(neighbors)} nearby localities for {locality}")
                    for neighbor, distance, weight in neighbors:
                        print(f"  - {neighbor}: {distance:.2f} km, weight: {weight:.6f}")
                else:
                    print(f"Warning: No coordinates found for {locality}")
                    locality_neighbors[locality] = []
            
            # Initialize with original data
            for locality, df in self.raw_data.items():
                cleaned_dict[locality] = df.copy()
            
            # Fill missing values using distance-weighted average
            for locality, df in cleaned_dict.items():
                missing_pm25 = df['PM2.5'].isna()
                missing_count = missing_pm25.sum()
                
                if missing_count > 0:
                    print(f"Found {missing_count} timestamps with missing PM2.5 values in {locality}")
                    neighbors = locality_neighbors.get(locality, [])
                    
                    if not neighbors:
                        print(f"Warning: No nearby localities found for {locality}, using other cleaning methods")
                        # Fall back to linear interpolation if no neighbors
                        df['PM2.5'] = df['PM2.5'].interpolate(method='linear')
                        df['PM2.5'] = df['PM2.5'].fillna(method='ffill')
                        df['PM2.5'] = df['PM2.5'].fillna(method='bfill')
                        continue
                    
                    filled_count = 0
                    for idx in df[missing_pm25].index:
                        weighted_values = []
                        total_weight = 0
                        
                        for neighbor, distance, weight in neighbors:
                            neighbor_df = self.raw_data[neighbor]
                            if idx in neighbor_df.index and pd.notna(neighbor_df.loc[idx, 'PM2.5']):
                                weighted_values.append(neighbor_df.loc[idx, 'PM2.5'] * weight)
                                total_weight += weight
                        
                        if weighted_values and total_weight > 0:
                            # Calculate weighted average
                            df.loc[idx, 'PM2.5'] = sum(weighted_values) / total_weight
                            filled_count += 1
                    
                    print(f"Filled {filled_count} missing values using spatial data in {locality}")
                    
                    # For any remaining missing values, use interpolation
                    still_missing = df['PM2.5'].isna().sum()
                    if still_missing > 0:
                        print(f"Still have {still_missing} missing values, using interpolation")
                        df['PM2.5'] = df['PM2.5'].interpolate(method='linear')
                        df['PM2.5'] = df['PM2.5'].fillna(method='ffill')
                        df['PM2.5'] = df['PM2.5'].fillna(method='bfill')
                
                # Fill other columns using standard interpolation
                for column in df.select_dtypes(include=[np.number]).columns:
                    if column != 'PM2.5':
                        df[column] = df[column].interpolate(method='linear')
                        df[column] = df[column].fillna(method='ffill')
                        df[column] = df[column].fillna(method='bfill')
                
                cleaned_dict[locality] = df
        
        except Exception as e:
            print(f"Error in spatial average cleaning: {str(e)}")
            print("Falling back to simple spatial average")
            
            # Fall back to simple spatial average if there's an error
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
