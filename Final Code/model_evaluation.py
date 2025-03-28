import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns
from typing import Dict, List, Tuple

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator"""
        self.results = {}
        self.models = {}
        
    def load_cleaned_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all cleaned datasets
        Returns: Dictionary with structure {cleaning_method: {locality: dataframe}}
        """
        cleaned_data = {}
        
        csv_files = glob(os.path.join('cleaned_data', '*.csv'))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            locality, method = filename.replace('.csv', '').split('_', 1)
            
            df = pd.read_csv(file_path)
            if 'Unnamed: 0' in df.columns:
                df.set_index(pd.to_datetime(df['Unnamed: 0']), inplace=True)
                df.drop('Unnamed: 0', axis=1, inplace=True)
            
            if method not in cleaned_data:
                cleaned_data[method] = {}
            
            cleaned_data[method][locality] = df
            
        return cleaned_data
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for modeling
        """
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != 'PM2.5']
        
        if not feature_cols:
            raise ValueError(f"No numeric features found in columns: {df.columns}")
            
        X = df[feature_cols].values
        y = df['PM2.5'].values
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets (70-15-15)
        """
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def evaluate_locality(self, df: pd.DataFrame, locality: str, method: str) -> Dict:
        """
        Evaluate a single locality's data
        """
        try:
            df = df.dropna()
            
            if len(df) < 100:  # arbitrary minimum size
                print(f"Skipping {locality} due to insufficient data (only {len(df)} samples)")
                return None
                
            X, y = self.prepare_features(df)
            
            if X.shape[1] == 0:
                print(f"Skipping {locality} due to no valid features")
                return None
                
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            if method not in self.models:
                self.models[method] = {}
            self.models[method][locality] = model
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'val_mae': mean_absolute_error(y_val, y_val_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'val_r2': r2_score(y_val, y_val_pred),
                'test_r2': r2_score(y_test, y_test_pred)
            }
            
            return metrics
        except Exception as e:
            print(f"Error processing {locality}: {str(e)}")
            return None
    
    def evaluate_all_methods(self):
        """
        Evaluate all cleaning methods and localities
        """
        cleaned_data = self.load_cleaned_data()
        
        for method, localities_data in cleaned_data.items():
            print(f"\nEvaluating {method} method...")
            method_results = {}
            
            for locality, df in localities_data.items():
                print(f"Processing {locality}...")
                metrics = self.evaluate_locality(df, locality, method)
                if metrics is not None:
                    method_results[locality] = metrics
            
            if method_results:
                self.results[method] = method_results
    
    def plot_comparison(self):
        """
        Plot comparison of different cleaning methods
        """
        if not self.results:
            print("No results to plot!")
            return
            
        os.makedirs('model_results', exist_ok=True)
        
        methods = list(self.results.keys())
        metrics = ['test_rmse', 'test_mae', 'test_r2']
        metric_names = ['RMSE', 'MAE', 'R²']
        
        for metric, metric_name in zip(metrics, metric_names):
            plt.figure(figsize=(12, 6))
            data = []
            labels = []
            
            for method in methods:
                values = [self.results[method][locality][metric] 
                         for locality in self.results[method].keys()]
                if values:
                    data.append(values)
                    labels.extend([method] * len(values))
            
            if data:
                plt.boxplot(data, labels=methods)
                plt.title(f'Comparison of {metric_name} Across Cleaning Methods')
                plt.ylabel(metric_name)
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'model_results/comparison_{metric}.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        results_df = pd.DataFrame()
        for method in methods:
            method_data = pd.DataFrame(self.results[method]).T
            method_data['method'] = method
            results_df = pd.concat([results_df, method_data])
        
        results_df.to_csv('model_results/all_results.csv')
        print("\nResults have been saved to 'model_results' directory!")

def main():
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_methods()
    evaluator.plot_comparison()

if __name__ == "__main__":
    main()
