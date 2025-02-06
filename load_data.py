import pandas as pd
import os
from glob import glob

def load_all_localities_data(data_dir='air_sensor_calib_data/CAAQMS_MPCB'):
    """
    Load data from all localities in Mumbai from MPCB Excel files
    Returns:
        dict: Dictionary with locality names as keys and corresponding DataFrames as values
    """
    # Get all Excel files
    excel_files = glob(os.path.join(data_dir, 'MPCB_*.xlsx'))
    
    # Dictionary to store data for each locality
    localities_data = {}
    
    for file_path in excel_files:
        # Extract locality name from filename (remove MPCB_ prefix and .xlsx extension)
        locality = os.path.basename(file_path).replace('MPCB_', '').replace('.xlsx', '')
        
        try:
            # Skip the header rows and read the actual data
            # First, read the entire file to find where the data starts
            df_raw = pd.read_excel(file_path)
            
            # Find the row index where the actual data starts
            # Usually after "From Date" or where the first numeric data appears
            data_start_idx = None
            for idx, row in df_raw.iterrows():
                if 'From Date' in str(row.values):
                    data_start_idx = idx + 1
                    break
            
            if data_start_idx is None:
                print(f"Could not find data start for {locality}")
                continue
                
            # Read the data again, now skipping the header rows
            df = pd.read_excel(file_path, skiprows=data_start_idx)
            
            # Convert datetime columns
            df['From Date'] = pd.to_datetime(df['From Date'], format='%d-%m-%Y %H:%M')
            df['To Date'] = pd.to_datetime(df['To Date'], format='%d-%m-%Y %H:%M')
            
            # Set From Date as index
            df = df.set_index('From Date')
            
            # Convert numeric columns to float
            numeric_columns = ['PM2.5', 'PM10', 'RH', 'AT', 'Temp', 'WS']
            for col in df.columns:
                if col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Basic statistics
            print(f"\nLoaded data for {locality}:")
            print(f"Time range: {df.index.min()} to {df.index.max()}")
            print(f"Number of records: {len(df)}")
            print("Data completeness:")
            for col in df.columns:
                if col != 'To Date':
                    pct_valid = (df[col].notna().sum() / len(df)) * 100
                    print(f"  {col}: {pct_valid:.1f}% valid data")
            
            # Store in dictionary
            localities_data[locality] = df
            
        except Exception as e:
            print(f"Error loading data for {locality}: {str(e)}")
    
    return localities_data

if __name__ == "__main__":
    # Load all data
    localities_data = load_all_localities_data()
