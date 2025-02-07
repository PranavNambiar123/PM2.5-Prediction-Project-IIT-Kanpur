import pandas as pd
import os
from glob import glob

def load_all_localities_data(data_dir='air_sensor_calib_data/CAAQMS_MPCB'):
    # Load data from all localities in Mumbai from MPCB Excel files
    excel_files = glob(os.path.join(data_dir, 'MPCB_*.xlsx'))
    localities_data = {}
    
    for file_path in excel_files:
        locality = os.path.basename(file_path).replace('MPCB_', '').replace('.xlsx', '')
        
        try:
            df_raw = pd.read_excel(file_path)
            data_start_idx = None
            for idx, row in df_raw.iterrows():
                if 'From Date' in str(row.values):
                    data_start_idx = idx + 1
                    break
            
            if data_start_idx is None:
                print("Could not find data start for", locality)
                continue
                
            df = pd.read_excel(file_path, skiprows=data_start_idx)
            df['From Date'] = pd.to_datetime(df['From Date'], format='%d-%m-%Y %H:%M')
            df['To Date'] = pd.to_datetime(df['To Date'], format='%d-%m-%Y %H:%M')
            df = df.set_index('From Date')
            
            numeric_columns = ['PM2.5', 'PM10', 'RH', 'AT', 'Temp', 'WS']
            for col in df.columns:
                if col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print("\nLoaded data for", locality)
            print("Time range:", df.index.min(), "to", df.index.max())
            print("Number of records:", len(df))
            print("Data completeness:")
            for col in df.columns:
                if col != 'To Date':
                    pct_valid = (df[col].notna().sum() / len(df)) * 100
                    print(" ", col, ":", round(pct_valid, 1), "% valid data")
            
            localities_data[locality] = df
            
        except Exception as e:
            print("Error loading data for", locality, ":", str(e))
    
    return localities_data

if __name__ == "__main__":
    localities_data = load_all_localities_data()
