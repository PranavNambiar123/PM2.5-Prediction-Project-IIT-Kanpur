import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_all_localities_data
import numpy as np
from datetime import datetime
import os

def create_output_dir():
    """Create output directory for plots if it doesn't exist"""
    os.makedirs('plots', exist_ok=True)

def plot_pm25_time_series(data_dict, save=True):
    """Plot PM2.5 time series for all localities"""
    plt.figure(figsize=(15, 8))
    for locality, df in data_dict.items():
        daily_mean = df['PM2.5'].resample('D').mean()
        plt.plot(daily_mean.index, daily_mean.values, label=locality, alpha=0.7)
    
    plt.title('Daily Average PM2.5 Concentration Over Time')
    plt.xlabel('Date')
    plt.ylabel('PM2.5 Concentration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('plots/pm25_time_series.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_monthly_boxplots(data_dict, save=True):
    """Create monthly boxplots for PM2.5 levels"""
    plt.figure(figsize=(15, 8))
    all_monthly_data = []
    months = []
    localities = []
    
    for locality, df in data_dict.items():
        monthly_data = df['PM2.5'].groupby(df.index.month)
        for month, values in monthly_data:
            all_monthly_data.extend(values.dropna())
            months.extend([month] * len(values.dropna()))
            localities.extend([locality] * len(values.dropna()))
    
    df_monthly = pd.DataFrame({
        'Month': months,
        'PM2.5': all_monthly_data,
        'Locality': localities
    })
    
    sns.boxplot(data=df_monthly, x='Month', y='PM2.5', hue='Locality')
    plt.title('Monthly PM2.5 Distribution by Locality')
    plt.xlabel('Month')
    plt.ylabel('PM2.5 Concentration')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save:
        plt.savefig('plots/monthly_boxplots.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_daily_patterns(data_dict, save=True):
    """Plot average daily patterns of PM2.5"""
    plt.figure(figsize=(15, 8))
    for locality, df in data_dict.items():
        hourly_mean = df['PM2.5'].groupby(df.index.hour).mean()
        plt.plot(hourly_mean.index, hourly_mean.values, label=locality, alpha=0.7)
    
    plt.title('Average Daily PM2.5 Patterns')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average PM2.5 Concentration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(24))
    plt.tight_layout()
    if save:
        plt.savefig('plots/daily_patterns.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_correlation_heatmaps(data_dict, save=True):
    """Plot correlation heatmaps for each locality"""
    for locality, df in data_dict.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            continue
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(f'Feature Correlations for {locality}')
        plt.tight_layout()
        if save:
            plt.savefig(f'plots/correlation_{locality}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_pm25_distribution(data_dict, save=True):
    """Plot PM2.5 distribution for each locality"""
    plt.figure(figsize=(15, 8))
    data = []
    labels = []
    for locality, df in data_dict.items():
        data.append(df['PM2.5'].dropna())
        labels.extend([locality] * len(df['PM2.5'].dropna()))
    
    plt.violinplot(data, positions=range(len(data)))
    plt.xticks(range(len(data)), data_dict.keys(), rotation=45)
    plt.title('PM2.5 Distribution Across Localities')
    plt.ylabel('PM2.5 Concentration')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('plots/pm25_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_missing_data_heatmap(data_dict, save=True):
    """Plot missing data patterns"""
    plt.figure(figsize=(15, 8))
    missing_data = pd.DataFrame()
    
    for locality, df in data_dict.items():
        missing = df.isna().mean() * 100
        missing_data[locality] = missing
    
    sns.heatmap(missing_data, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Percentage of Missing Data by Feature and Locality')
    plt.ylabel('Features')
    plt.tight_layout()
    if save:
        plt.savefig('plots/missing_data_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_seasonal_comparison(data_dict, save=True):
    """Compare PM2.5 levels across seasons"""
    plt.figure(figsize=(15, 8))
    
    for locality, df in data_dict.items():
        df['season'] = pd.cut(df.index.month, 
                            bins=[0, 2, 5, 9, 12],
                            labels=['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'])
        
        seasonal_means = df.groupby('season')['PM2.5'].mean()
        plt.plot(range(4), seasonal_means.values, 'o-', label=locality)
    
    plt.title('Seasonal Comparison of PM2.5 Levels')
    plt.xlabel('Season')
    plt.ylabel('Average PM2.5 Concentration')
    plt.xticks(range(4), ['Winter', 'Summer', 'Monsoon', 'Post-Monsoon'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('plots/seasonal_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_weekday_patterns(data_dict, save=True):
    """Plot average PM2.5 levels by day of week"""
    plt.figure(figsize=(15, 8))
    
    for locality, df in data_dict.items():
        weekday_means = df.groupby(df.index.dayofweek)['PM2.5'].mean()
        plt.plot(weekday_means.index, weekday_means.values, 'o-', label=locality)
    
    plt.title('Average PM2.5 Levels by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average PM2.5 Concentration')
    plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('plots/weekday_patterns.png', bbox_inches='tight', dpi=300)
    plt.close()

def generate_all_visualizations():
    """Generate all visualizations"""
    print("Loading data...")
    data_dict = load_all_localities_data()
    
    print("\nCreating output directory...")
    create_output_dir()
    
    print("\nGenerating visualizations...")
    print("1. Time series plot")
    plot_pm25_time_series(data_dict)
    
    print("2. Monthly boxplots")
    plot_monthly_boxplots(data_dict)
    
    print("3. Daily patterns")
    plot_daily_patterns(data_dict)
    
    print("4. Correlation heatmaps")
    plot_correlation_heatmaps(data_dict)
    
    print("5. PM2.5 distribution")
    plot_pm25_distribution(data_dict)
    
    print("6. Missing data heatmap")
    plot_missing_data_heatmap(data_dict)
    
    print("7. Seasonal comparison")
    plot_seasonal_comparison(data_dict)
    
    print("8. Weekday patterns")
    plot_weekday_patterns(data_dict)
    
    print("\nAll visualizations have been saved in the 'plots' directory!")

if __name__ == "__main__":
    generate_all_visualizations()
