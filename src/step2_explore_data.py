import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def explore_and_process_data():
    """
    Step 2: Explore and process traffic data, focusing on Boroondara SCATS sites
    """
    print("Step 2: Exploring and processing traffic data...\n")
    
    # File paths
    data_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load the traffic data 
    scats_data_path = os.path.join(data_dir, 'Scats Data October 2006.xlsx')
    
    print("Loading SCATS traffic data...")
    traffic_data = pd.read_excel(scats_data_path, sheet_name='Data', header=1)
    
    # Load the summary data to get list of SCATS sites
    summary_data = pd.read_excel(scats_data_path, sheet_name='Summary Of Data', header=3)
    
    # === DATA EXPLORATION ===
    print("\n" + "=" * 50)
    print("EXPLORING TRAFFIC DATA")
    print("=" * 50)
    
    # 1. Basic statistics
    print(f"Traffic data shape: {traffic_data.shape}")
    print("\nColumn names:")
    print(traffic_data.columns.tolist()[:15]) # Show first 15 columns to keep output manageable
    
    # 2. Check for missing values
    missing_values = traffic_data.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0]
    print(f"\nColumns with missing values: {len(columns_with_missing)}")
    if len(columns_with_missing) > 0:
        print(columns_with_missing)
    
    # 3. Identify SCATS sites in the data
    print(f"\nNumber of unique SCATS sites: {traffic_data['SCATS Number'].nunique()}")
    print(f"SCATS sites: {sorted(traffic_data['SCATS Number'].unique())}")
    
    # 4. Time period of the data
    if 'Date' in traffic_data.columns:
        print(f"\nData time period: {traffic_data['Date'].min()} to {traffic_data['Date'].max()}")
    
    # === DATA PROCESSING ===
    print("\n" + "=" * 50)
    print("PROCESSING TRAFFIC DATA")
    print("=" * 50)
    
    # 1. Extract volume columns (V00 to V95 representing 15-minute intervals)
    volume_cols = [col for col in traffic_data.columns if col.startswith('V') and col[1:].isdigit()]
    print(f"Found {len(volume_cols)} volume columns")
    
    # 2. Process the data for a single site as an example
    sample_site = traffic_data['SCATS Number'].iloc[0]
    print(f"\nProcessing sample site: {sample_site}")
    
    site_data = traffic_data[traffic_data['SCATS Number'] == sample_site].copy()
    print(f"Site data shape: {site_data.shape}")
    
    # 3. Calculate daily traffic patterns
    # For each day, calculate the average volume for each 15-min interval
    daily_pattern = site_data[volume_cols].mean().reset_index()
    daily_pattern.columns = ['Interval', 'Average Volume']
    
    # Create time labels (00:00, 00:15, ..., 23:45)
    time_labels = []
    for i in range(96):  # 96 15-minute intervals in a day
        hour = i // 4
        minute = (i % 4) * 15
        time_labels.append(f"{hour:02d}:{minute:02d}")
    
    if len(daily_pattern) == len(time_labels):
        daily_pattern['Time'] = time_labels
        print("\nDaily traffic pattern (sample):")
        print(daily_pattern.head())
        
        # Create a visualization of the daily pattern
        plt.figure(figsize=(12, 6))
        plt.plot(daily_pattern['Time'], daily_pattern['Average Volume'])
        plt.title(f'Average Daily Traffic Pattern - SCATS Site {sample_site}')
        plt.xlabel('Time of Day')
        plt.ylabel('Average Volume (vehicles per 15 min)')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(processed_dir, f'site_{sample_site}_daily_pattern.png'))
        plt.close()
        print(f"Saved daily pattern visualization to {processed_dir}/site_{sample_site}_daily_pattern.png")
    
    # 4. Create time series data for ML model training
    # Reshape the data into a time series format with past values as features
    print("\nCreating time series data for ML model training...")
    
    # Combine date and time to create a proper time series
    # 4. Create time series data for ML model training

    # Prepare site data in chronological order
    site_data = site_data.sort_values('Date')

    # Extract volume data
    volume_data = site_data[volume_cols].values

    # Create sliding windows for time series
    X_data = []
    y_data = []
    window_size = 4  # Use 4 previous 15-min intervals to predict the next one

    # Create sliding windows across the entire dataset
    for i in range(len(volume_data) - window_size):
        X_data.append(volume_data[i:i+window_size])
        y_data.append(volume_data[i+window_size])

    X = np.array(X_data)
    y = np.array(y_data)

    print(f"Created time series data with shape: X={X.shape}, y={y.shape}")
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"Created time series data with shape: X={X.shape}, y={y.shape}")
    
    # 5. Save processed data for ML model training
    np.save(os.path.join(processed_dir, f'site_{sample_site}_X.npy'), X)
    np.save(os.path.join(processed_dir, f'site_{sample_site}_y.npy'), y)
    print(f"Saved time series data to {processed_dir}")
    
    # 6. Save daily pattern data
    daily_pattern.to_csv(os.path.join(processed_dir, f'site_{sample_site}_daily_pattern.csv'), index=False)
    
    # === PREPARE BOROONDARA SITES LIST ===
    print("\n" + "=" * 50)
    print("PREPARING BOROONDARA SITES LIST")
    print("=" * 50)
    
    # From the summary sheet, extract all the SCATS sites
    # First, clean up the summary data
    summary_data = summary_data.dropna(subset=['SCATS Number'])
    
    # Get unique SCATS site numbers
    boroondara_sites = summary_data['SCATS Number'].unique()
    print(f"Identified {len(boroondara_sites)} unique SCATS sites in Boroondara")
    print(f"Sample sites: {sorted(boroondara_sites)[:10]}")
    
    # Save the list of Boroondara SCATS sites
    pd.DataFrame({'SCATS_Site': boroondara_sites}).to_csv(
        os.path.join(processed_dir, 'boroondara_scats_sites.csv'), index=False)
    print(f"Saved Boroondara SCATS sites list to {processed_dir}/boroondara_scats_sites.csv")
    
    print("\nStep 2 complete: Data exploration and processing finished")

if __name__ == "__main__":
    explore_and_process_data()