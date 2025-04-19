"""
Step 2: Explore and Process Traffic Data
This script analyzes traffic data from SCATS sites in Boroondara by:
- Loading and checking the traffic volume data
- Creating daily traffic patterns and visualizations
- Preparing data for machine learning analysis
- Saving processed data files for later use
"""

# Import needed packages
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import os  # For file and directory operations
import matplotlib.pyplot as plt  # For creating visualizations
from datetime import datetime, timedelta  # For date/time handling

def explore_and_process_data():
    # Print a message showing we're starting Step 2
    print("Step 2: Exploring and processing traffic data...\n")
    
    # Define paths for raw and processed data folders
    data_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)  # Create processed folder if it doesn't exist
    
    # Load the Excel file containing traffic data
    scats_data_path = os.path.join(data_dir, 'Scats Data October 2006.xlsx')
    print("Loading SCATS traffic data...")
    traffic_data = pd.read_excel(scats_data_path, sheet_name='Data', header=1)
    
    # Load the summary sheet to get information about SCATS sites
    summary_data = pd.read_excel(scats_data_path, sheet_name='Summary Of Data', header=3)
    
    # Print a header for the data exploration section
    print("\n" + "=" * 50)
    print("EXPLORING TRAFFIC DATA")
    print("=" * 50)
    
    # Show the dimensions of the traffic data
    print(f"Traffic data shape: {traffic_data.shape}")
    print("\nColumn names:")
    print(traffic_data.columns.tolist()[:15])  # Show first 15 columns only
    
    # Check for missing values in the dataset
    missing_values = traffic_data.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0]
    print(f"\nColumns with missing values: {len(columns_with_missing)}")
    if len(columns_with_missing) > 0:
        print(columns_with_missing)
    
    # Show how many unique SCATS sites are in the data
    print(f"\nNumber of unique SCATS sites: {traffic_data['SCATS Number'].nunique()}")
    print(f"SCATS sites: {sorted(traffic_data['SCATS Number'].unique())}")
    
    # Show the time period covered by the data
    if 'Date' in traffic_data.columns:
        print(f"\nData time period: {traffic_data['Date'].min()} to {traffic_data['Date'].max()}")
    
    # Print a header for the data processing section
    print("\n" + "=" * 50)
    print("PROCESSING TRAFFIC DATA")
    print("=" * 50)
    
    # Find all columns containing traffic volume data (V00-V95)
    volume_cols = [col for col in traffic_data.columns if col.startswith('V') and col[1:].isdigit()]
    print(f"Found {len(volume_cols)} volume columns")
    
    # Select the first SCATS site as an example to process
    sample_site = traffic_data['SCATS Number'].iloc[0]
    print(f"\nProcessing sample site: {sample_site}")
    site_data = traffic_data[traffic_data['SCATS Number'] == sample_site].copy()
    
    # Calculate the average traffic volume for each 15-minute interval
    daily_pattern = site_data[volume_cols].mean().reset_index()
    daily_pattern.columns = ['Interval', 'Average Volume']
    
    # Create time labels for each 15-minute interval (00:00, 00:15, etc.)
    time_labels = []
    for i in range(96):  # 96 intervals in a day
        hour = i // 4
        minute = (i % 4) * 15
        time_labels.append(f"{hour:02d}:{minute:02d}")
    
    # Create and save a visualization of the daily traffic pattern
    if len(daily_pattern) == len(time_labels):
        daily_pattern['Time'] = time_labels
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
    
    # Sort the site data by date to ensure chronological order
    site_data = site_data.sort_values('Date')
    
    # Extract just the traffic volume data as a numpy array (each row = 1 day, 96 intervals)
    volume_data = site_data[volume_cols].values

    # We'll now create training data for a machine learning model using a sliding window approach
    # Each input (X) will be 10 consecutive days of traffic data
    # The target/output (y) will be the traffic data of the day that comes right after those 10 days

    X_data = []  # This will store input sequences (10 days each)
    y_data = []  # This will store the actual next day (used as label/output)

    window_size = 10  # Number of past days we want to use to predict the next day

    # Loop through the dataset and create input-output pairs
    for i in range(len(volume_data) - window_size):
        # X[i] = 10 days of traffic data (i to i+9)
        X_data.append(volume_data[i:i + window_size])
        
        # y[i] = 11th day (i+10), which is the day we want to predict
        # This is still taken from the actual dataset — we just save it separately
        y_data.append(volume_data[i + window_size])

    # Convert lists to numpy arrays so we can use them in machine learning models
    # X shape → (number of samples, 10 days, 96 intervals)
    # y shape → (number of samples, 96 intervals for the next day)
    X = np.array(X_data)
    y = np.array(y_data)

    print(f"Created time series data with shape: X={X.shape}, y={y.shape}")

    
    # Save the processed data for later use
    np.save(os.path.join(processed_dir, f'site_{sample_site}_X.npy'), X)
    np.save(os.path.join(processed_dir, f'site_{sample_site}_y.npy'), y)
    daily_pattern.to_csv(os.path.join(processed_dir, f'site_{sample_site}_daily_pattern.csv'), index=False)
    
    # Print a header for the Boroondara sites section
    print("\n" + "=" * 50)
    print("PREPARING BOROONDARA SITES LIST")
    print("=" * 50)
    
    # Clean the summary data by removing rows with missing SCATS numbers
    summary_data = summary_data.dropna(subset=['SCATS Number'])
    
    # Get a list of all unique SCATS sites in Boroondara
    boroondara_sites = summary_data['SCATS Number'].unique()
    print(f"Identified {len(boroondara_sites)} unique SCATS sites in Boroondara")
    print(f"Sample sites: {sorted(boroondara_sites)[:10]}")
    
    # Save the list of Boroondara SCATS sites for later steps
    pd.DataFrame({'SCATS_Site': boroondara_sites}).to_csv(
        os.path.join(processed_dir, 'boroondara_scats_sites.csv'), index=False)
    print(f"Saved Boroondara SCATS sites list to {processed_dir}/boroondara_scats_sites.csv")
    
    # Print completion message
    print("\nStep 2 complete: Data exploration and processing finished")

# Run the function if this script is executed directly
if __name__ == "__main__":
    explore_and_process_data()
