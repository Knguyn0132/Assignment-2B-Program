import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pickle

def clean_and_prepare_data():
    """
    Step 4: Clean, normalize, and prepare traffic data for ML model training
    """

    print("Step 4: Cleaning and preparing traffic data for ML modeling...\n")
    
    # Setup file paths
    processed_dir = os.path.join('data', 'processed')
    sites_dir = os.path.join(processed_dir, 'sites')          # raw site data
    cleaned_dir = os.path.join(processed_dir, 'cleaned')      # where cleaned files will be saved
    os.makedirs(cleaned_dir, exist_ok=True)
    
    # Load the site metadata (like SCATS site IDs)
    with open(os.path.join(processed_dir, 'sites_metadata.json'), 'r') as f:
        sites_metadata = json.load(f)
    
    print(f"Processing {len(sites_metadata)} SCATS sites")
    
    # Dictionary to store cleaning info for each site
    site_stats = {}
    
    # Loop through each site
    for site_id, metadata in sites_metadata.items():
        print(f"\nCleaning data for SCATS site {site_id}")
        
        # Define paths for input and output files for this site
        site_dir = os.path.join(sites_dir, site_id)
        site_clean_dir = os.path.join(cleaned_dir, site_id)
        os.makedirs(site_clean_dir, exist_ok=True)
        
        x_path = os.path.join(site_dir, 'X_data.npy')  # input features (e.g., past traffic, time info)
        y_path = os.path.join(site_dir, 'y_data.npy')  # output values (e.g., next time step traffic volume)
        
        # Skip site if no data exists
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"  No time series data found for site {site_id}, skipping")
            continue
        
        try:
            X = np.load(x_path)
            y = np.load(y_path)
            
            # Skip empty data
            if X.size == 0 or y.size == 0:
                print(f"  Empty time series data for site {site_id}, skipping")
                continue
            
            print(f"  Original data shape: X={X.shape}, y={y.shape}")
            
            # ====== STEP 1: Handle missing values ======
            # Check for missing values (NaN) in features or targets
            nan_count_X = np.isnan(X).sum()
            nan_count_y = np.isnan(y).sum()
            
            # Replace NaNs in X with column mean
            if nan_count_X > 0:
                print(f"  Found {nan_count_X} NaN values in X data")
                for i in range(X.shape[1]):
                    col_mean = np.nanmean(X[:, i])
                    nan_indices = np.isnan(X[:, i])
                    X[nan_indices, i] = col_mean
            
            # Replace NaNs in y with its mean
            if nan_count_y > 0:
                print(f"  Found {nan_count_y} NaN values in y data")
                y_mean = np.nanmean(y)
                y[np.isnan(y)] = y_mean
            
            # ====== STEP 2: Handle outliers ======
            # Use z-score to find values far from average (more than 3 std devs)
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_z = np.abs((X - X_mean) / X_std)
            
            y_mean = np.mean(y)
            y_std = np.std(y)
            y_z = np.abs((y - y_mean) / y_std)
            
            # Count how many outliers exist
            X_outliers = (X_z > 3).sum()
            y_outliers = (y_z > 3).sum()
            
            # Cap outliers in X to the [mean - 3*std, mean + 3*std] range
            if X_outliers > 0:
                print(f"  Found {X_outliers} outliers in X data")
                for i in range(X.shape[1]):
                    cap_high = X_mean[i] + 3 * X_std[i]
                    cap_low = X_mean[i] - 3 * X_std[i]
                    X[:, i] = np.clip(X[:, i], cap_low, cap_high)
            
            # Same for y
            if y_outliers > 0:
                print(f"  Found {y_outliers} outliers in y data")
                cap_high = y_mean + 3 * y_std
                cap_low = y_mean - 3 * y_std
                y = np.clip(y, cap_low, cap_high)
            
            # ====== STEP 3: Normalize to [0, 1] range ======
            # Scale values so they all fall between 0 and 1 (helps model training)
            X_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Reshape to 2D for the scaler (scikit-learn expects 2D arrays)
            X_reshaped = X.reshape(-1, X.shape[-1])
            y_reshaped = y.reshape(-1, 1)
            
            X_scaled = X_scaler.fit_transform(X_reshaped)
            y_scaled = y_scaler.fit_transform(y_reshaped)
            
            # Reshape back to original shape
            X_scaled = X_scaled.reshape(X.shape)
            y_scaled = y_scaled.flatten()
            
            print(f"  Data normalized to [0, 1] range")
            
            # ====== STEP 4: Split into training and testing (80%/20%) ======
            # Split based on number of 15-min intervals (not by day!)
            # First 80% used to train the model; last 20% used to test how well it performs
            train_size = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
            y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
            
            print(f"  Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
            
            # ====== STEP 5: Save everything ======
            # These will be used later to train the model
            np.save(os.path.join(site_clean_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(site_clean_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(site_clean_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(site_clean_dir, 'y_test.npy'), y_test)
            
            # Save scalers to reverse normalization later (for prediction output)
            with open(os.path.join(site_clean_dir, 'X_scaler.pkl'), 'wb') as f:
                pickle.dump(X_scaler, f)
            with open(os.path.join(site_clean_dir, 'y_scaler.pkl'), 'wb') as f:
                pickle.dump(y_scaler, f)
            
            # ====== STEP 6: Plot original vs normalized target values ======
            sample_size = min(100, len(X))  # Only show a small sample
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(y[:sample_size])
            plt.title('Original Data (Sample)')
            plt.xlabel('Sample Index')
            plt.ylabel('Traffic Volume')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(y_scaled[:sample_size])
            plt.title('Normalized Data (Sample)')
            plt.xlabel('Sample Index')
            plt.ylabel('Normalized Volume')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(site_clean_dir, 'data_normalization.png'))
            plt.close()
            
            # Record cleaning summary for this site
            site_stats[site_id] = {
                'original_samples': len(X),
                'nan_values': {
                    'X': int(nan_count_X),
                    'y': int(nan_count_y)
                },
                'outliers': {
                    'X': int(X_outliers),
                    'y': int(y_outliers)
                },
                'train_test_split': {
                    'train': len(X_train),
                    'test': len(X_test)
                }
            }
            
            print(f"  Cleaned data saved to {site_clean_dir}")
            
        except Exception as e:
            print(f"  Error processing site {site_id}: {e}")
    
    # Save stats for all sites to one file (handy for analysis later)
    with open(os.path.join(cleaned_dir, 'cleaning_stats.json'), 'w') as f:
        json.dump(site_stats, f, indent=2)
    
    print(f"\nProcessed and cleaned data for {len(site_stats)} sites")
    print(f"Cleaning statistics saved to {os.path.join(cleaned_dir, 'cleaning_stats.json')}")
    print("\nStep 4 complete: Data cleaning and preparation finished")

if __name__ == "__main__":
    clean_and_prepare_data()
