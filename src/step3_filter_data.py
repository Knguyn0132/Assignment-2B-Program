import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

def filter_process_boroondara_data():
    """
    Step 3: Filter and process data for all Boroondara SCATS sites
    """
    print("Step 3: Filtering and processing Boroondara traffic data...\n")
    
    # File paths
    data_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    sites_dir = os.path.join(processed_dir, 'sites')
    os.makedirs(sites_dir, exist_ok=True)
    
    # Load the traffic data
    scats_data_path = os.path.join(data_dir, 'Scats Data October 2006.xlsx')
    traffic_data = pd.read_excel(scats_data_path, sheet_name='Data', header=1)
    
    # Load the locations data
    locations_path = os.path.join(data_dir, 'Traffic_Count_Locations_with_LONG_LAT.csv')
    locations_data = pd.read_csv(locations_path)
    
    # Load the list of Boroondara SCATS sites
    boroondara_sites_path = os.path.join(processed_dir, 'boroondara_scats_sites.csv')
    boroondara_sites = pd.read_csv(boroondara_sites_path)['SCATS_Site'].astype(int).tolist()
    
    print(f"Processing data for {len(boroondara_sites)} Boroondara SCATS sites")
    
    # Extract volume columns (V00 to V95)
    volume_cols = [col for col in traffic_data.columns if col.startswith('V') and col[1:].isdigit()]
    
    # Create time labels (00:00, 00:15, ..., 23:45)
    time_labels = []
    for i in range(96):  # 96 15-minute intervals in a day
        hour = i // 4
        minute = (i % 4) * 15
        time_labels.append(f"{hour:02d}:{minute:02d}")
    
    # Dictionary to store site metadata
    sites_metadata = {}
    
    # Process each SCATS site
    for site_id in boroondara_sites:
        print(f"\nProcessing SCATS site {site_id}")
        
        # 1. Filter data for this site
        site_data = traffic_data[traffic_data['SCATS Number'] == site_id].copy()
        
        if len(site_data) == 0:
            print(f"  No data found for site {site_id}, skipping")
            continue
        
        # 2. Get site location information
        site_location = site_data['Location'].iloc[0] if 'Location' in site_data.columns else "Unknown"
        site_lat = site_data['NB_LATITUDE'].iloc[0] if 'NB_LATITUDE' in site_data.columns else None
        site_lon = site_data['NB_LONGITUDE'].iloc[0] if 'NB_LONGITUDE' in site_data.columns else None
        
        print(f"  Location: {site_location}")
        print(f"  Data points: {len(site_data)}")
        
        # 3. Calculate daily traffic pattern
        daily_pattern = site_data[volume_cols].mean().reset_index()
        daily_pattern.columns = ['Interval', 'Average Volume']
        
        if len(daily_pattern) == len(time_labels):
            daily_pattern['Time'] = time_labels
            
            # Save daily pattern to CSV
            site_dir = os.path.join(sites_dir, str(site_id))
            os.makedirs(site_dir, exist_ok=True)
            daily_pattern.to_csv(os.path.join(site_dir, 'daily_pattern.csv'), index=False)
            
            # Create visualization of daily pattern
            plt.figure(figsize=(12, 6))
            plt.plot(daily_pattern['Time'], daily_pattern['Average Volume'])
            plt.title(f'Average Daily Traffic Pattern - SCATS Site {site_id}')
            plt.xlabel('Time of Day')
            plt.ylabel('Average Volume (vehicles per 15 min)')
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(site_dir, 'daily_pattern.png'))
            plt.close()
        
                # 4. Create time series data for ML model training
        # Sort data by date to ensure chronological order
        site_data = site_data.sort_values('Date')

        # Extract volume data: shape = (days, 96 intervals)
        volume_data = site_data[volume_cols].values

        # Create sliding windows for time series prediction
        X_data = []
        y_data = []
        window_size = 12  # Use 12 previous 15-min intervals to predict the next one (same as Step 2)

        # Loop through all days
        for i in range(len(volume_data)):
            # Loop through intervals in a day using sliding window
            for j in range(len(volume_data[i]) - window_size):
                # Input = 10 previous intervals
                X_data.append(volume_data[i][j:j + window_size])

                # Output = next interval
                y_data.append(volume_data[i][j + window_size])

        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)

        if len(X) > 0:
            print(f"  Created time series data with shape: X={X.shape}, y={y.shape}")

            # Save time series data
            np.save(os.path.join(site_dir, 'X_data.npy'), X)
            np.save(os.path.join(site_dir, 'y_data.npy'), y)
        else:
            print(f"  Unable to create time series data for site {site_id}")
        
        # 5. Try to match with location data in the CSV
        site_in_locations = locations_data[locations_data['TFM_ID'] == site_id]
        if not site_in_locations.empty:
            # Use coordinates from locations data if available
            site_lon = site_in_locations['X'].iloc[0]
            site_lat = site_in_locations['Y'].iloc[0]
            site_description = site_in_locations['SITE_DESC'].iloc[0]
            print(f"  Found in locations data: {site_description}")
        
        # 6. Store metadata
        sites_metadata[str(site_id)] = {
            'site_id': int(site_id),
            'location': site_location,
            'latitude': float(site_lat) if site_lat is not None else None,
            'longitude': float(site_lon) if site_lon is not None else None,
            'data_points': len(site_data),
            'data_period': {
                'start': site_data['Date'].min().strftime('%Y-%m-%d') if 'Date' in site_data.columns else None,
                'end': site_data['Date'].max().strftime('%Y-%m-%d') if 'Date' in site_data.columns else None
            },
            'time_series_shape': {
                'X': X.shape if len(X) > 0 else None,
                'y': y.shape if len(y) > 0 else None
            }
        }
    
    # Save sites metadata
    with open(os.path.join(processed_dir, 'sites_metadata.json'), 'w') as f:
        json.dump(sites_metadata, f, indent=2)
    
    print(f"\nProcessed {len(sites_metadata)} SCATS sites successfully")
    print(f"Data saved to {sites_dir}")
    
    # Create a map of all Boroondara SCATS sites
    try:
        plt.figure(figsize=(12, 10))
        
        # Get coordinates for all sites
        site_coords = []
        for site_id, metadata in sites_metadata.items():
            if metadata['latitude'] is not None and metadata['longitude'] is not None:
                site_coords.append((metadata['longitude'], metadata['latitude'], site_id))
        
        if site_coords:
            # Plot all sites
            lons, lats, ids = zip(*site_coords)
            plt.scatter(lons, lats, s=50, c='red', alpha=0.7)
            
            # Add site labels
            for lon, lat, site_id in site_coords:
                plt.text(lon, lat, site_id, fontsize=8)
            
            plt.title('Boroondara SCATS Sites Locations')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(processed_dir, 'boroondara_scats_map.png'))
            plt.close()
            print(f"Created map of all Boroondara SCATS sites")
    except Exception as e:
        print(f"Error creating map: {e}")
    
    print("\nStep 3 complete: Boroondara data filtering and processing finished")

if __name__ == "__main__":
    filter_process_boroondara_data()