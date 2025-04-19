# Import all needed libraries
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pickle
from datetime import datetime, timedelta

def engineer_features():
    """
    Step 5: Create extra features (like hour, day, peak hour flags)
    so the machine learning model can learn patterns more easily.
    # In this step, we engineer 38 additional time-based features to help the model better understand traffic patterns.
    # These include:
    # - One-hot encoded hour of day (hour_0 to hour_23): captures daily traffic trends (24)
    # - One-hot encoded day of the week (day_of_week_0 to day_of_week_6): captures weekly behavior (e.g. weekdays vs weekends) (7)
    # - Binary weekend flag (is_weekend_0/1): helps the model detect different traffic flow on weekends (2)
    # - Binary peak hour flags (is_peak_morning_0/1 and is_peak_evening_0/1): highlight common rush hour periods (4)
    # - A continuous 'hour_of_day' feature (e.g. 7.25 for 7:15 AM): captures time as a smooth trend across the day (1)
    # These features add time-awareness to the data, so the ML model can learn not just "what the traffic was before," 
    # but also "when" the data was recorded — helping it make more accurate, context-aware predictions.

    """
    print("Step 5: Engineering additional features for ML models...\n")
    
    # Set folder paths
    processed_dir = os.path.join('data', 'processed')
    cleaned_dir = os.path.join(processed_dir, 'cleaned')    # Where Step 4 saved cleaned data
    feature_dir = os.path.join(processed_dir, 'featured')   # This is where new features will go
    os.makedirs(feature_dir, exist_ok=True)
    
    # Load SCATS site metadata (list of sites to process)
    with open(os.path.join(processed_dir, 'sites_metadata.json'), 'r') as f:
        sites_metadata = json.load(f)
    
    # Load the original Excel file (contains timestamps)
    data_dir = os.path.join('data', 'raw')
    scats_data_path = os.path.join(data_dir, 'Scats Data October 2006.xlsx')
    traffic_data = pd.read_excel(scats_data_path, sheet_name='Data', header=1)
    
    # Dictionary to store stats for each site
    feature_stats = {}
    
    # Process each site one at a time
    for site_id, metadata in sites_metadata.items():
        print(f"\nEngineering features for SCATS site {site_id}")
        
        # Define input and output folders for this site
        site_clean_dir = os.path.join(cleaned_dir, site_id)
        site_feature_dir = os.path.join(feature_dir, site_id)
        os.makedirs(site_feature_dir, exist_ok=True)
        
        # Check that cleaned training/testing files exist
        x_train_path = os.path.join(site_clean_dir, 'X_train.npy')
        y_train_path = os.path.join(site_clean_dir, 'y_train.npy')
        x_test_path = os.path.join(site_clean_dir, 'X_test.npy')
        y_test_path = os.path.join(site_clean_dir, 'y_test.npy')
        
        # If any cleaned data is missing, skip this site
        if not all(os.path.exists(p) for p in [x_train_path, y_train_path, x_test_path, y_test_path]):
            print(f"  Missing cleaned data for site {site_id}, skipping")
            continue
        
        try:
            # Load the cleaned traffic data
            X_train = np.load(x_train_path)
            y_train = np.load(y_train_path)
            X_test = np.load(x_test_path)
            y_test = np.load(y_test_path)
            
            # Load the scalers from step 4
            with open(os.path.join(site_clean_dir, 'X_scaler.pkl'), 'rb') as f:
                X_scaler = pickle.load(f)
            with open(os.path.join(site_clean_dir, 'y_scaler.pkl'), 'rb') as f:
                y_scaler = pickle.load(f)
            
            print(f"  Loaded cleaned data: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # Extract raw data just for this SCATS site
            site_data = traffic_data[traffic_data['SCATS Number'] == int(site_id)].copy()
            site_data = site_data.sort_values('Date')  # Sort by date to keep time order
            
            # We want to generate time-based features for every 15-minute interval
            # There are 96 intervals in one full day (24*4)
            time_df = pd.DataFrame()
            intervals = []
            hours = []
            minutes = []
            hour_of_day = []
            is_peak_morning = []
            is_peak_evening = []

            # Loop through all 96 intervals of a day to fill time info
            for i in range(96):
                interval_hour = i // 4              # Each 4 intervals = 1 hour
                interval_minute = (i % 4) * 15       # Each step = 15 minutes

                intervals.append(i)
                hours.append(interval_hour)
                minutes.append(interval_minute)
                hour_of_day.append(interval_hour + interval_minute/60.0)

                # Morning peak: 7–9am → set to 1 if within that
                is_peak_am = 1 if (interval_hour >= 7 and interval_hour < 9) else 0
                # Evening peak: 4–6pm → set to 1 if within that
                is_peak_pm = 1 if (interval_hour >= 16 and interval_hour < 18) else 0

                is_peak_morning.append(is_peak_am)
                is_peak_evening.append(is_peak_pm)
            
            # Add all columns to the time_df
            time_df['interval'] = intervals
            time_df['hour'] = hours
            time_df['minute'] = minutes
            time_df['hour_of_day'] = hour_of_day
            time_df['is_peak_morning'] = is_peak_morning
            time_df['is_peak_evening'] = is_peak_evening
            
            # Figure out how many days of data we have
            start_date = site_data['Date'].min().date()
            days = (site_data['Date'].max().date() - start_date).days + 1

            # Create time features for every 15-minute slot of every day
            all_features = []
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                day_features = time_df.copy()

                # Add date-specific features
                day_features['day'] = current_date.day
                day_features['month'] = current_date.month
                day_features['day_of_week'] = current_date.weekday()  # Monday = 0
                day_features['is_weekend'] = 1 if current_date.weekday() >= 5 else 0

                all_features.append(day_features)
            
            # Combine all daily feature data into one big DataFrame
            all_features_df = pd.concat(all_features, ignore_index=True)

            print("  Creating time-based features")

            # These columns will be one-hot encoded
            cat_features = ['hour', 'day_of_week', 'is_weekend', 'is_peak_morning', 'is_peak_evening']

            # Random sample of 1000 rows from time features (for testing/demo)
            sample_features = all_features_df.sample(n=min(len(all_features_df), 1000))

            # Create encoder and fit it on sample data
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False)
            encoder.fit(sample_features[cat_features])

            # Save the encoder to use later
            with open(os.path.join(site_feature_dir, 'feature_encoder.pkl'), 'wb') as f:
                pickle.dump(encoder, f)
            
            # Use a small sample of training data for demo
            n_samples = min(100, len(X_train))
            X_sample = X_train[:n_samples]

            # Pick synthetic time features (same length as sample) to demonstrate
            synth_time = sample_features.sample(n=n_samples, replace=True).reset_index(drop=True)

            # One-hot encode the categorical time features
            encoded_cats = encoder.transform(synth_time[cat_features])

            # Build column names for one-hot encoded data
            encoded_names = []
            for i, feature in enumerate(cat_features):
                categories = encoder.categories_[i]
                for cat in categories:
                    encoded_names.append(f"{feature}_{cat}")
            
            # Convert one-hot encoded array to a DataFrame
            encoded_df = pd.DataFrame(encoded_cats, columns=encoded_names)

            # Add continuous feature (hour_of_day)
            continuous_features = ['hour_of_day']
            for feature in continuous_features:
                encoded_df[feature] = synth_time[feature]

            # Normalize continuous feature(s)
            from sklearn.preprocessing import MinMaxScaler
            cont_scaler = MinMaxScaler()
            encoded_df[continuous_features] = cont_scaler.fit_transform(encoded_df[continuous_features])

            # Save the scaler for continuous values
            with open(os.path.join(site_feature_dir, 'continuous_scaler.pkl'), 'wb') as f:
                pickle.dump(cont_scaler, f)

            # ==== Plot and show which features matter most ====
            plt.figure(figsize=(14, 6))

            corr_values = []
            feature_names = encoded_df.columns.tolist()

            # Calculate correlation (relationship) between each feature and y (traffic volume)
            for feature in feature_names:
                if encoded_df[feature].nunique() <= 1:
                    continue  # skip constant features
                corr = np.corrcoef(encoded_df[feature], y_train[:n_samples])[0, 1]
                corr_values.append((feature, abs(corr)))

            # Sort from strongest to weakest correlation
            corr_values.sort(key=lambda x: x[1], reverse=True)

            # Plot the top 15 features
            top_features = corr_values[:15]
            names, values = zip(*top_features)
            plt.barh(names, values)
            plt.title(f'Top 15 Feature Correlations with Target - Site {site_id}')
            plt.xlabel('Absolute Correlation')
            plt.tight_layout()
            plt.savefig(os.path.join(site_feature_dir, 'feature_correlations.png'))
            plt.close()

            # Save correlation values to JSON
            feature_importance = {name: float(value) for name, value in corr_values}
            with open(os.path.join(site_feature_dir, 'feature_importance.json'), 'w') as f:
                json.dump(feature_importance, f, indent=2)

            print("  Creating sample of enhanced data")

            # Save feature column names
            with open(os.path.join(site_feature_dir, 'feature_columns.json'), 'w') as f:
                json.dump({
                    'categorical_features': cat_features,
                    'continuous_features': continuous_features,
                    'encoded_features': encoded_names
                }, f, indent=2)

            # Save stats summary for this site
            feature_stats[site_id] = {
                'original_features': X_train.shape[1],
                'engineered_features': len(encoded_df.columns),
                'total_features': X_train.shape[1] + len(encoded_df.columns),
                'top_features': [name for name, _ in top_features[:5]]
            }

            print(f"  Feature engineering completed with {len(encoded_df.columns)} new features")
            print(f"  Results saved to {site_feature_dir}")

        # If anything crashes, print error and keep going
        except Exception as e:
            print(f"  Error engineering features for site {site_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save all stats from all sites
    with open(os.path.join(feature_dir, 'feature_stats.json'), 'w') as f:
        json.dump(feature_stats, f, indent=2)

    print(f"\nFeature engineering completed for {len(feature_stats)} sites")
    print(f"Statistics saved to {os.path.join(feature_dir, 'feature_stats.json')}")
    print("\nStep 5 complete: Feature engineering finished")

# Run the function
if __name__ == "__main__":
    engineer_features()
