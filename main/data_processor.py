#!/usr/bin/env python3
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from datetime import timedelta

def process_traffic_data(
    raw_dir='data/raw',
    proc_dir='data/processed',
    window_size=12,
    train_frac=0.8
):
    # 1. Prepare folders
    os.makedirs(proc_dir, exist_ok=True)
    sites_dir = os.path.join(proc_dir, 'sites')
    os.makedirs(sites_dir, exist_ok=True)

    # 2. Load raw data
    excel = os.path.join(raw_dir, 'Scats Data October 2006.xlsx')
    traffic = pd.read_excel(excel, sheet_name='Data', header=1)
    summary = pd.read_excel(excel, sheet_name='Summary Of Data', header=3)
    summary = summary.dropna(subset=['SCATS Number'])
    loc_csv = os.path.join(raw_dir, 'Traffic_Count_Locations_with_LONG_LAT.csv')
    locations = pd.read_csv(loc_csv)

    # 3. Data Exploration
    print("=== Data Exploration ===")
    print("Traffic data shape:", traffic.shape)
    print("Missing values per column:\n", traffic.isnull().sum()[lambda x: x > 0])
    print("Date range:", traffic['Date'].min(), "to", traffic['Date'].max())
    print("Unique SCATS sites:", traffic['SCATS Number'].nunique())

    # Identify volume columns & time labels
    volume_cols = [c for c in traffic.columns if c.startswith('V') and c[1:].isdigit()]
    time_labels = [f"{i//4:02d}:{(i%4)*15:02d}" for i in range(len(volume_cols))]

    metadata = {}
    G = nx.Graph()

    # 4. Process each SCATS site
    for site_id in summary['SCATS Number'].astype(int).unique():
        df = traffic[traffic['SCATS Number'] == site_id].copy()
        if df.empty:
            continue
        df = df.sort_values('Date')
        vols = df[volume_cols].values

        # a) Create folder
        sd = os.path.join(sites_dir, str(site_id))
        os.makedirs(sd, exist_ok=True)

        # b) Daily average pattern
        avg = vols.mean(axis=0)
        dp = pd.DataFrame({'Time': time_labels, 'AvgVolume': avg})
        dp.to_csv(os.path.join(sd, 'daily_pattern.csv'), index=False)
        plt.figure(figsize=(10, 4))
        plt.plot(dp['Time'], dp['AvgVolume'])
        plt.xticks(rotation=90)
        plt.title(f"Site {site_id} – Avg Daily Pattern")
        plt.tight_layout()
        plt.savefig(os.path.join(sd, 'daily_pattern.png'))
        plt.close()

        # c) Time-series data
        X_raw, y_raw, sample_map = [], [], []
        for day_idx in range(len(vols)):
            row = vols[day_idx]
            for j in range(len(row) - window_size):
                X_raw.append(row[j:j+window_size])
                y_raw.append(row[j+window_size])
                sample_map.append((day_idx, j))
        X_raw = np.array(X_raw)
        y_raw = np.array(y_raw)
        np.save(os.path.join(sd, 'X_raw.npy'), X_raw)
        np.save(os.path.join(sd, 'y_raw.npy'), y_raw)

        # 5. Data cleaning
        X = X_raw.copy()
        y = y_raw.copy()
        # NaN handling
        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
        if np.isnan(y).any():
            y = np.nan_to_num(y, nanmean=True)

        # Outlier capping (±3σ)
        mean_X, std_X = X.mean(axis=0), X.std(axis=0)
        X = np.clip(X, mean_X - 3 * std_X, mean_X + 3 * std_X)
        mean_y, std_y = y.mean(), y.std()
        y = np.clip(y, mean_y - 3 * std_y, mean_y + 3 * std_y)

        # Normalize
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = x_scaler.fit_transform(X_flat).reshape(X.shape)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Split into train/test
        split = int(len(X_scaled) * train_frac)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y_scaled[:split], y_scaled[split:]

        # Save cleaned data
        np.save(os.path.join(sd, 'X_train.npy'), X_train)
        np.save(os.path.join(sd, 'X_test.npy'), X_test)
        np.save(os.path.join(sd, 'y_train.npy'), y_train)
        np.save(os.path.join(sd, 'y_test.npy'), y_test)
        with open(os.path.join(sd, 'x_scaler.pkl'), 'wb') as f:
            pickle.dump(x_scaler, f)
        with open(os.path.join(sd, 'y_scaler.pkl'), 'wb') as f:
            pickle.dump(y_scaler, f)

        # 6. Feature engineering
        tf_list = []
        # === Only run once per site ===
        peak_interval_count = 6
        peak_indices = avg.argsort()[-peak_interval_count:]
        peak_mask = np.zeros_like(avg, dtype=bool)
        peak_mask[peak_indices] = True

        # Save site profile just once
        site_profile = {
            "site_id": int(site_id),
            "top_peak_intervals": [int(i) for i in peak_indices[::-1]],
            "top_peak_times": [time_labels[i] for i in peak_indices[::-1]],
            "peak_volumes": [float(avg[i]) for i in peak_indices[::-1]],
            "total_samples": int(len(y_raw))
        }
        with open(os.path.join(sd, 'site_profile.json'), 'w') as f:
            json.dump(site_profile, f, indent=2)

        # === Now build time features using the peak mask ===
        tf_list = []
        for (d, j) in sample_map:
            date = pd.to_datetime(df['Date'].iloc[d])
            interval = j + window_size
            hour = interval // 4
            minute = (interval % 4) * 15
            dow = date.weekday()
            is_weekend = int(dow >= 5)
            is_peak = int(peak_mask[interval])
            tf_list.append([dow, is_weekend, is_peak, hour + minute / 60.0])
        tf_arr = np.array(tf_list)

        # Encode categorical & scale continuous
        encoder = OneHotEncoder(sparse_output=False)
        tf_cat = encoder.fit_transform(tf_arr[:, :4])
        cont_scaler = MinMaxScaler()
        tf_cont = cont_scaler.fit_transform(tf_arr[:, 3].reshape(-1, 1))

        tf_final = np.hstack([tf_cat, tf_cont])
        np.save(os.path.join(sd, 'time_features.npy'), tf_final)
        with open(os.path.join(sd, 'feature_encoder.pkl'), 'wb') as f:
            pickle.dump(encoder, f)
        with open(os.path.join(sd, 'cont_scaler.pkl'), 'wb') as f:
            pickle.dump(cont_scaler, f)

        # 7. Location matching
        rec = locations[locations['TFM_ID'] == site_id]
        if not rec.empty:
            lon = float(rec['X'].iloc[0])
            lat = float(rec['Y'].iloc[0])
            desc = rec['SITE_DESC'].iloc[0]
            G.add_node(site_id, pos=(lon, lat))
        else:
            lon = lat = desc = None

        # Metadata
        metadata[str(site_id)] = {
            'samples': int(X.shape[0]),
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'window_size': window_size,
            'date_range': [str(df['Date'].min().date()), str(df['Date'].max().date())],
            'location': desc,
            'lat': lat,
            'lon': lon
        }
        print(f"→ Site {site_id}: {X.shape[0]} samples ({X_train.shape[0]} train, {X_test.shape[0]} test)")

    # 8. Save metadata and graph
    with open(os.path.join(proc_dir, 'sites_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(proc_dir, 'sites_graph.gpickle'), 'wb') as f:
        pickle.dump(G, f)


    # Plot graph
    if G.nodes:
        plt.figure(figsize=(6, 6))
        for n, data in G.nodes(data=True):
            x, y = data['pos']
            plt.scatter(x, y, c='blue', s=20)
            plt.text(x, y, n, fontsize=6)
        plt.title('SCATS Sites Graph')
        plt.xlabel('Lon')
        plt.ylabel('Lat')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(proc_dir, 'sites_graph.png'))
        plt.close()

    print("\nData processing complete!")

if __name__ == "__main__":
    process_traffic_data()
