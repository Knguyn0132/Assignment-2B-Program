#!/usr/bin/env python3
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import pickle

# Set page configuration
st.set_page_config(
    page_title="Traffic Pattern Analysis",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Traffic Pattern Analysis")
st.markdown("""
Analyze traffic patterns and time series data from various monitoring sites.
Compare patterns across sites and visualize traffic trends.
""")

# Function to load site metadata
@st.cache_data
def load_site_metadata(metadata_path='data/processed/sites_metadata.json'):
    """Load the site metadata from JSON file"""
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        st.error(f"Metadata file not found at {metadata_path}")
        return {}

# Function to load a daily pattern for a specific site
@st.cache_data
def load_daily_pattern(site_id):
    """Load the daily traffic pattern for a specific site"""
    pattern_file = f"data/processed/sites/{site_id}/daily_pattern.csv"
    if os.path.exists(pattern_file):
        return pd.read_csv(pattern_file)
    return None

# Function to load time series data
@st.cache_data
def load_time_series(site_id):
    """Load time series data for a specific site"""
    x_file = f"data/processed/sites/{site_id}/X_train.npy"
    y_file = f"data/processed/sites/{site_id}/y_train.npy"
    
    if os.path.exists(x_file) and os.path.exists(y_file):
        X = np.load(x_file)
        y = np.load(y_file)
        return X, y
    return None, None

# Function to load scaler for denormalization
def load_scaler(site_id):
    """Load the scaler for denormalizing data"""
    scaler_file = f"data/processed/sites/{site_id}/y_scaler.pkl"
    if os.path.exists(scaler_file):
        with open(scaler_file, 'rb') as f:
            return pickle.load(f)
    return None

# Load metadata
metadata = load_site_metadata()
all_sites = list(metadata.keys())

# Sidebar for site selection
st.sidebar.title("Select Sites")
selected_sites = st.sidebar.multiselect(
    "Choose sites to analyze:",
    all_sites,
    default=all_sites[:2] if len(all_sites) >= 2 else all_sites[:1]
)

if not selected_sites:
    st.warning("Please select at least one site to analyze.")
else:
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Daily Patterns", "Time Series", "Site Comparison"])
    
    with tab1:
        st.header("Daily Traffic Patterns")
        
        # Plot daily patterns for selected sites
        if len(selected_sites) > 0:
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            
            for site_id in selected_sites:
                pattern_df = load_daily_pattern(site_id)
                if pattern_df is not None:
                    ax.plot(pattern_df['Time'], pattern_df['AvgVolume'], label=f"Site {site_id}")
            
            ax.set_xlabel("Time of Day")
            ax.set_ylabel("Average Traffic Volume")
            ax.set_title("Daily Traffic Patterns Comparison")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Show peak times
            st.subheader("Peak Traffic Times")
            
            peak_data = []
            for site_id in selected_sites:
                pattern_df = load_daily_pattern(site_id)
                if pattern_df is not None:
                    # Find top 3 peak times
                    top_peaks = pattern_df.nlargest(3, 'AvgVolume')
                    for _, peak in top_peaks.iterrows():
                        peak_data.append({
                            'Site ID': site_id,
                            'Time': peak['Time'],
                            'Volume': round(peak['AvgVolume'], 2)
                        })
            
            peak_df = pd.DataFrame(peak_data)
            st.dataframe(peak_df, use_container_width=True)
    
    with tab2:
        st.header("Time Series Analysis")
        
        # Only show for a single selected site
        if len(selected_sites) == 1:
            site_id = selected_sites[0]
            X, y = load_time_series(site_id)
            
            if X is not None and y is not None:
                st.subheader(f"Time Series Data for Site {site_id}")
                
                # Plot a sample window from the time series
                sample_idx = st.slider("Select sample window:", 0, len(X)-1, 0)
                
                # Get the scaler to denormalize if available
                scaler = load_scaler(site_id)
                
                fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
                
                # Plot the window (input sequence)
                window = X[sample_idx]
                if scaler:
                    # Reshape for inverse transform
                    window_denorm = scaler.inverse_transform(window.reshape(-1, 1)).flatten()
                    next_val_denorm = scaler.inverse_transform([[y[sample_idx]]])[0][0]
                    
                    ax.plot(range(len(window)), window_denorm, 'b-', label='Input Window')
                    ax.plot(len(window), next_val_denorm, 'ro', label='Next Value (Target)')
                else:
                    ax.plot(range(len(window)), window, 'b-', label='Input Window (Normalized)')
                    ax.plot(len(window), y[sample_idx], 'ro', label='Next Value (Normalized)')
                
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("Traffic Volume")
                ax.set_title(f"Time Series Window {sample_idx} for Site {site_id}")
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Display statistics
                st.subheader("Time Series Statistics")
                
                stats = {
                    "Total Samples": len(X),
                    "Window Size": X.shape[1],
                    "Min Value (normalized)": np.min(X),
                    "Max Value (normalized)": np.max(X),
                    "Mean Value (normalized)": np.mean(X),
                }
                
                stats_df = pd.DataFrame([stats])
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.warning(f"No time series data available for site {site_id}")
        else:
            st.info("Please select exactly one site to view time series analysis.")
    
    with tab3:
        st.header("Site Comparison")
        
        # Create a map with all selected sites
        if len(selected_sites) > 0:
            # Gather site coordinates
            coords = []
            for site_id in selected_sites:
                site_info = metadata.get(site_id, {})
                if site_info.get('lat') is not None and site_info.get('lon') is not None:
                    coords.append({
                        'site_id': site_id,
                        'lat': site_info['lat'],
                        'lon': site_info['lon'],
                        'location': site_info.get('location', 'Unknown'),
                    })
            
            if coords:
                # Calculate center of map
                center_lat = sum(c['lat'] for c in coords) / len(coords)
                center_lon = sum(c['lon'] for c in coords) / len(coords)
                
                # Create a folium map
                m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
                
                # Add markers for each site
                for site in coords:
                    popup_text = f"""
                    <b>Site ID:</b> {site['site_id']}<br>
                    <b>Location:</b> {site['location']}<br>
                    """
                    
                    folium.Marker(
                        [site['lat'], site['lon']],
                        popup=popup_text,
                        tooltip=f"Site {site['site_id']}",
                        icon=folium.Icon(color="green", icon="info-sign"),
                    ).add_to(m)
                
                # Add lines connecting sites
                if len(coords) > 1:
                    for i in range(len(coords)):
                        for j in range(i+1, len(coords)):
                            folium.PolyLine(
                                [[coords[i]['lat'], coords[i]['lon']], 
                                [coords[j]['lat'], coords[j]['lon']]],
                                color="blue",
                                weight=2,
                                opacity=0.5,
                                dash_array="5"
                            ).add_to(m)
                
                st.subheader("Selected Sites Map")
                folium_static(m, width=800, height=500)
                
                # Add a comparison table
                st.subheader("Sites Comparison Table")
                
                comparison_data = []
                for site_id in selected_sites:
                    site_info = metadata.get(site_id, {})
                    pattern_df = load_daily_pattern(site_id)
                    
                    if pattern_df is not None:
                        max_vol = pattern_df['AvgVolume'].max()
                        avg_vol = pattern_df['AvgVolume'].mean()
                        peak_time = pattern_df.loc[pattern_df['AvgVolume'].idxmax(), 'Time']
                    else:
                        max_vol = avg_vol = peak_time = None
                    
                    comparison_data.append({
                        'Site ID': site_id,
                        'Location': site_info.get('location', 'Unknown'),
                        'Samples': site_info.get('samples', 0),
                        'Avg Volume': round(avg_vol, 2) if avg_vol is not None else None,
                        'Max Volume': round(max_vol, 2) if max_vol is not None else None,
                        'Peak Time': peak_time
                    })
                
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True)
            else:
                st.warning("No location data available for the selected sites.") 