#!/usr/bin/env python3
import os
import json
import streamlit as st
import folium
from folium.plugins import MarkerCluster, Search
from streamlit_folium import folium_static
import pandas as pd
import requests
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Traffic Data Viewer",
    page_icon="🚦",
    layout="wide"
)

# App title and description
st.title("🚦 Traffic Data Visualization")
st.markdown("""
This application displays traffic monitoring sites on an interactive map. 
You can search for specific locations, view site details, and explore traffic patterns.
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

# Function to geocode an address using the Nominatim API
@st.cache_data
def geocode_location(location_name):
    """Convert a location name to latitude and longitude using Nominatim API"""
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json"
        response = requests.get(url)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        return None, None
    except Exception as e:
        st.error(f"Error geocoding location: {e}")
        return None, None

# Create sidebar for options
st.sidebar.title("Options")

# Search option in sidebar
search_query = st.sidebar.text_input("Search for a location:", "")
if search_query:
    lat, lon = geocode_location(search_query)
    if lat and lon:
        st.sidebar.success(f"Found location at: {lat:.6f}, {lon:.6f}")
        search_coords = (lat, lon)
    else:
        st.sidebar.error("Location not found. Please try a different search term.")
        search_coords = None
else:
    search_coords = None

# Load the site metadata
metadata = load_site_metadata()

# Filter options
st.sidebar.subheader("Filter Sites")
all_sites = list(metadata.keys())
selected_sites = st.sidebar.multiselect(
    "Select specific sites:",
    all_sites,
    default=[]
)

# Create a dataframe for the sites
sites_data = []
for site_id, site_info in metadata.items():
    if not selected_sites or site_id in selected_sites:
        if site_info.get('lat') is not None and site_info.get('lon') is not None:
            sites_data.append({
                'site_id': site_id,
                'location': site_info.get('location', 'Unknown'),
                'lat': site_info.get('lat'),
                'lon': site_info.get('lon'),
                'samples': site_info.get('samples', 0),
                'date_range': site_info.get('date_range', [])
            })

sites_df = pd.DataFrame(sites_data)

# If we have sites with coordinates
if not sites_df.empty:
    # Create two columns for map and details
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # Calculate center of map (average of all points or search location)
        if search_coords:
            center_lat, center_lon = search_coords
        elif not sites_df.empty:
            center_lat = sites_df['lat'].mean()
            center_lon = sites_df['lon'].mean()
        else:
            # Default to Melbourne, Australia if no points
            center_lat, center_lon = -37.8136, 144.9631
        
        # Create a folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
        
        # Add search location marker if any
        if search_coords:
            folium.Marker(
                search_coords,
                popup="Search Location",
                icon=folium.Icon(color="red", icon="search"),
            ).add_to(m)
        
        # Create a marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each site
        for _, site in sites_df.iterrows():
            popup_html = f"""
            <b>Site ID:</b> {site['site_id']}<br>
            <b>Location:</b> {site['location']}<br>
            <b>Samples:</b> {site['samples']}<br>
            <b>Date Range:</b> {' to '.join(site['date_range']) if site['date_range'] else 'Unknown'}<br>
            <button onclick="window.parent.postMessage({{type: 'site_selected', site_id: '{site['site_id']}'}}, '*')">
                View Details
            </button>
            """
            
            folium.Marker(
                [site['lat'], site['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Site {site['site_id']}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(marker_cluster)
        
        # Add tile layers (different map styles)
        folium.TileLayer('openstreetmap').add_to(m)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.LayerControl().add_to(m)
        
        # Display the map
        st.subheader("Traffic Monitoring Sites Map")
        folium_static(m, width=800, height=600)
    
    with col2:
        st.subheader("Site Information")
        
        # Show site details when selected
        selected_site_id = st.selectbox("Select a site to view details:", [""] + list(sites_df['site_id']))
        
        if selected_site_id:
            site_info = metadata.get(selected_site_id, {})
            st.write(f"**Site ID:** {selected_site_id}")
            st.write(f"**Location:** {site_info.get('location', 'Unknown')}")
            st.write(f"**Samples:** {site_info.get('samples', 0)}")
            
            if site_info.get('date_range'):
                st.write(f"**Data Period:** {' to '.join(site_info['date_range'])}")
            
            # Load daily pattern if available
            site_dir = f"data/processed/sites/{selected_site_id}"
            pattern_file = os.path.join(site_dir, 'daily_pattern.csv')
            
            if os.path.exists(pattern_file):
                pattern_df = pd.read_csv(pattern_file)
                
                st.subheader("Daily Traffic Pattern")
                
                # Create a simple bar chart
                st.bar_chart(pattern_df.set_index('Time')['AvgVolume'])
                
                # Show the raw data in a collapsible section
                with st.expander("View Raw Data"):
                    st.dataframe(pattern_df)
            
            # Display site on small map
            if site_info.get('lat') and site_info.get('lon'):
                mini_map = folium.Map(
                    location=[site_info['lat'], site_info['lon']], 
                    zoom_start=15,
                    width=300,
                    height=200
                )
                folium.Marker(
                    [site_info['lat'], site_info['lon']],
                    popup=f"Site {selected_site_id}",
                    icon=folium.Icon(color="green"),
                ).add_to(mini_map)
                st.subheader("Site Location")
                folium_static(mini_map, width=300, height=200)
else:
    st.warning("No sites with location data available to display on map")

# Add a footer
st.markdown("---")
st.markdown("© 2023 Traffic Data Viewer | Developed with Streamlit and Folium") 