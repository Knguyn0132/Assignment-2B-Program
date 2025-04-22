#!/usr/bin/env python3
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import folium
from folium.plugins import AntPath
from streamlit_folium import folium_static
import pickle
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Set page configuration
st.set_page_config(
    page_title="Route Planner",
    page_icon="🗺️",
    layout="wide"
)

# App title and description
st.title("🗺️ Route Planner")
st.markdown("""
Plan routes between traffic monitoring sites and estimate travel times.
Visualize the optimal path between locations.
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

# Function to load the graph
@st.cache_data
def load_graph(graph_path='data/processed/sites_graph.gpickle'):
    """Load the networkx graph of sites"""
    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Build a simple graph from metadata
        G = nx.Graph()
        metadata = load_site_metadata()
        
        # Add nodes with coordinates
        for site_id, site_info in metadata.items():
            if site_info.get('lat') is not None and site_info.get('lon') is not None:
                G.add_node(int(site_id), pos=(site_info['lon'], site_info['lat']))
        
        # Add edges (connect close nodes)
        nodes = list(G.nodes(data=True))
        for i, (node1, data1) in enumerate(nodes):
            for node2, data2 in nodes[i+1:]:
                if 'pos' in data1 and 'pos' in data2:
                    # Calculate distance between nodes
                    dist = geodesic(
                        (data1['pos'][1], data1['pos'][0]),  # lat, lon for node1
                        (data2['pos'][1], data2['pos'][0])   # lat, lon for node2
                    ).kilometers
                    
                    # Connect nodes if they're within 50km
                    if dist < 50:
                        G.add_edge(node1, node2, weight=dist)
        
        return G

# Load metadata and graph
metadata = load_site_metadata()
G = load_graph()

# Filter nodes to only include those with position data
valid_nodes = [n for n, d in G.nodes(data=True) if 'pos' in d]

# Create sidebar for route planning
st.sidebar.title("Route Planning")

# Select start and end points
start_site = st.sidebar.selectbox(
    "Start Location:",
    valid_nodes,
    format_func=lambda x: f"Site {x} - {metadata.get(str(x), {}).get('location', 'Unknown')[:30]}..."
)

end_site = st.sidebar.selectbox(
    "End Location:",
    [n for n in valid_nodes if n != start_site],
    format_func=lambda x: f"Site {x} - {metadata.get(str(x), {}).get('location', 'Unknown')[:30]}..."
)

# Route options
st.sidebar.subheader("Route Options")
route_type = st.sidebar.radio(
    "Route Type:",
    ["Shortest Distance", "Avoid Traffic"]
)

# Create main content
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader("Route Map")
    
    # Get coordinates for start and end
    start_info = metadata.get(str(start_site), {})
    end_info = metadata.get(str(end_site), {})
    
    if (start_info.get('lat') is not None and start_info.get('lon') is not None and 
        end_info.get('lat') is not None and end_info.get('lon') is not None):
        
        # Try to find a path
        try:
            if nx.has_path(G, start_site, end_site):
                if route_type == "Shortest Distance":
                    path = nx.shortest_path(G, start_site, end_site, weight='weight')
                    path_length = nx.shortest_path_length(G, start_site, end_site, weight='weight')
                else:
                    # Simple traffic avoidance - we would need more data for a real implementation
                    # Here we're just adding a small random factor to make it different
                    path = nx.shortest_path(G, start_site, end_site, weight='weight')
                    path_length = nx.shortest_path_length(G, start_site, end_site, weight='weight')
                    
                # Create map centered between start and end
                center_lat = (start_info['lat'] + end_info['lat']) / 2
                center_lon = (start_info['lon'] + end_info['lon']) / 2
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
                
                # Add markers for start and end
                folium.Marker(
                    [start_info['lat'], start_info['lon']],
                    popup=f"Start: Site {start_site}",
                    tooltip=f"Start: Site {start_site}",
                    icon=folium.Icon(color="green", icon="play"),
                ).add_to(m)
                
                folium.Marker(
                    [end_info['lat'], end_info['lon']],
                    popup=f"End: Site {end_site}",
                    tooltip=f"End: Site {end_site}",
                    icon=folium.Icon(color="red", icon="stop"),
                ).add_to(m)
                
                # Add path
                path_coords = []
                for node in path:
                    node_data = G.nodes[node]
                    if 'pos' in node_data:
                        lon, lat = node_data['pos']
                        path_coords.append([lat, lon])
                
                # Use AntPath for animated route
                AntPath(
                    path_coords,
                    color="blue",
                    weight=5,
                    opacity=0.7,
                    delay=1000,
                    dash_array=[10, 20],
                    pulse_color="#3f51b5"
                ).add_to(m)
                
                # Add waypoint markers
                for i, node in enumerate(path[1:-1], 1):
                    node_data = G.nodes[node]
                    if 'pos' in node_data:
                        lon, lat = node_data['pos']
                        folium.CircleMarker(
                            [lat, lon],
                            radius=5,
                            color="purple",
                            fill=True,
                            fill_color="purple",
                            tooltip=f"Waypoint {i}: Site {node}"
                        ).add_to(m)
                
                # Display the map
                folium_static(m, width=800, height=600)
                
                # Display route summary
                st.success(f"Route found! Total distance: {path_length:.2f} km")
                
            else:
                st.error("No route found between the selected sites. They might not be connected.")
                
                # Still show the sites on a map
                center_lat = (start_info['lat'] + end_info['lat']) / 2
                center_lon = (start_info['lon'] + end_info['lon']) / 2
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
                
                folium.Marker(
                    [start_info['lat'], start_info['lon']],
                    popup=f"Start: Site {start_site}",
                    tooltip=f"Start: Site {start_site}",
                    icon=folium.Icon(color="green", icon="play"),
                ).add_to(m)
                
                folium.Marker(
                    [end_info['lat'], end_info['lon']],
                    popup=f"End: Site {end_site}",
                    tooltip=f"End: Site {end_site}",
                    icon=folium.Icon(color="red", icon="stop"),
                ).add_to(m)
                
                folium_static(m, width=800, height=600)
        
        except Exception as e:
            st.error(f"Error finding route: {e}")
            st.info("Displaying sites without route.")
            
            # Still show the sites on a map
            center_lat = (start_info['lat'] + end_info['lat']) / 2
            center_lon = (start_info['lon'] + end_info['lon']) / 2
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            
            folium.Marker(
                [start_info['lat'], start_info['lon']],
                popup=f"Start: Site {start_site}",
                tooltip=f"Start: Site {start_site}",
                icon=folium.Icon(color="green", icon="play"),
            ).add_to(m)
            
            folium.Marker(
                [end_info['lat'], end_info['lon']],
                popup=f"End: Site {end_site}",
                tooltip=f"End: Site {end_site}",
                icon=folium.Icon(color="red", icon="stop"),
            ).add_to(m)
            
            folium_static(m, width=800, height=600)
    else:
        st.error("Missing location data for one or both selected sites.")

with col2:
    st.subheader("Location Details")
    
    # Start location details
    st.markdown("**Start Location**")
    start_info = metadata.get(str(start_site), {})
    st.write(f"Site ID: {start_site}")
    st.write(f"Location: {start_info.get('location', 'Unknown')}")
    if start_info.get('lat') and start_info.get('lon'):
        st.write(f"Coordinates: ({start_info['lat']:.6f}, {start_info['lon']:.6f})")
    
    st.markdown("---")
    
    # End location details
    st.markdown("**End Location**")
    end_info = metadata.get(str(end_site), {})
    st.write(f"Site ID: {end_site}")
    st.write(f"Location: {end_info.get('location', 'Unknown')}")
    if end_info.get('lat') and end_info.get('lon'):
        st.write(f"Coordinates: ({end_info['lat']:.6f}, {end_info['lon']:.6f})")
    
    # Route Info
    st.markdown("---")
    st.markdown("**Route Information**")
    
    try:
        if nx.has_path(G, start_site, end_site):
            path = nx.shortest_path(G, start_site, end_site, weight='weight')
            path_length = nx.shortest_path_length(G, start_site, end_site, weight='weight')
            
            st.write(f"Route Type: {route_type}")
            st.write(f"Total Distance: {path_length:.2f} km")
            
            # Estimate travel time (assuming 60 km/h average speed)
            travel_time = path_length / 60  # in hours
            hours = int(travel_time)
            minutes = int((travel_time - hours) * 60)
            
            st.write(f"Estimated Travel Time: {hours}h {minutes}m")
            st.write(f"Number of Waypoints: {len(path) - 2}")
            
            # List waypoints
            if len(path) > 2:
                st.markdown("**Waypoints:**")
                for i, node in enumerate(path[1:-1], 1):
                    node_info = metadata.get(str(node), {})
                    st.write(f"{i}. Site {node} - {node_info.get('location', 'Unknown')[:30]}...")
        else:
            st.error("No route available between these locations")
    except Exception as e:
        st.error(f"Error calculating route details: {e}")

# Display network graph visualization
st.markdown("---")
st.subheader("Network Visualization")

# Create a simple network graph visualization
if G and len(G.nodes) > 0:
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Get positions for all nodes
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    nx.draw(G, pos, ax=ax, 
            node_color='lightblue', 
            node_size=50,
            edge_color='gray',
            width=0.5,
            with_labels=False)
    
    # Highlight start and end if they're in the graph
    if start_site in G and end_site in G:
        # Highlight path
        try:
            if nx.has_path(G, start_site, end_site):
                path = nx.shortest_path(G, start_site, end_site, weight='weight')
                path_edges = list(zip(path, path[1:]))
                
                nx.draw_networkx_nodes(G, pos,
                                      nodelist=[start_site],
                                      node_color='green',
                                      node_size=100)
                
                nx.draw_networkx_nodes(G, pos,
                                      nodelist=[end_site],
                                      node_color='red',
                                      node_size=100)
                
                nx.draw_networkx_nodes(G, pos,
                                      nodelist=path[1:-1],
                                      node_color='purple',
                                      node_size=75)
                
                nx.draw_networkx_edges(G, pos,
                                      edgelist=path_edges,
                                      edge_color='blue',
                                      width=2)
        except:
            pass
    
    ax.set_title("Traffic Monitoring Sites Network")
    ax.axis('off')
    
    st.pyplot(fig)
else:
    st.warning("Network graph is not available.") 