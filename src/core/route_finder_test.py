import random
from src.algorithms.search_graph import SearchGraph
from src.algorithms.astar import AStar
from src.algorithms.dfs import DFS
from src.algorithms.bfs import BFS
from src.algorithms.gbfs import GBFS
from src.algorithms.ucs import UCS
from src.algorithms.fringe import Fringe
from datetime import datetime, time, date
import streamlit as st
import pandas as pd
import math
import os

class RouteFinder:
    """
    Class for finding optimal routes between SCATS sites using various search algorithms
    """
    def __init__(self, network):
        """
        Initialize with a SiteNetwork object
        """
        self.network = network
        self.graph = None
        self.model_dataframes = self._load_dataframes()
        self.traffic_volume_lookup = {}  # Store traffic volume lookup
        
    def _load_dataframes(self):
        """
        Load traffic prediction data from CSV files
        """
        model_data = {}
        base_path = "processed_data/complete_csv_oct_nov_2006"
        
        # Load LSTM model data
        lstm_path = os.path.join(base_path, "lstm_model/lstm_model_complete_data.csv")
        if os.path.exists(lstm_path):
            model_data["LSTM"] = pd.read_csv(lstm_path)
            # Set column names if not already set
            if model_data["LSTM"].columns[0] != "Location":
                model_data["LSTM"].columns = ["Location", "Date", "interval_id", "traffic_volume", "data_source"]
            
        # Load GRU model data
        gru_path = os.path.join(base_path, "gru_model/gru_model_complete_data.csv")
        if os.path.exists(gru_path):
            model_data["GRU"] = pd.read_csv(gru_path)
            # Set column names if not already set
            if model_data["GRU"].columns[0] != "Location":
                model_data["GRU"].columns = ["Location", "Date", "interval_id", "traffic_volume", "data_source"]
            
        # Load CNN_LSTM model data
        cnn_lstm_path = os.path.join(base_path, "cnn_lstm_model/cnn_lstm_model_complete_data.csv")
        if os.path.exists(cnn_lstm_path):
            model_data["Custom"] = pd.read_csv(cnn_lstm_path)
            # Set column names if not already set
            if model_data["Custom"].columns[0] != "Location":
                model_data["Custom"].columns = ["Location", "Date", "interval_id", "traffic_volume", "data_source"]
            
        return model_data
    
    def _get_algorithm(self, name):
        """
        Get the specified search algorithm instance
        """
        if name == "A*":
            return AStar(self.graph)
        elif name == "DFS":
            return DFS(self.graph)
        elif name == "BFS":
            return BFS(self.graph)
        elif name == "GBFS":
            return GBFS(self.graph)
        elif name == "UCS":
            return UCS(self.graph)
        elif name == "Fringe":
            return Fringe(self.graph)
        return None
    
    def _create_search_graph(self, prediction_model="LSTM", datetime_str=None):
        """
        Convert SiteNetwork to SearchGraph format for search algorithms
        """
        try:
            graph = SearchGraph()
            
            # Parse datetime if provided
            date_obj = None
            interval_id = None
            if datetime_str:
                try:
                    dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                    date_obj = dt.date()
                    # Convert time to interval_id (assuming intervals are in 15-minute increments)
                    hour = dt.hour
                    minute = dt.minute
                    interval_id = (hour * 4) + (minute // 15)
                except ValueError:
                    # If parsing fails, use current date
                    date_obj = datetime.now().date()
                    current_hour = datetime.now().hour
                    current_minute = datetime.now().minute
                    interval_id = (current_hour * 4) + (current_minute // 15)
            else:
                # If no datetime is provided, use current date
                date_obj = datetime.now().date()
                current_hour = datetime.now().hour
                current_minute = datetime.now().minute
                interval_id = (current_hour * 4) + (current_minute // 15)
            
            date_str = date_obj.strftime("%Y-%m-%d")
            
            # Create a traffic volume lookup dictionary
            self.traffic_volume_lookup = {}
            
            # Get the appropriate dataframe based on the prediction model
            df = self.model_dataframes.get(prediction_model)
            if df is not None:
                try:
                    # Filter the dataframe by date and interval_id if available
                    # Use the most recent available date if exact date not found
                    available_dates = df["Date"].unique()
                    if date_str in available_dates:
                        filtered_df = df[df["Date"] == date_str]
                    elif len(available_dates) > 0:
                        # Sort dates and get the closest available date
                        available_dates = sorted(available_dates)
                        # Use the most recent date
                        closest_date = available_dates[-1]
                        filtered_df = df[df["Date"] == closest_date]
                        print(f"Using closest available date: {closest_date}")
                    else:
                        filtered_df = df
                    
                    # Further filter by interval_id if applicable
                    if interval_id is not None and "interval_id" in filtered_df.columns:
                        # Find closest interval_id if exact match not available
                        available_intervals = filtered_df["interval_id"].unique()
                        if len(available_intervals) > 0:
                            closest_interval = min(available_intervals, key=lambda x: abs(int(x) - interval_id))
                            interval_filtered_df = filtered_df[filtered_df["interval_id"] == closest_interval]
                            if not interval_filtered_df.empty:
                                filtered_df = interval_filtered_df
                    
                    # Create traffic volume lookup by location
                    for _, row in filtered_df.iterrows():
                        location = row["Location"]  # Use named column
                        traffic_volume = float(row["traffic_volume"])  # Use named column
                        self.traffic_volume_lookup[location] = traffic_volume
                except Exception as e:
                    print(f"Error processing traffic data: {e}")
            
            # Add all site IDs as nodes with their coordinates
            for site_id, site in self.network.sites_data.items():
                site_id = int(site_id)
                
                graph.node_coordinates[site_id] = (site['latitude'], site['longitude'])

                # Initialize empty adjacency list for each node
                if site_id not in graph.adjacency_list:
                    graph.adjacency_list[site_id] = []
            
            # Add connections as edges with costs
            for conn in self.network.connections:
                from_id = conn['from_id']
                to_id = conn['to_id']
                distance = conn['distance']  # in km
                
                # Get approach location to find traffic volume
                approach_location = conn.get('approach_location', '')
                traffic_volume = self.traffic_volume_lookup.get(approach_location, None)
                
                # Calculate travel time using traffic volume
                travel_time = self._calculate_travel_time(distance, traffic_volume)
                    
                # Add edge to the graph
                if from_id not in graph.adjacency_list:
                    graph.adjacency_list[from_id] = []
                
                graph.adjacency_list[from_id].append((to_id, travel_time))
            
            return graph
        except Exception as e:
            print(f"Error creating search graph: {e}")
            return SearchGraph()  # Return an empty graph if there's an error
        
    def _calculate_travel_time(self, distance, traffic_volume):
        """
        Calculate travel time based on distance and traffic volume
        Uses a quadratic relationship between traffic volume and speed
        """
        # Default values if traffic data is not available
        if traffic_volume is None or traffic_volume <= 0:
            # Use a default medium traffic flow
            traffic_volume = 100
        
        # Parameters for the quadratic equation: speed = ax² + bx + c, where x is traffic volume
        a = -1.4648375  # Coefficient for traffic_volume²
        b = 93.75    # Coefficient for traffic_volume
        c = -traffic_volume
        d = b * b - (4 * a * c)
        speed = (-b + math.sqrt(d)) / (2 * a)  # km/h
        speed = min(speed, 60)  # Cap speed at 60 km/h
        speed = max(speed, 5)  # Minimum speed of 5 km/h

        # Convert to minutes and add 30 seconds for intersection delay
        travel_time = (distance / speed) * 60 + 30 / 60
        return travel_time
    
    def find_multiple_routes(self, origin_id, destination_id, selected_algorithms=None, prediction_model="LSTM", datetime_str=None):
        """
        Find routes from origin to destination using multiple algorithms
        Returns a list of routes with their details
        """
        # Use all algorithms if none specified
        all_algorithms = ["A*", "DFS", "BFS", "GBFS", "UCS", "Fringe"]
        if selected_algorithms is None or "All" in selected_algorithms:
            selected_algorithms = all_algorithms
        
        # Create the graph ONCE for all algorithms
        self.graph = self._create_search_graph(prediction_model, datetime_str)
        
        routes = []

        # Run each selected algorithm
        for alg_name in selected_algorithms:
            # Now use the already created graph, don't create a new one
            path, total_cost, route_info = self.find_best_route(
                origin_id, destination_id, alg_name
            )

            if path:
                routes.append({
                    'algorithm': alg_name,
                    'path': path,
                    'total_cost': total_cost,
                    'route_info': route_info,
                    'traffic_level': "",  # Will assign later
                    'prediction_model': prediction_model,
                    'datetime': datetime_str
                })
        
        # Sort routes by total cost (travel time)
        routes.sort(key=lambda x: x['total_cost'])
        
        # Assign colors based on relative performance (best to worst)
        route_colors = ["green", "yellow", "orange", "red", "darkred"]
        route_descriptions = [
            "Best route", 
            "Second best", 
            "Third best", 
            "Fourth best", 
            "Fifth best"
        ]
        
        for i, route in enumerate(routes[:5]):
            color_index = min(i, len(route_colors)-1)
            route['traffic_level'] = route_colors[color_index]
            route['route_rank'] = route_descriptions[color_index]
        
        # Limit to at most 5 routes
        return routes[:5]

    def find_best_route(self, origin_id, destination_id, algorithm="All"):
        """
        Find the best route using an already created graph
        """
        # Set the origin and destination in the graph
        self.graph.origin = origin_id
        self.graph.destinations = {destination_id}
        
        # Get the search algorithm instance
        search_alg = self._get_algorithm(algorithm)
        
        if not search_alg:
            return None, None, None
        
        # Execute search algorithm
        goal, nodes_expanded, path = search_alg.search(origin_id, [destination_id])
        
        # If no path is found, return None
        if not path:
            return None, None, None
        
        # Calculate route information
        return path, *self._calculate_route_details(path)
    
    def _calculate_route_details(self, path):
        """
        Calculate total travel time and create a list of steps for a path
        """
        total_cost = 0
        route_info = []
        
        for i in range(len(path) - 1):
            from_id = path[i]
            to_id = path[i + 1]
            
            # Find the connection between these sites
            connection = self._find_connection(from_id, to_id)
            
            if connection:
                # Get the actual travel time from the graph
                # (This already includes traffic and intersection delay)
                travel_time = None
                for neighbor, cost in self.graph.adjacency_list.get(from_id, []):
                    if neighbor == to_id:
                        travel_time = cost
                        break
                
                if travel_time is None:
                    # Fallback calculation if not found in graph
                    distance = connection['distance']
                    travel_time = (distance / 45) * 60 + 0.5  # Use average speed
                    
                # Add to total cost
                total_cost += travel_time
                
                # Get approach location to find traffic volume
                approach_location = connection.get('approach_location', '')
                
                # Get traffic volume from lookup dictionary
                traffic_volume = self.traffic_volume_lookup.get(approach_location)
                
                # If traffic_volume is still None, calculate it from observed speed
                if traffic_volume is None:
                    # Try to reverse-engineer the traffic volume based on travel time
                    # This ensures there's always a meaningful traffic volume value
                    speed = (connection['distance'] / (travel_time - 0.5)) * 60  # in km/h
                    
                    if speed >= 54:  # Almost free flow (90% of free flow speed)
                        traffic_volume = 50
                    elif speed >= 48:  # 80% of free flow
                        traffic_volume = 75
                    elif speed >= 42:  # 70% of free flow
                        traffic_volume = 125
                    elif speed >= 36:  # 60% of free flow
                        traffic_volume = 150
                    elif speed >= 30:  # 50% of free flow
                        traffic_volume = 175
                    elif speed >= 24:  # 40% of free flow
                        traffic_volume = 200
                    elif speed >= 18:  # 30% of free flow
                        traffic_volume = 225
                    elif speed >= 12:  # 20% of free flow
                        traffic_volume = 250
                    else:  # Highly congested
                        traffic_volume = 275
                
                # Add step info
                route_info.append({
                    'from_id': from_id,
                    'to_id': to_id,
                    'road': connection['shared_road'],
                    'distance': connection['distance'],
                    'travel_time': travel_time,
                    'from_lat': connection['from_lat'],
                    'from_lng': connection['from_lng'],
                    'to_lat': connection['to_lat'],
                    'to_lng': connection['to_lng'],
                    'traffic_volume': traffic_volume  # Add traffic volume to the route info
                })
        
        return total_cost, route_info
    
    def _find_connection(self, from_id, to_id):
        """
        Find a connection between two sites
        """
        for conn in self.network.connections:
            if conn['from_id'] == from_id and conn['to_id'] == to_id:
                return conn
        return None 