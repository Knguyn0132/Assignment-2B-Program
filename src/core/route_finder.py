import random
from src.problem.vicroads_graph_problem import VicRoadsGraphProblem

from src.algorithms.search_graph import SearchGraph
from src.algorithms.astar import AStar
from src.algorithms.dfs import DFS
from src.algorithms.bfs import BFS
from src.algorithms.gbfs import GBFS
from src.algorithms.ucs import UCS
from src.algorithms.fringe import Fringe
from src.graph.graph import Graph
from datetime import datetime, time, date
import streamlit as st
import pandas as pd
import numpy as np
import math



class RouteFinder:
    """
    Class for finding optimal routes between SCATS sites using various search algorithms
    """
    def __init__(self, network):
        """
        Initialize with a SiteNetwork object
        """
        self.network = network
        self.graph = self._create_search_graph()
        self.complete_oct_nov_dataframes = self._load_dataframes()

    def _load_dataframes(self):
        """
        Load dataframes for October and November that have predicted travel times
        by LSTM, GRU, and CNN_LSTM models
        """
        lstm_df = pd.read_csv(
            "processed_data/complete_csv_oct_nov_2006/lstm_model/lstm_model_complete_data.csv"
        )

        gru_df = pd.read_csv(
            "processed_data/complete_csv_oct_nov_2006/gru_model/gru_model_complete_data.csv"
        )
        cnn_lstm_df = pd.read_csv(
            "processed_data/complete_csv_oct_nov_2006/cnn_lstm_model/cnn_lstm_model_complete_data.csv"
        )

        return {
            "LSTM": lstm_df,
            "GRU": gru_df,
            "CNN_LSTM": cnn_lstm_df,
        }
    
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
    
    def _create_search_graph(
        self, origin, destination, prediction_model="LSTM", datetime_str=None
    ):
        """
        Convert SiteNetwork to SearchGraph format for search algorithms
        datetime_str is in the format like this "2006-11-30 14:15"
        """
        graph_dict = {}
        locations = {}

        # Add all site IDs as nodes with their coordinates
        for site_id, site in self.network.sites_data.items():
            site_id = int(site_id)

            locations[site_id] = (site["latitude"], site["longitude"])

            # Initialize empty dict for each node
            if site_id not in graph_dict:
                graph_dict[site_id] = {}

        # Process datetime to get Date and interval_id if provided
        traffic_volume_lookup = {}
        if datetime_str:
            try:
                # Parse datetime string
                dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                date_str = dt.date().strftime("%Y-%m-%d")

                # Calculate interval_id (0-95) based on time
                # Each interval is 15 minutes, so 4 intervals per hour
                hour = dt.hour
                minute = dt.minute
                interval_id = (hour * 4) + (minute // 15)

                # Get the appropriate dataframe based on the prediction model
                df = self.complete_oct_nov_dataframes.get(prediction_model)
                if df is not None:
                    # Create lookup table for traffic volumes by location
                    for _, row in df[
                        (df["Date"] == date_str) & (df["interval_id"] == interval_id)
                    ].iterrows():
                        location = row["Location"].strip()
                        traffic_volume = int(row["traffic_volume"])
                        traffic_volume_lookup[location] = traffic_volume
            except (ValueError, KeyError):
                # Handle date parsing errors or missing dataframe
                st.warning(f"Could not process datetime: {datetime_str}")
                pass

        # print("Length of total traffic volume retrieved: ", len(traffic_volume_lookup))

        # Add connections as edges with costs
        for conn in self.network.connections:
            from_id = int(conn["from_id"])
            to_id = int(conn["to_id"])
            distance = conn["distance"]  # in km

            # Retrieve traffic volume from the dataframe
            location = conn.get("approach_location", "").strip()

            traffic_volume = traffic_volume_lookup[location]

            # Calculate travel time in minutes
            travel_time = self._calculate_travel_time(distance, traffic_volume)

            # Add edge to the graph
            if from_id not in graph_dict:
                graph_dict[from_id] = {}

            graph_dict[from_id][to_id] = travel_time

        return traffic_volume_lookup, VicRoadsGraphProblem(
            origin, destination, Graph(graph_dict), locations
        )

    def _calculate_travel_time(self, distance, traffic_volume):
        """
        Calculate travel time based on distance and traffic volume
        """
        a, b, c = -1.4648375, 93.75, -traffic_volume
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
                    'to_lng': connection['to_lng']
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