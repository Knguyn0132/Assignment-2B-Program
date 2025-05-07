import random
from algorithms.search_graph import SearchGraph
from algorithms.astar import AStar
from algorithms.dfs import DFS
from algorithms.bfs import BFS
from algorithms.gbfs import GBFS
from algorithms.ucs import UCS
from algorithms.fringe import Fringe
from datetime import datetime, time, date
import streamlit as st

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
                # Calculate cost based on distance
            
                # Calculate travel time in minutes
                travel_time = self._calculate_travel_time(distance, from_id, to_id, prediction_model, datetime_str)
                
                    
                # Add edge to the graph
                if from_id not in graph.adjacency_list:
                    graph.adjacency_list[from_id] = []
                
                graph.adjacency_list[from_id].append((to_id, travel_time))
            
            return graph
        except Exception as e:
            print(f"Error creating search graph: {e}")
            return SearchGraph()  # Return an empty graph if there's an error
        
    def _calculate_travel_time(self, distance, from_id, to_id, prediction_model, datetime_str=None):
        """
        Calculate travel time based on distance and selected prediction model
        """
        # TODO: Replace with actual model predictions when ML models are integrated
        
        if prediction_model == "LSTM":
            # Simulate LSTM prediction with specific random pattern
            random_speed = random.uniform(35, 55)  # LSTM tends to predict medium speeds
        elif prediction_model == "GRU":
            # Simulate GRU prediction with different pattern
            random_speed = random.uniform(30, 50)  # GRU might predict slightly lower speeds
        elif prediction_model == "Custom":
            # Simulate custom model prediction
            random_speed = random.uniform(40, 60)  # Custom model might have different range
        else:
            # Default fallback
            random_speed = random.uniform(30, 60)
        
        # Store datetime_str for future use when models are integrated
            if datetime_str:
                # This will be used later to fetch appropriate data
                # For now, just log it or use it as a parameter
                pass

        # Calculate travel time in minutes
        travel_time = (distance / random_speed) * 60
        
        # Add intersection delay (30 seconds = 0.5 minutes)
        travel_time += 0.5
        
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