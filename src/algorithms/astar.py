import heapq
import math
from src.algorithms.search_algorithm import SearchAlgorithm
from haversine import haversine
class AStar(SearchAlgorithm):
    def __init__(self, graph):
        # Initialize the A* search with the graph
        super().__init__(graph)
        self.nodes = graph.node_coordinates
        
        # Create an adjacency list from the graph
        self.adjacency_list = {}
        for from_node, neighbors in graph.adjacency_list.items():
            if from_node not in self.adjacency_list:
                self.adjacency_list[from_node] = []
            
            for to_node, cost in neighbors:
                self.adjacency_list[from_node].append((to_node, cost))
    
    # Execute A* Search from the start node to any goal node
    def search(self, start, goals):
        # Convert goals to a set for fast lookup
        goals = set(goals)
        
        # Track the cost from the start to each node
        g_scores = {start: 0}
        
        # Unique entry ID for priority queue to avoid conflicts
        entry_id = 0
        
        # Priority queue: (f_score, entry_id, node, path)
        open_list = []
        
        # Calculate the initial heuristic (min distance to any goal)
        initial_h = min([self.graph.get_heuristic_time(start, goal) for goal in goals])
        heapq.heappush(open_list, (initial_h, entry_id, start, [start]))
        entry_id += 1
        
        # Track visited nodes
        visited = set()
        nodes_expanded = 0
        
        while open_list:
            # Pop the node with the lowest f_score
            _, _, current, path = heapq.heappop(open_list)
            
            if current in visited:  # Skip already visited nodes
                continue
            
            # Mark the current node as visited and increment nodes expanded
            visited.add(current)
            nodes_expanded += 1
            
            if current in goals:  # Check if the current node is a goal
                return current, nodes_expanded, path  # Return goal, count, and path
            
            # Explore neighbors of the current node
            neighbors = sorted(self.adjacency_list.get(current, []))
            for neighbor, cost in neighbors:
                # Calculate the new g_score
                new_g_score = g_scores[current] + cost
                
                # Only consider this path if it's better than any previous one
                if neighbor not in g_scores or new_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = new_g_score  # Update g_score
                    
                    # Calculate h_score as the minimum distance to any goal
                    h_score = min([self.graph.get_heuristic_time(neighbor, goal) for goal in goals])
                    
                    # Calculate f_score (g_score + h_score)
                    f_score = new_g_score + h_score
                  
                    # Add the neighbor to the priority queue with updated path
                    new_path = path + [neighbor]
                    heapq.heappush(open_list, (f_score, entry_id, neighbor, new_path))
                    entry_id += 1
        
        # Return None if no path to a goal is found
        return None, nodes_expanded, []