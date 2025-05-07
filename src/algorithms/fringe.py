from src.algorithms.search_algorithm import SearchAlgorithm
import math
import heapq
from haversine import haversine


class Fringe(SearchAlgorithm):
    # Fringe Search combines A* and IDA* to minimize node expansions while finding optimal paths

    def __init__(self, graph):
        # Initialize with a graph
        super().__init__(graph)
        self.nodes = graph.node_coordinates
        
        # Create an adjacency list from the graph
        self.adjacency_list = {}
        for from_node, neighbors in graph.adjacency_list.items():
            if from_node not in self.adjacency_list:
                self.adjacency_list[from_node] = []
            
            for to_node, cost in neighbors:
                self.adjacency_list[from_node].append((to_node, cost))
        
    def get_min_heuristic(self, node, goals):
        # Get the minimum heuristic to any goal
        return min([self.graph.get_heuristic_time(node, goal) for goal in goals])

    # Execute Fringe Search from start to any goal
    def search(self, start, goals):
        # Convert goals to set for fast lookup
        goals = set(goals)

        # Check if the start node is already a goal
        if start in goals:
            return start, 1, [start]

        nodes_expanded = 0  # Count the number of nodes expanded

        # Priority queue for open nodes
        counter = 0
        open_set = []  # List for heapq operations

        # Initial f-limit based on the heuristic of the start node
        f_limit = self.get_min_heuristic(start, goals)

        # Entry format: (f, g, counter, node, path)
        heapq.heappush(open_set, (f_limit, 0, counter, start, [start]))
        counter += 1

        # List to store nodes exceeding the current f_limit
        later = []

        # Track the best g-value for each node
        best_g = {start: 0}

        while open_set or later:
            # If open_set is empty, update f_limit and restore nodes from later
            if not open_set:
                if not later:
                    # No path found
                    return None, nodes_expanded, []

                # Update f_limit to the minimum f-value in the later list
                f_limit = min(f for f, _, _, _, _ in later)

                # Move all nodes from later to open_set
                open_set = later
                heapq.heapify(open_set)
                later = []
                continue

            # Get the node with the lowest f-value
            f, g, _, node, path = heapq.heappop(open_set)
            nodes_expanded += 1  # Increment nodes expanded

            # Check if the current node is a goal
            if node in goals:
                return node, nodes_expanded, path  # Return goal, count, and path

            # If f exceeds f_limit, add to later list
            if f > f_limit:
                later.append((f, g, counter, node, path))
                counter += 1
                continue

            # Expand neighbors of the current node
            for neighbor, cost in sorted(self.adjacency_list.get(node, [])):
                # Calculate the new path cost
                new_g = g + cost

                # Skip if a better path to this neighbor already exists
                if neighbor in best_g and best_g[neighbor] <= new_g:
                    continue

                # Update the best path to this neighbor
                best_g[neighbor] = new_g

                # Calculate f-value (g + h)
                h = self.get_min_heuristic(neighbor, goals)
                new_f = new_g + h

                # Create the new path
                new_path = path + [neighbor]

                # Add to the appropriate list based on f_limit
                if new_f > f_limit:
                    later.append((new_f, new_g, counter, neighbor, new_path))
                else:
                    heapq.heappush(open_set, (new_f, new_g, counter, neighbor, new_path))

                counter += 1

        # No solution found
        return None, nodes_expanded, []