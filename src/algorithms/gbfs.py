from src.algorithms.search_algorithm import SearchAlgorithm
import heapq


class GBFS(SearchAlgorithm):
    # Implements Greedy Best-First Search

    def __init__(self, graph):
        super().__init__(graph)  # Initialize the base class with the graph

    # Calculate the heuristic distance (Euclidean) between a node and the goal
    def heuristic(self, node, goal):
        
        if node not in self.graph.node_coordinates or goal not in self.graph.node_coordinates:
            return float('inf')  # Return infinity if coordinates are missing

        # Get coordinates of the current node and the goal
        x1, y1 = self.graph.node_coordinates[node]
        x2, y2 = self.graph.node_coordinates[goal]

        # Return the Euclidean distance
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    # Find the smallest heuristic distance to any goal
    def get_best_goal_heuristic(self, node, goals):
        return min(self.graph.get_heuristic_time(node, goal) for goal in goals)

    # Execute Greedy Best-First Search from the start node to any goal node
    def search(self, start, goals):
        # Counter for tie-breaking in the priority queue
        insertion_counter = 0

        # Priority queue: (heuristic_value, insertion_order, node, path)
        open_list = [(self.get_best_goal_heuristic(start, goals), insertion_counter, start, [start])]
        heapq.heapify(open_list)  # Ensure the list is a valid heap
        closed_set = set()  # Track visited nodes
        nodes_expanded = 0  # Count the number of nodes expanded

        while open_list:
            # Get the node with the smallest heuristic value
            h, insertion_order, current_node, path = heapq.heappop(open_list)

            if current_node in closed_set:  # Skip already visited nodes
                continue

            # Mark the current node as visited and increment the expanded node counter
            closed_set.add(current_node)
            nodes_expanded += 1

            if current_node in goals:  # Check if the goal is reached
                return current_node, nodes_expanded, path  # Return the goal, count, and path

            # Explore neighbors of the current node
            for neighbor, _ in sorted(self.graph.adjacency_list.get(current_node, [])):
                if neighbor not in closed_set:  # Only consider unvisited neighbors
                    insertion_counter += 1  # Increment counter for tie-breaking
                    h = self.get_best_goal_heuristic(neighbor, goals)  # Calculate heuristic for the neighbor
                    # Add the neighbor to the priority queue with updated path
                    heapq.heappush(open_list, (
                        h,  # Sort by heuristic value
                        insertion_counter,  # Tie-break by insertion order
                        neighbor,  # Neighbor node
                        path + [neighbor]  # Path to the neighbor
                    ))

        # Return None if no path to a goal is found
        return None, nodes_expanded, []