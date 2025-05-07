from src.algorithms.search_algorithm import SearchAlgorithm
from collections import deque

class BFS(SearchAlgorithm):

    def __init__(self, graph):
        super().__init__(graph)  # Initialize the base class with the graph

    def search(self, start, goals):
        # Initialize the queue with the start node and its path
        queue = deque([(start, [start])])
        visited = set()  # Track visited nodes
        nodes_expanded = 0  # Count the number of nodes expanded

        while queue:
            # Get the next node and its path from the queue
            node, path = queue.popleft()

            if node in visited:  # Skip already visited nodes
                continue
            visited.add(node)  # Mark the node as visited
            nodes_expanded += 1  # Increment the expanded nodes count

            if node in goals:  # Check if the goal is reached
                return node, nodes_expanded, path  # Return the goal, count, and path

            # Add neighbors to the queue with updated paths
            for neighbor, _ in sorted(self.graph.adjacency_list.get(node, [])):
                queue.append((neighbor, path + [neighbor]))

        # Return None if no solution is found
        return None, nodes_expanded, []