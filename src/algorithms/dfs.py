from src.algorithms.search_algorithm import SearchAlgorithm

class DFS(SearchAlgorithm):

    def __init__(self, graph):
        super().__init__(graph)  

    def search(self, start, goals):

        stack = [(start, [start])]
        visited = set()        
        nodes_expanded = 0

        # Loop while there are still nodes in the stack
        while stack:
            # Pop the last-added node from the stack (DFS follows LIFO order)
            node, path = stack.pop()

            # Skip this node if it has already been visited
            if node in visited:
                continue

            # Mark the node as visited and increase the expanded nodes counter
            visited.add(node)
            nodes_expanded += 1

            # Check if we have reached a goal node
            if node in goals:
                return node, nodes_expanded, path  # Return the goal node, count, and path

            # Get all neighboring nodes of the current node
            neighbors = self.graph.adjacency_list.get(node, [])

            # Sort neighbors in reverse order (ensures correct DFS order)
            sorted_neighbors = sorted(neighbors, reverse=True)

            # Add neighbors to the stack (DFS will explore the last-added first)
            for neighbor, _ in sorted_neighbors: # extract the value ignore the cost
                new_path = path + [neighbor]  # Create a new path that includes the neighbor
                stack.append((neighbor, new_path))


        return None, nodes_expanded, []
