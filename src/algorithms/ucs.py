from src.algorithms.search_algorithm import SearchAlgorithm
import heapq


class UCS(SearchAlgorithm):
    

    def search(self, start, goals):
       
        # Use a counter to break ties in chronological order when costs are equal
        insertion_counter = 0

        # Priority queue to store (cost, insertion_order, node_id, path)       
        frontier = [(0, insertion_counter, start, [start])]
        heapq.heapify(frontier)
        insertion_counter += 1

        # Set to keep track of explored nodes to avoid revisiting
        explored = set()

        # Track number of nodes expanded
        nodes_expanded = 0

        while frontier:
            # Get the node with the lowest cost (primary criterion)
            # If costs are equal, the tie is broken by insertion order (FIFO)
            # If insertion order is also equal, the tie is broken by node ID (ascending)
            current_cost, _, current_node, current_path = heapq.heappop(frontier)

            # Skip if we've already explored this node
            if current_node in explored:
                continue

            # Mark node as explored
            explored.add(current_node)
            nodes_expanded += 1

            # Check if current node is a goal (NOTE 1: reach one of the destination nodes)
            if current_node in goals:
                return current_node, nodes_expanded, current_path

            # Get all neighbors and their costs, then sort by node ID to ensure
            # ascending order processing (NOTE 2)
            neighbors = self.graph.adjacency_list.get(current_node, [])

            # Process neighbors in ascending node ID order when adding to frontier
            for neighbor, edge_cost in sorted(neighbors, key=lambda x: x[0]):
                if neighbor not in explored:
                    # Compute new cost
                    new_cost = current_cost + edge_cost
                    new_path = current_path + [neighbor]

                    # Add to frontier with cost as primary sort key, insertion order as secondary,
                    # And node ID as tertiary for correct tie-breaking
                    heapq.heappush(frontier, (new_cost, insertion_counter, neighbor, new_path))
                    insertion_counter += 1  # Increment counter for stable tie-breaking

        # No path found
        return None, nodes_expanded, []
