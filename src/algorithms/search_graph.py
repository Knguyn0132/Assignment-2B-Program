from haversine import haversine

class SearchGraph:
    """
    A class to represent a graph for the Route Finding Problem.
    It loads nodes, edges, origin, and destinations from a structured text file.
    """

    def __init__(self):
        """Initializes an empty graph structure."""
        self.adjacency_list = {}  # Adjacency list: {node: [(neighbor, cost), (neightbor, cost),...]} useful for other search methods not dfs
        self.node_coordinates = {}  # Stores node positions: {node: (x, y)}
        self.origin = None  # The start node
        self.destinations = set()  # Set of goal nodes
        self.speed_limit = 60  # km/h
    

    def load_from_file(self, filename):
        """Reads a structured graph file and extracts nodes, edges, origin, and destinations."""
        try:
            with open(filename, 'r') as file:
                section = None  # Tracks which section we are reading

                for line in file:
                    line = line.strip()  # Remove whitespace

                    if not line or line.startswith("#"):  # Ignore empty lines and comments
                        continue

                    # Detect section headers and continue processing the current line
                    new_section = self.read_section(line)
                    if new_section:
                        section = new_section
                        continue  # Move to the next line

                    # Process valid lines within each section
                    if section == "nodes":
                        self.parse_nodes(line)
                    elif section == "edges":
                        self.parse_edges(line)
                    elif section == "origin":
                        self.parse_origin(line)
                    elif section == "destinations":
                        self.parse_destinations(line)

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            exit(1)

    def read_section(self, line):
        """Detects and updates the section being read in the file."""
        sections = ["Nodes:", "Edges:", "Origin:", "Destinations:"]
        if line in sections:
            return line[:-1].lower()  # Convert "Nodes:" → "nodes"
        return None  # Return None if the line is not a section header

    def parse_nodes(self, line):
        """Parses node coordinates and stores them in a dictionary."""
        try:
            if ":" not in line:
                return  # Skip invalid lines

            node_id, coords = line.split(":")
            node_id = int(node_id.strip())  # Convert node ID to integer

            coords = coords.strip().replace("(", "").replace(")", "")
            x, y = map(int, coords.split(","))  # Extract (x, y) as integers

            self.node_coordinates[node_id] = (x, y)  # Store node coordinates
        except ValueError:
            print(f"Warning: Skipping invalid node line: {line}")

    def parse_edges(self, line):
        """Parses edges and stores them in an adjacency list."""
        try:
            if ":" not in line or "(" not in line or ")" not in line:
                return  # Skip invalid lines

            edge_data, cost = line.split(":")
            cost = int(cost.strip())  # Convert cost to integer
            node_a, node_b = map(int, edge_data.strip("()").split(","))  # Extract edge nodes

            if node_a not in self.adjacency_list:
                self.adjacency_list[node_a] = []
            if node_b not in self.adjacency_list:
                self.adjacency_list[node_b] = []

            self.adjacency_list[node_a].append((node_b, cost))
        except ValueError:
            print(f"Warning: Skipping invalid edge line: {line}")

    def parse_origin(self, line):
        """Parses and stores the origin (start node)."""
        try:
            self.origin = int(line.strip())
        except ValueError:
            print(f"Warning: Invalid origin value: {line}")

    def parse_destinations(self, line):
        """Parses and stores the destination nodes."""
        try:
            self.destinations = set(map(int, line.split(";")))  # Convert to a set
        except ValueError:
            print(f"Warning: Invalid destinations format: {line}")

    def display(self):
        """Prints the graph details for debugging."""
        print("Graph (Adjacency List):", self.adjacency_list)
        print("Node Coordinates:", self.node_coordinates)
        print("Origin:", self.origin)
        print("Destinations:", self.destinations)
        
    def get_heuristic_time(self, node1, node2):
        """
        Calculate heuristic travel time between two nodes
        Returns time in minutes
        """
        if node1 not in self.node_coordinates or node2 not in self.node_coordinates:
            return float('inf')
        
        lat1, lon1 = self.node_coordinates[node1]
        lat2, lon2 = self.node_coordinates[node2]
        
        # Get distance in km
        distance = haversine((lat1, lon1), (lat2, lon2))
        
        # Convert to travel time in minutes
        # distance (km) / speed (km/h) * 60 (min/h)
        travel_time = (distance / self.speed_limit) * 60
        
        return travel_time