class SearchAlgorithm:
    """
    Parent class for all search algorithms.
    """

    def __init__(self, graph):
        """
        Initializes the search algorithm with a graph.
        :param graph: The Graph object containing nodes and edges.
        """
        self.graph = graph  # Store the graph for searching

    def search(self, start, goals):
        """
        Abstract method that must be implemented by subclasses.
        Each search algorithm will define its own search logic.
        :param start: The starting node.
        :param goals: A set of goal nodes.
        :return: (goal_node, num_nodes_expanded, path_to_goal)
        """
        raise NotImplementedError("Search method must be implemented in subclasses")
