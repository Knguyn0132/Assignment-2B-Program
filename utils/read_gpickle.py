import pickle
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_gpickle.py <path_to_gpickle_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)

        print("\nSuccessfully loaded the graph!")
        print("\n--- Nodes ---")
        print(graph.nodes(data=True))
        print("\n--- Edges ---")
        print(graph.edges(data=True))

    except Exception as e:
        print(f"Failed to read the file: {e}")

if __name__ == "__main__":
    main()
