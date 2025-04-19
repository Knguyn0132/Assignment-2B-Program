import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

def read_numpy_file(file_path):
    """
    Read and display the contents of a NumPy .npy file.
    
    Args:
        file_path (str): Path to the .npy file
    
    Returns:
        np.ndarray: The loaded NumPy array
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None
    
    if not file_path.endswith('.npy'):
        print(f"Warning: File '{file_path}' does not have .npy extension.")
    
    try:
        # Load the NumPy array from the file
        data = np.load(file_path)
        
        # Display basic information about the array
        print("\n" + "=" * 50)
        print(f"NUMPY FILE: {os.path.basename(file_path)}")
        print("=" * 50)
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Size: {data.size} elements")
        print(f"Memory usage: {data.nbytes / (1024*1024):.2f} MB")
        
        # Display statistics
        print("\nStatistics:")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Mean value: {np.mean(data)}")
        print(f"Standard deviation: {np.std(data)}")
        
        # Display a sample of the data
        print("\nData sample:")
        if data.ndim == 1:
            # For 1D arrays, show first and last few elements
            sample_size = min(5, len(data))
            print(f"First {sample_size} elements: {data[:sample_size]}")
            print(f"Last {sample_size} elements: {data[-sample_size:]}")
        elif data.ndim == 2:
            # For 2D arrays, show a small sample
            rows = min(3, data.shape[0])
            cols = min(5, data.shape[1])
            print(f"First {rows}x{cols} elements:")
            print(data[:rows, :cols])
        else:
            # For higher-dimensional arrays, show shape and first element
            print(f"First element: {data.flatten()[0]}")
            print("(Note: Array has more than 2 dimensions, showing limited sample)")
        
        return data
    
    except Exception as e:
        print(f"Error loading NumPy file: {e}")
        return None

def visualize_numpy_data(data, file_path, save_plot=False):
    """
    Create a visualization of the NumPy data.
    
    Args:
        data (np.ndarray): The NumPy array to visualize
        file_path (str): Path to the original .npy file (for naming the plot)
        save_plot (bool): Whether to save the plot to a file
    """
    if data is None:
        return
    
    try:
        plt.figure(figsize=(12, 6))
        
        if data.ndim == 1:
            # For 1D arrays, create a line plot
            plt.plot(data)
            plt.title(f'1D Array Visualization - {os.path.basename(file_path)}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.grid(True)
            
        elif data.ndim == 2:
            # For 2D arrays, create a heatmap
            plt.imshow(data, aspect='auto', cmap='viridis')
            plt.colorbar(label='Value')
            plt.title(f'2D Array Visualization - {os.path.basename(file_path)}')
            plt.xlabel('Column Index')
            plt.ylabel('Row Index')
            
        else:
            # For higher dimensions, flatten to 2D for visualization
            # Take the first 'slice' if it's 3D or higher
            if data.ndim == 3 and data.shape[0] <= 10:
                # If it's a small 3D array, show multiple subplots
                fig, axes = plt.subplots(1, min(data.shape[0], 5), figsize=(15, 3))
                for i in range(min(data.shape[0], 5)):
                    if min(data.shape[0], 5) == 1:
                        ax = axes
                    else:
                        ax = axes[i]
                    im = ax.imshow(data[i], aspect='auto', cmap='viridis')
                    ax.set_title(f'Slice {i}')
                    ax.set_xlabel('Column Index')
                    if i == 0:
                        ax.set_ylabel('Row Index')
                fig.colorbar(im, ax=axes, label='Value')
                plt.suptitle(f'3D Array Visualization - {os.path.basename(file_path)}')
            else:
                # Otherwise, just flatten to 2D
                reshaped = data.reshape(data.shape[0], -1)
                plt.imshow(reshaped, aspect='auto', cmap='viridis')
                plt.colorbar(label='Value')
                plt.title(f'Higher-Dimensional Array (Flattened) - {os.path.basename(file_path)}')
                plt.xlabel('Flattened Column Index')
                plt.ylabel('First Dimension Index')
        
        plt.tight_layout()
        
        if save_plot:
            # Save the plot with the same name as the .npy file but with .png extension
            plot_path = os.path.splitext(file_path)[0] + '_visualization.png'
            plt.savefig(plot_path)
            print(f"\nVisualization saved to: {plot_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error visualizing data: {e}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Read and display NumPy (.npy) files')
    parser.add_argument('file_path', type=str, help='Path to the .npy file')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize the data')
    parser.add_argument('--save', '-s', action='store_true', help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Read the NumPy file
    data = read_numpy_file(args.file_path)
    
    # Visualize if requested
    if args.visualize and data is not None:
        visualize_numpy_data(data, args.file_path, args.save)

if __name__ == "__main__":
    main()
