import cv2
import numpy as np


def count_islands(matrix):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    # Ensure it's a binary matrix
    matrix[matrix > 0] = 1
    
    matrix = matrix.astype(np.uint8)
    
    # Perform connected components analysis with 4-connectivity
    num_labels, _ = cv2.connectedComponents(matrix, connectivity=4)
    
    return num_labels - 1
    
    
if __name__ == '__main__':
    # Input arrays
    test_array1 = np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 1]
    ], dtype=np.uint8)
    
    test_array2 = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=np.uint8)
    
    test_array3 = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ], dtype=np.uint8)
    
    # Print results
    print("Matrix:")
    print(test_array1)
    print(f"Number of islands: {count_islands(test_array1)}")
    
    print("\nMatrix:")
    print(test_array2)
    print(f"Number of islands: {count_islands(test_array2)}")
    
    print("\nMatrix:")
    print(test_array3)
    print(f"Number of islands: {count_islands(test_array3)}")