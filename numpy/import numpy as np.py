import numpy as np
import matplotlib.pyplot as plt

# Create a 5x5 matrix
matrix = np.array([
    [0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0],
    [0, 255, 0, 255, 0],
    [0, 255, 255, 255, 0],
    [0, 0, 0, 0, 0]
])

# Show as image
plt.imshow(matrix, cmap='gray')
plt.title("Matrix as Image")
plt.colorbar()
plt.show()