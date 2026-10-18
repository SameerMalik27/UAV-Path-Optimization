import numpy as np
import matplotlib.pyplot as plt


GRID_SIZE = 50

def create_environment(size=GRID_SIZE, hazard_probability=0.2):
    """
    Creates a 2D matrix where 0 = open space and 1 = hazard/obstacle.
    """
    
    grid = np.random.choice([0, 1], size=(size, size), p=[1-hazard_probability, hazard_probability])
    return grid

def visualize_environment(grid):
    """
    Plots the grid using matplotlib to see the safe zones and hazards.
    """
    plt.figure(figsize=(8, 8))
    
    plt.imshow(grid, cmap='Greys', interpolation='nearest')
    plt.title(f"UAV Disaster Zone Map ({GRID_SIZE}x{GRID_SIZE})")
    plt.colorbar(ticks=[0, 1], label="0: Clear Space, 1: Hazard Zone")
    plt.show()

if __name__ == "__main__":
    print("Generating UAV Environment...")
    uav_grid = create_environment()
    print("Environment generated successfully! Opening visual map...")
    visualize_environment(uav_grid)