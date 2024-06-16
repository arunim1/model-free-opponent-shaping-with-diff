import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def sigmoid(z):
        return 1 / (1 + np.exp(-z)) 


    # Create a grid of x, y values
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)

    # Compute Z values for f(x,y)
    Z_f = sigmoid(X) * sigmoid(Y)

    # Compute Z values for g(x,y)
    Z_g = (sigmoid(X) + sigmoid(Y)) / 2

    # Plot g(x,y)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off') 

    ax.plot_surface(X, Y, Z_g, cmap='viridis', edgecolor='none', antialiased=False)´
    ax.set_title('3D plot of g(x,y) = (sigmoid(x) + sigmoid(y)) / 2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # without the mesh texture effect
    # ax.plot_surface(X, Y, Z_g, cmap='viridis', edgecolor='none')
    # without the grid and the axes
    # ax.axis('off')
    # plt.show()
    plt.show()

    

if __name__ == "__main__":
    main("mfos_ppo_IPD_NL_dumptest")
    