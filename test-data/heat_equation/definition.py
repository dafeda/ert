from collections import namedtuple

import numpy as np

# Number of grid-cells in x and y direction
nx = 10

# time steps
k_start = 0
k_end = 500

# Define initial condition, i.e., the initial temperature distribution.
# How you define initial conditions will effect the spread of results,
# i.e., how similar different realisations are.
u_init = np.zeros((k_end, nx, nx))
u_init[:, 5:7, 5:7] = 100

# Initialize 3D arrays (with depth 1 to simulate 2D)
u_3d = np.zeros((k_end, 1, nx, nx))
u_3d[0, 0] = u_init[0]
alpha_3d = np.ones((1, nx, nx))

# Resolution in the x-direction (nothing to worry about really)
dx = 1

Coordinate = namedtuple("Coordinate", ["x", "y"])

obs_coordinates = [
    Coordinate(5, 3),
    Coordinate(3, 5),
    Coordinate(5, 7),
    Coordinate(7, 5),
    Coordinate(2, 2),
    Coordinate(7, 2),
]

obs_times = np.linspace(10, k_end, 8, endpoint=False, dtype=int)
