import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/cbf_avoid_colli')
from utilities.cbf_funcs import *

# Load data from file .npz
# data_tb0 = np.load("cbf_avoid_colli/data_plotting/plot_python/cbf_exp_scene/tb_0_data_cbf.npz")
# data_tb1 = np.load("cbf_avoid_colli/data_plotting/plot_python/cbf_exp_scene/tb_1_data_cbf.npz")
data_tb0 = np.load("cbf_avoid_colli/data_plotting/plot_python/cbf_rg_scene/tb_0_data_cbf.npz")

# === Plot trajectory of 2 robots ===
plt.figure()
plt.plot(data_tb0['x'], data_tb0['y'], label='Robot 0 trajectory', linestyle='-', color='green')
plt.scatter(data_tb0['x'][0], data_tb0['y'][0], color='green', marker='o', label='Start R0')
plt.scatter(data_tb0['target'][0], data_tb0['target'][1], color='orange', marker='*', label='Target R0')

# plt.plot(data_tb1['x'], data_tb1['y'], label='Robot 1 trajectory', linestyle='-', color='blue')
# plt.scatter(data_tb1['x'][0], data_tb1['y'][0], color='blue', marker='o', label='Start R1')
# plt.scatter(data_tb1['target'][0], data_tb1['target'][1], color='red', marker='*', label='Target R1')

# # Plot the first obstacle
# vertices_x1 = [0, 0.2, 0.2, 0, 0]  # x-coordinates of the vertices (closed shape)
# vertices_y1 = [-0.3, -0.3, -0.5, -0.5, -0.3]  # y-coordinates of the vertices (closed shape)

# plt.plot(vertices_x1, vertices_y1, label='Obstacle 1', linestyle='--', color='purple')  # Outline of the shape
# plt.fill(vertices_x1, vertices_y1, color='purple', alpha=0.3)  # Fill the shape with transparency

# # Plot the second obstacle
# vertices_x2 = [0, 0.3, 0, 0]  # x-coordinates of the vertices (closed shape)
# vertices_y2 = [0.3, 0.2, 0.0, 0.3]  # y-coordinates of the vertices (closed shape)

# plt.plot(vertices_x2, vertices_y2, label='Obstacle 2', linestyle='--', color='orange')  # Outline of the shape
# plt.fill(vertices_x2, vertices_y2, color='orange', alpha=0.3)  # Fill the shape with transparency

# Plot the obstacles and their outer ellipses
obstacles = []
# obstacles.append(np.array([[0.9, -0.6],
#                             [0.9, -0.9],
#                             [0.6, -0.9],
#                             [0.6, -0.6]]))
# Obstacles in experiment scene
obstacles.append(np.array([[0.5, -0.24],
                            [0.5, -0.51],
                            [0.2, -0.51],
                            [0.2, -0.24]])) # scene 3
obstacles.append(np.array([[-0.07, 0.91],
                            [0.14, 0.74],
                            [-0.13, 0.63]]))

# Obstacles in RG scene
obstacles.append(np.array([[0.5, 0.5],
                            [1.2, 0.4],
                            [1., 1.3]]))
obstacles.append(np.array([[2.3, 1.],
                            [1.6, 0.9],
                            [1.5, 1.5],
                            [2.2, 2.0]]))
obstacles.append(np.array([[0.2, 1.3],
                            [0.5, 1.4],
                            [0.4, 1.7],
                            [0.3, 1.6]]))
obstacles.append(np.array([[1.0, 2.5], 
                            [1.5, 2.5], 
                            [2.0, 3.3], 
                            [0.5, 3.5]]))
obstacles.append(np.array([[3.0, 1.5], 
                            [2.7, 2.5], 
                            [3.2, 3.4], 
                            [3.7, 3.0],
                            [3.5, 2.0]]))
obstacles.append(np.array([[2.3, 0.0], 
                            [2.3, 0.5], 
                            [2.8, 0.5], 
                            [2.8, 0.0]]))

num_obs = len(obstacles)
list_P_ellipse = []
list_center_ellipse = []
# Find the smallest bouding ellipsoid of a polytope
for i in range(num_obs):
    # polytope_k_ver = self.obstacles[i].vertices
    polytope_k_ver = obstacles[i]
    P_k, c_k = ellipse_outer_polytope(polytope_k_ver)
    list_P_ellipse.append(P_k)
    list_center_ellipse.append(c_k)
# Plot obstacles and their bounding ellipses
for i, obstacle in enumerate(obstacles):
    # Plot obstacle vertices
    vertices = np.vstack((obstacle, obstacle[0]))  # Close the polygon
    plt.plot(vertices[:, 0], vertices[:, 1], 'b--', label=f'Obstacle {i+1}' if i == 0 else None)
    plt.fill(vertices[:, 0], vertices[:, 1], color='purple', alpha=0.3)

    # Plot bounding ellipse
    P = list_P_ellipse[i]
    c = list_center_ellipse[i].flatten()

    # Compute ellipse parameters
    eigenvalues, eigenvectors = np.linalg.eig(P)
    axis_lengths = 2 / np.sqrt(eigenvalues)  # Semi-axis lengths
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))  # Rotation angle

    # Create and add the ellipse
    ellipse = Ellipse(xy=c, width=axis_lengths[0], height=axis_lengths[1], angle=angle,
                      edgecolor='red', facecolor='none', lw=2, label='Bounding Ellipse' if i == 0 else None)
    plt.gca().add_patch(ellipse)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Robot Trajectories')
plt.legend()
plt.axis('equal')
plt.grid(True)

# === Plot linear velocity following the time ===
plt.figure()
plt.plot(data_tb0['time'], data_tb0['lin_v'], label='Linear velocity R0')
# plt.plot(data_tb1['time'], data_tb1['lin_v'], label='Linear velocity R1')
plt.xlabel('Time (s)')
plt.ylabel('Linear velocity (m/s)')
plt.title('Linear Velocity')
plt.legend()
plt.grid(True)

# === Plot angular velocity following the time ===
plt.figure()
plt.plot(data_tb0['time'], data_tb0['ang_v'], label='Angular velocity R0')
# plt.plot(data_tb1['time'], data_tb1['ang_v'], label='Angular velocity R1')
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Angular Velocity')
plt.legend()
plt.grid(True)

plt.show()
