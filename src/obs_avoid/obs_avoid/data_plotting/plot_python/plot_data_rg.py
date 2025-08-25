import matplotlib.pyplot as plt
import numpy as np
import sys


# Load data from file .npz
data_tb = np.load("/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/data_plotting/plot_python/tb_data_rg.npz")

# === Plot trajectory of 2 robots ===
plt.figure()
plt.plot(data_tb['x'], data_tb['y'], label='Robot trajectory', linestyle='-', color='green')
plt.scatter(data_tb['x'][0], data_tb['y'][0], color='green', marker='o', label='Start')
plt.scatter(data_tb['target'][0], data_tb['target'][1], color='orange', marker='*', label='Target')


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
# Plot obstacles and their bounding ellipses
for i, obstacle in enumerate(obstacles):
    # Plot obstacle vertices
    vertices = np.vstack((obstacle, obstacle[0]))  # Close the polygon
    plt.plot(vertices[:, 0], vertices[:, 1], 'b--', label=f'Obstacle {i+1}' if i == 0 else None)
    plt.fill(vertices[:, 0], vertices[:, 1], color='purple', alpha=0.3)

plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Robot Trajectories')
plt.legend()
plt.axis('equal')
plt.grid(True)

# === Plot linear velocity following the time ===
plt.figure()
plt.plot(data_tb['time'], data_tb['lin_v'], label='Linear velocity')
# plt.plot(data_tb1['time'], data_tb1['lin_v'], label='Linear velocity R1')
plt.xlabel('Time (s)')
plt.ylabel('Linear velocity (m/s)')
plt.title('Linear Velocity')
plt.legend()
plt.grid(True)

# === Plot angular velocity following the time ===
plt.figure()
plt.plot(data_tb['time'], data_tb['ang_v'], label='Angular velocity')
# plt.plot(data_tb1['time'], data_tb1['ang_v'], label='Angular velocity R1')
plt.xlabel('Time (s)')
plt.ylabel('Angular velocity (rad/s)')
plt.title('Angular Velocity')
plt.legend()
plt.grid(True)

plt.show()
