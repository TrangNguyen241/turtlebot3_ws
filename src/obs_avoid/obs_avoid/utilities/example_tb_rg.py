from pypoman import compute_polytope_halfspaces
import numpy as np
import matplotlib.pyplot as plt
import gdspy
from poly_decomp import poly_decomp as pd
from pypoman import compute_polytope_vertices
from matplotlib.patches import Circle, Wedge, Polygon
import polytopemap_functions as wsft
import polytope
from scipy.sparse.csgraph import shortest_path
from numpy import linalg as LA
from shapely.geometry import LineString
from shapely.geometry import Polygon as PG
import copy
from scipy.spatial import ConvexHull
import rg_funcs as rgf
from scipy.io import savemat
import os


def plot_poly_map(ws, ax, col):
    for i in range(len(ws)):
        x_de = [ws[i][k][0] for k in range(len(ws[i]))]
        x_de.append(ws[i][0][0])
        y_de = [ws[i][k][1] for k in range(len(ws[i]))]
        y_de.append(ws[i][0][1])
        ax.plot(x_de, y_de, color=col)
    return 0








#### MAP 1 ####
# ws_limit = [[0, 0], [2, 0], [2, 2], [0, 2]]
# obstacle1 = [[0.5, 0.5], [1.2, 0.4], [1., 0.7], [1., 1.3]]
# obstacle2 = [[1.8, 1.], [1.6, 0.9], [1.5, 1.5], [1.8, 1.7]]
# obstacle3 = [[0.2, 1.3], [0.5, 1.4], [0.4, 1.7], [0.3, 1.6]]

##### MAP 2 #####
ws_limit = [[0, 0], [4, 0], [4, 4], [0, 4]]
obstacle1 = [[0.5, 0.5], [1.2, 0.4], [1., 1.3]]
obstacle2 = [[2.3, 1.], [1.6, 0.9], [1.5, 1.5], [2.2, 2.0]]
obstacle3 = [[0.2, 1.3], [0.5, 1.4], [0.4, 1.7], [0.3, 1.6]]
# obstacle3 = [[0.5, 1.4], [0.4, 1.7], [0.6, 1.8], [0.8, 1.6]]
obstacle4 = [[1.0, 2.5], [1.5, 2.5], [2.0, 3.3], [0.5, 3.5]]
obstacle5 = [[3.0, 1.5], [2.7, 2.5], [3.2, 3.4], [3.7, 3.0], [3.5, 2.0]]
obstacle6 = [[2.3, 0.0], [2.3, 0.5], [2.8, 0.5], [2.8, 0.0]] # add for check error


ws_poly = gdspy.Polygon(ws_limit)
hole1 = gdspy.Polygon(obstacle1)
hole2 = gdspy.Polygon(obstacle2)
hole3 = gdspy.Polygon(obstacle3)
hole4 = gdspy.Polygon(obstacle4)
hole5 = gdspy.Polygon(obstacle5)
hole6 = gdspy.Polygon(obstacle6) 

safety_offset = 0.0
hole1_large = gdspy.offset(hole1, safety_offset)
hole2_large = gdspy.offset(hole2, safety_offset)
hole3_large = gdspy.offset(hole3, safety_offset)
hole4_large = gdspy.offset(hole4, safety_offset)
hole5_large = gdspy.offset(hole5, safety_offset)
hole6_large = gdspy.offset(hole6, safety_offset)

# subtraction 
# Find the free space
# poly_with_hole = gdspy.boolean(ws_poly, [hole1_large, hole2_large, hole3_large], "not")
poly_with_hole = gdspy.boolean(ws_poly, [hole1_large, hole2_large, hole4_large, hole3_large, hole5_large, hole6_large], "not")

# decomposition 
# Partition the free space into polytopes
ws = pd.polygonQuickDecomp(poly_with_hole.polygons[0])  # the workspace

#---- Convert the list of polytopes in "ws" into a format MATLAB can readd
ws_array = [np.stack(polygon) for polygon in ws]  # Each polytope is converted to a NumPy array

ws_array_obj = np.empty(len(ws_array), dtype=object)
ws_array_obj[:] = ws_array
# Save the data to a .mat file
# savemat("/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/utilities/example_tb_rg.py/ws_polytopes.mat", {"ws": ws_array_obj})
base_dir_mat = "/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/utilities"
filename = f"ws_polytopes.mat"
savemat(os.path.join(base_dir_mat, filename), {"ws": ws_array_obj})

print("Saved ws data to ws_data.mat")
n_polyp = len(ws)
center_list = []

# init = [1, 1.75]  # starting point => Map 1
init = [0.26, 3.0]  # starting point => Map 2
# init = [1.5, 0.5]  
# goal = [1.75, 0.5]  # goal => chay dep voi map 1
# goal = [3.8, 1.6]  # goal => chay dep voi map 2
goal = [2.3, 0.8] # dep cho map 2
# goal = [3.8, 1.9]
# Find center of each polytope
for j in range(n_polyp):
    center_list.append(wsft.center(ws[j]).A1)


N = np.empty((0, n_polyp), float)
N_temp = np.array([])
for i in range(n_polyp):
    N_temp = []
    for j in range(n_polyp):
        if wsft.check_consecutive_polytopes(ws[i], ws[j]) and (i != j):
            temp = [center_list[i], center_list[j]]
            N_temp.append(LA.norm(temp))
        
        else:
            N_temp.append(0)
    N = np.append(N, [N_temp], axis=0)
# Dijkstra’s algorithm to find the shortest path
D, Pr = shortest_path(N, directed=False, method='D', return_predecessors=True)

init_goal_idx = [wsft.find_polyp(ws, init), wsft.find_polyp(ws, goal)]

sequence = wsft.get_path(Pr, init_goal_idx[0], init_goal_idx[1])

print("Find path to go from ", init, "to ", goal)

print("The index of the sequence of polytopes are: ", sequence)

" This part returns the transitioning polytope formed by two neigboring polytopes"
transZone_list = []
transPolytope_list = []

# Find the transitioning polytope formed by two neigboring polytopes
for i in range(len(sequence)-1):
    IndexPolytopeToCheck = [sequence[i], sequence[i+1]]
    (A,B) = wsft.transition_zone(ws[IndexPolytopeToCheck[0]], ws[IndexPolytopeToCheck[1]])
    transZone_k = polytope.Polytope(np.array(A), np.array(B))
    transZone_list.append(transZone_k)
    # convert zone to polytope
    # transZone_k_vertices = wsft.vertices_from_halfspace(np.array(A), np.array(B))
    transZone_k_vertices = wsft.vertices_from_halfspace(transZone_k.A, transZone_k.b)
    trans_k_polytope = polytope.qhull(np.array(transZone_k_vertices))
    transPolytope_list.append(trans_k_polytope)


# Method 2: Use polytope.qhull(...) function => don gian, chinh xac
poly_merge_list = []
for i in range(len(sequence)):
    poly_k = ws[sequence[i]]
    poly_k_vertices = np.array(poly_k)
    if i==0:
        trans_k = transPolytope_list[0]
        trans_k_vertices = trans_k.vertices
    elif i==len(sequence)-1:
        trans_k = transPolytope_list[-1]
        trans_k_vertices = trans_k.vertices
    else:
        trans_k1 = transPolytope_list[i-1]
        trans_k2 = transPolytope_list[i]
        trans_k_vertices = np.vstack((trans_k1.vertices, trans_k2.vertices))
    poly_k_trans_vertices = np.vstack((poly_k_vertices, trans_k_vertices))
    poly_k_trans_qhull = polytope.qhull(poly_k_trans_vertices)
    poly_merge_list.append(poly_k_trans_qhull)

polytope_merge_plot = []
# Convert the vertices to polytopes
for i in range(len(sequence)):
    poly_k_merge = poly_merge_list[i]
    # for poltting
    poly_merge_vertices_sorted = wsft.sort_vertices_clockwise(poly_k_merge.vertices)
    polytope_merge_plot.append(poly_merge_vertices_sorted)

#+++++++++++++++++++++++ REFERENCE GOVERNOR FOR ROBOT TO AVOID OBSTACLE +++++++++++++++++++++++
# System parameters
A = np.zeros((2,2))
B = np.eye(2)
Ts = 0.1
nx = A.shape[1]
nu = A.shape[1]
C = np.eye(2)
D = np.zeros((2, 2))
nr = C.shape[0]
Mc = np.block([[A, B], [C, D]])
Mc_inv = np.linalg.inv(Mc)
Kxr = Mc_inv[:nx, -nu:]
Kur = Mc_inv[nx:, -nu:]
beta = 0.025
# Q = np.array([[1.0, 0], [0, 1.0]])
Q = np.array([[10.0, 0], [0, 10.0]])
# R = np.array([[-5.125, 0], [0, -5.125]])
R = np.array([[-5.125, 0], [0, -5.125]])
P = np.linalg.inv(Q)
K = np.dot(R, P)

# Imput constraint
b= 0.1
max_angular_vel = 2.84
max_linear_vel = 0.22
ru = min(b*max_angular_vel, max_linear_vel)

# Constraints of input
ptsU = []
for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
    ptsU.append([ru * np.cos(tta), ru * np.sin(tta)])
ptsU = np.array(ptsU)
U_input = polytope.qhull(ptsU)

# Constraints of state (moving space for robot)
corridor = copy.deepcopy(poly_merge_list)

# Temporary ref
# Get the center of transition zone as a temporary reference
tmp_ref = []
for i in range(len(transPolytope_list)):
    vertices_trans = transPolytope_list[i].vertices
    center_trans = np.mean(vertices_trans, axis=0)
    tmp_ref.append(center_trans)
desired_target = copy.deepcopy(goal)
desired_target = np.array(desired_target)
tmp_ref.append(desired_target)
tsim = 200
Nsim = round(tsim/Ts)
pose = np.zeros((3,Nsim))
init_point = copy.deepcopy(init)
init_pose_np = np.array(init_point).reshape(-1,1)
init_pose = np.array([[init_pose_np[0,0]],[init_pose_np[1,0]],[np.pi/2]]) 
pose[:,[0]] = init_pose
z_init = np.array([
                [init_pose[0,0] + b * np.cos(init_pose[2,0])],
                [init_pose[1,0] + b * np.sin(init_pose[2,0])]])
z_list = np.zeros((2, Nsim-1))
ureal = np.zeros((nu, Nsim-1))
kappa = 350.0
eta = 0.075
target_idx = 0
AllConstraint = {
        "A": np.vstack([
            np.hstack([U_input.A @ K, -U_input.A @ K @ Kxr]),
            np.hstack([corridor[target_idx].A, np.zeros((corridor[target_idx].A.shape[0], nr))])
        ]),
        "b": np.concatenate((U_input.b, corridor[target_idx].b))
    }
tmp_ref_k = tmp_ref[target_idx]
filteredRef = z_init
time_plot = []

#*********** CONTROL LOOP *******************

for i in range(Nsim-1):
    z = np.array([[pose[0, i] + b*np.cos(pose[2, i])],
                  [pose[1, i] + b*np.sin(pose[2, i])]])
    z_list[:, [i]] = z
    print("index===========: ", i)
    if target_idx < len(tmp_ref)-1:
        if rgf.check_corridor(corridor[target_idx], z) and rgf.check_corridor(corridor[target_idx+1],z):
        # if rgf.check_corridor(transPolytope_list[target_idx], z):
        # if np.all(corridor[target_idx].A @ z <= corridor[target_idx].b.reshape(-1,1)) and np.all(corridor[target_idx+1].A @ z <= corridor[target_idx+1].b.reshape(-1,1)):
            target_idx = target_idx +1
            AllConstraint["A"] = np.vstack([
                        np.hstack([U_input.A @ K, - U_input.A @ K @ Kxr]),
                        np.hstack([corridor[target_idx].A,
                                   np.zeros((corridor[target_idx].A.shape[0], nr))])
                    ])
            AllConstraint["b"] = np.vstack((
                        U_input.b.reshape(-1, 1), corridor[target_idx].b.reshape(-1,1)
                    ))
            tmp_ref_k = tmp_ref[target_idx]
            print(f"+++++++target_idx++++++++++++: {target_idx}")
            print(f"tmp_ref_k when in if loop: {tmp_ref_k}")
            print("************************Change set-point!**************************************************")
    # Compute feedback and virtual controls
    xfb = z
    xv = Kxr @ filteredRef
    all_constraint_A = AllConstraint["A"]
    all_constraint_b = AllConstraint["b"]
   
    # Dynamic safety margin
    Gamma = rgf.RGThreshold(AllConstraint["A"], AllConstraint["b"], P, xv, filteredRef)
    Delta = kappa * (Gamma - (xfb - xv).T @ P @ (xfb - xv))
    Delta_float = Delta.item()
  

 
    # Navigation field
    tmp_ref_k = tmp_ref_k.reshape(2,1)
    rho = (tmp_ref_k - filteredRef) / max(np.linalg.norm(tmp_ref_k - filteredRef), eta)
   
    # Integrate and update the filtered reference
    filteredRef = filteredRef + Ts * Delta_float * rho


    xv = Kxr @ filteredRef

    # Compute the virtual control input and transform to real inputs
    u_vir = K @ (xfb - xv)
    matrix_transform = np.array([[np.cos(pose[2, i]), np.sin(pose[2, i])],
                                [-np.sin(pose[2, i])/b, np.cos(pose[2, i])/b]])
    ureal[:, [i]] = matrix_transform @ u_vir

    x_k = pose[0, i]
    y_k = pose[1, i]
    theta_k = pose[2, i]
    lin_vel = ureal[0, i]
    ang_vel = ureal[1, i]
    pose[:, [i+1]] = np.array([[x_k + Ts*lin_vel*np.cos(theta_k)],
                               [y_k + Ts*lin_vel*np.sin(theta_k)],
                               [theta_k + Ts*ang_vel]])
    
    time_k = i*Ts
    time_plot.append(time_k)
    # print("Current Corridor Check:", corridor[target_idx].A @ z, corridor[target_idx].b)
    # print("Next Corridor Check:", corridor[target_idx+1].A @ z, corridor[target_idx+1].b)
    # print("Shape of corridor[target_idx].A @ z:", (corridor[target_idx].A @ z).shape)
    # print("Shape of corridor[target_idx].b:", corridor[target_idx].b.shape)



# Debug
# Assuming ureal is already defined
max_values = np.max(ureal, axis=1)
print("Maximum values of each row:", max_values)
# Plot the result
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
plot_poly_map(ws, ax, 'black')
plt.scatter(z_init[0,0], z_init[1,0], color='red', s=100, marker='*')
plt.scatter(goal[0], goal[1], color='blue', s=100, marker='*')
# Plot center of each polytope in the path
# Convert center_list to numpy array
center_list_np = np.array(center_list)
# Plot center of polytope in shortest path
selected_center = center_list_np[sequence]
# plt.scatter(selected_center[:, 0], selected_center[:, 1], color='black', s=30)
# Plot center of transition 
tmp_ref_plot = np.array(tmp_ref)
plt.scatter(tmp_ref_plot[:-1, 0], tmp_ref_plot[:-1, 1], color='black', s=30)

"The transition zone"
# for i in range(len(transZone_list)):
#     transZone_list[i].plot(ax, color = 'g')

"The polytopes in check"

# p = Polygon(ws[IndexPolytopeToCheck[0]], facecolor='b', alpha=0.1)
# plt.gca().add_patch(p)
# p = Polygon(ws[IndexPolytopeToCheck[1]], facecolor='b', alpha=0.1)
# plt.gca().add_patch(p)

for i in range(len(polytope_merge_plot)):
    p = Polygon(polytope_merge_plot[i], facecolor='r', alpha = 0.2)
    plt.gca().add_patch(p)

"The map"
plot_poly_map([ws[i] for i in sequence], ax, 'red')
"The obstacles"
p = Polygon(hole1_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
p = Polygon(hole2_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
p = Polygon(hole3_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
p = Polygon(hole4_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
p = Polygon(hole5_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
p = Polygon(hole6_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)

## Plot trajectory of x
x_robot = z_list[0, :]
y_robot = z_list[1, :]
plt.plot(x_robot, y_robot, label='Trajectory')

fig2 = plt.figure()
plt.plot()
time_plot = np.array(time_plot)
plt.plot(time_plot, ureal[0, :], 'k', label='Trajectory')
# plt.gca().set_aspect('equal', adjustable='box')


plt.show()
