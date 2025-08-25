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

def check_consecutive_polytopes(p1, p2):
    check = False
    s = 0
    check_list = gdspy.inside(p1, gdspy.Polygon(p2))
    for i in range(len(check_list)):
        if check_list[i]:
            s = s + 1
    if s == 2:
        check = True
    return check


def find_common_vertices(p1, p2):
    ver_list = np.empty((0, 2), float)
    check_list = gdspy.inside(p1, gdspy.Polygon(p2))
    for i in range(len(check_list)):
        if check_list[i]:
            ver_list = np.append(ver_list, np.array([p1[i]]), axis=0)
    return ver_list


def plot_poly_map(ws, ax, col):
    for i in range(len(ws)):
        x_de = [ws[i][k][0] for k in range(len(ws[i]))]
        x_de.append(ws[i][0][0])
        y_de = [ws[i][k][1] for k in range(len(ws[i]))]
        y_de.append(ws[i][0][1])
        ax.plot(x_de, y_de, color=col)
    return 0


def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


def find_polyp(workspace, points):
    polyp_index = -1
    for i in range(len(workspace)):
        if (gdspy.inside([points], gdspy.Polygon(ws[i])))[0]:
            polyp_index = i
    return polyp_index


def center(polyp):
    xc = np.matrix(polyp)
    return xc.mean(0)

# Check intersect with obstacle
def is_intersecting_obstacle(start, end, obstacle_polygon):
    path_line = LineString([start, end])
    for obs in obstacle_polygon:
        if path_line.intersects(obs):
            return True
    return False


ws_limit = [[0, 0], [2, 0], [2, 2], [0, 2]]

obstacle1 = [[0.5, 0.5], [1.2, 0.4], [1., 0.7], [1., 1.3]]
obstacle2 = [[1.8, 1.], [1.6, 0.9], [1.5, 1.5], [1.8, 1.7]]
obstacle3 = [[0.2, 1.3], [0.5, 1.4], [0.4, 1.7], [0.3, 1.6]]

ws_poly = gdspy.Polygon(ws_limit)
hole1 = gdspy.Polygon(obstacle1)
hole2 = gdspy.Polygon(obstacle2)
hole3 = gdspy.Polygon(obstacle3)

# obs_1_pg = PG(obstacle1)
# obs_2_pg  = PG(obstacle2)
# obs_3_pg  = PG(obstacle3)
# obs_holes = [obs_1_pg, obs_2_pg, obs_3_pg]

safety_offset = 0.0
hole1_large = gdspy.offset(hole1, safety_offset)
hole2_large = gdspy.offset(hole2, safety_offset)
hole3_large = gdspy.offset(hole3, safety_offset)

# subtraction 
# Find the free space
poly_with_hole = gdspy.boolean(ws_poly, [hole1_large, hole2_large, hole3_large], "not")
# decomposition 
# Partition the free space into polytopes
ws = pd.polygonQuickDecomp(poly_with_hole.polygons[0])  # the workspace

n_polyp = len(ws)
center_list = []

init = [1, 1.75]  # starting point
goal = [1.75, 0.5]  # goal
# goal = [0.1, 0.5]  # goal
# goal = [0.75, 0.25]  # goal
# goal = [1.3, 0.8]  # goal

# Find center of each polytope
for j in range(n_polyp):
    center_list.append(center(ws[j]).A1)

print("aaaaaaaaaaaaaaaaa",center_list[0])
N = np.empty((0, n_polyp), float)
N_temp = np.array([])
for i in range(n_polyp):
    N_temp = []
    for j in range(n_polyp):
        if check_consecutive_polytopes(ws[i], ws[j]) and (i != j):
            temp = [center_list[i], center_list[j]]
            N_temp.append(LA.norm(temp))
        
        else:
            N_temp.append(0)
    N = np.append(N, [N_temp], axis=0)
# Dijkstra’s algorithm to find the shortest path
D, Pr = shortest_path(N, directed=False, method='D', return_predecessors=True)

init_goal_idx = [find_polyp(ws, init), find_polyp(ws, goal)]

sequence = get_path(Pr, init_goal_idx[0], init_goal_idx[1])

print("Find path to go from ", init, "to ", goal)

print("The index of the sequence of polytopes are: ", sequence)

" This part returns the transitioning polytope formed by two neigboring polytopes"
transZone_list = []
# for i in range(len(sequence)-1):
#     IndexPolytopeToCheck = [sequence[i], sequence[i+1]]
#     (A,B) = wsft.transition_zone(ws[IndexPolytopeToCheck[0]], ws[IndexPolytopeToCheck[1]])
#     transZone_k = polytope.Polytope(np.array(A), np.array(B))
#     transZone_list.append(transZone_k)
# poly1 = polytope.qhull(np.array(ws[sequence[0]])) 
# transZone_0_vertices = wsft.vertices_from_halfspace(transZone_list[0].A, transZone_list[0].b)
# trans_0 = polytope.qhull(np.array(transZone_0_vertices))
# transZone_1_vertices = wsft.vertices_from_halfspace(transZone_list[1].A, transZone_list[1].b)
# trans_1 = polytope.qhull(np.array(transZone_1_vertices))


def sort_vertices_clockwise(vertices):
    # Bước 1: Tính tâm của đa giác
    center = np.mean(vertices, axis=0)

    # Bước 2: Tính góc của từng điểm so với tâm
    angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])

    # Bước 3: Sắp xếp theo góc giảm dần (chiều kim đồng hồ)
    sort_order = np.argsort(-angles)  # dấu '-' để sort giảm dần
    sorted_vertices = vertices[sort_order]

    return sorted_vertices

transPolytope_list = []
poly_merge_k_vertices = []


ws_sc = {}
# for i in range(len(sequence)):
#     ws_sc[sequence[i]] = ws[sequence[i]]

for i in range(len(sequence)):
    ws_sc[sequence[i]] = copy.deepcopy(ws[sequence[i]])

# ws_sc = [None] * (max(sequence) + 1)
# for i in range(len(sequence)):
#     ws_sc[sequence[i]] = ws[sequence[i]]

# ws_sc = {
#     sequence[0]: ws[sequence[0]],
#     sequence[1]: ws[sequence[1]],
#     sequence[2]: ws[sequence[2]]
# }

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

# Merge transition part with original polytope
# Method 1 by Chang :))) ==> Khong work cho may truong hop phuc tap, nhung coi nhu co tu duy di :))
# for i in range(len(sequence)-1):
#     IndexPolytopeToCheck = [sequence[i], sequence[i+1]]
#     trans_i_polytope = transPolytope_list[i]

#     for j in range(len(IndexPolytopeToCheck)):
#         poly_k = polytope.qhull(np.array(ws[IndexPolytopeToCheck[j]]))
#         valid_vertices_list = []
#         for v1 in trans_i_polytope.vertices:
#             is_common = any(np.allclose(v1, v2, atol=1e-4) for v2 in poly_k.vertices)
#             Ax = poly_k.A @ v1.reshape(-1, 1)
#             is_inside = np.all(Ax.flatten() <= poly_k.b)
#             if not(is_common or is_inside):
#                 valid_vertice = v1
#                 # valid_vertices_list.append(v1)
#         # valid_vertice = np.array(valid_vertice)
#         # ws_2[IndexPolytopeToCheck[j]] = ws_2[IndexPolytopeToCheck[j]] + valid_vertices_list
#         # ws_sc[IndexPolytopeToCheck[j]].append(np.array(valid_vertices_list))

#         ws_sc[IndexPolytopeToCheck[j]].append(np.array(valid_vertice))

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
# Use convexhull to merge polytope with transition
# a_0 = ws[2]
# poly_0_convex = polytope.qhull(np.array(a_0))
# poly_0_convex_vertices = poly_0_convex.vertices
# trans_0 = transPolytope_list[0]
# trans_0_vertices = trans_0.vertices
# poly_0_trans_vertices = np.vstack((poly_0_convex_vertices, trans_0_vertices))
# poly_0_trans = ConvexHull(poly_0_trans_vertices)
# poly_0_trans_qhull = polytope.qhull(poly_0_trans_vertices)




polytope_merge_plot = []
# Convert the vertices to polytopes
for i in range(len(sequence)):
    poly_k_merge = poly_merge_list[i]
    # for poltting
    poly_merge_vertices_sorted = sort_vertices_clockwise(poly_k_merge.vertices)
    polytope_merge_plot.append(poly_merge_vertices_sorted)





# poly2 = transZone_list[0]
# poly_trans_1 = poly1.union(poly2)
# IndexPolytopeToCheck = [8, 6]
# (A, B) = wsft.transition_zone(ws[IndexPolytopeToCheck[0]], ws[IndexPolytopeToCheck[1]])
# TransZone = polytope.Polytope(np.array(A), np.array(B))

# Find a polytope includes original polytope and transition zone
# example with polytope 0 in sequence[0]
# merge_poly_0_vertices = []
# vertices_rest_trans = []
# # find common vertices
# for v1 in trans_0.vertices:
#     is_common = any(np.allclose(v1, v2) for v2 in poly1.vertices)
#     if is_common:
#         merge_poly_0_vertices.append(v1)
#     else:
#         vertices_rest_trans.append(v1)
            
# # find vertices outside
# vertices_outside_poly0 = []
# for v1 in vertices_rest_trans:
#     Ax = poly1.A @ v1.reshape(-1, 1)
#     if not np.all(Ax.flatten() <= poly1.b):
#         vertices_outside_poly0.append(v1)

# merge_poly_0_vertices.append(vertices_outside_poly0) # includes common vertices and the outside vertices

# # Merge poly and its transition
# poly_merge_1_vertices = []
# for v1 in trans_0.vertices:
#     is_common = any(np.allclose(v1, v2) for v2 in poly1.vertices)
#     Ax = poly1.A @ v1.reshape(-1, 1)
#     is_inside = np.all(Ax.flatten() <= poly1.b)
#     if not(is_common or is_inside):
#         poly_merge_1_vertices.append(v1)
# poly_merge_1_vertices = np.array(poly_merge_1_vertices)
# poly_merge_1_vertices = np.vstack((poly1.vertices, poly_merge_1_vertices))
# poly_merge = polytope.qhull(poly_merge_1_vertices)



fig = plt.figure()
ax = fig.add_subplot(111)

plot_poly_map(ws, ax, 'black')
plt.scatter(init[0], init[1], color='red', s=80, marker='*')
plt.scatter(goal[0], goal[1], color='blue', s=80, marker='*')
# Plot center of each polytope in the path
# Convert center_list to numpy array
center_list_np = np.array(center_list)
# Plot center of polytope in shortest path
selected_center = center_list_np[sequence]
plt.scatter(selected_center[:, 0], selected_center[:, 1], color='black', s=30)

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

# p = Polygon(polytope_merge_plot[0], facecolor='r', alpha = 0.2)
# plt.gca().add_patch(p)
# p = Polygon(polytope_merge_plot[1], facecolor='r', alpha = 0.2)
# plt.gca().add_patch(p)
# p = Polygon(polytope_merge_plot[2], facecolor='r', alpha = 0.2)
# plt.gca().add_patch(p)
# p = Polygon(polytope_merge_plot[3], facecolor='r', alpha = 0.2)
# plt.gca().add_patch(p)

"The map"
plot_poly_map([ws[i] for i in sequence], ax, 'red')
"The obstacles"
p = Polygon(hole1_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
p = Polygon(hole2_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
p = Polygon(hole3_large.polygons[0], facecolor='k', alpha=0.5)
plt.gca().add_patch(p)
plt.show()
