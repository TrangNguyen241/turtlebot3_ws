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

def check_corridor(cor, pos):
    """
    Check if the position is within the corridor constraints.

    Parameters:
    cor: Object with attributes `A` (matrix) and `b` (vector).
    pos: Position vector.

    Returns:
    bool: True if all constraints are satisfied, False otherwise.
    """

    return np.all(cor.A @ pos <= cor.b.reshape(-1,1))

def RGThreshold(A, h, P, xv, v):
    """
    Compute the RGThreshold value.

    Parameters:
    A: Constraint matrix (numpy array).
    h: Constraint bounds (numpy array).
    P: Positive definite matrix (numpy array).
    xv: State vector (numpy array).
    v: Auxiliary vector (numpy array).

    Returns:
    float: The minimum gamma value computed from the constraints.
    """
    nc = len(h)  # Number of constraints
    nx = len(xv) # Number of state variables

    gamma_list = np.zeros(nc) 
    for i in range(nc):

        # print(f"Ai_xv: {A[i, :nx]}")
        # print(f"Shape of Ai_xv: {A[i, :nx].shape}")
        
        # print(f"Ai_v: {A[i, nx:]}")
        # print(f"Shape of Ai_v: {A[i, nx:].shape}")
        
        # print("INSIDE RGTHRESHOLD FUNCTION!!!!!")
        numerator = (A[i, :nx] @ xv + A[i, nx:] @ v - h[i])**2
        denominator = A[i, :nx] @ np.linalg.inv(P) @ A[i, :nx].T

        # print(f"numerator: {numerator}")
        # print(f"denominator: {denominator}")
        # print(f"shape of numerator: {numerator.shape}")
        # print(f"shape of denominator: {denominator.shape}")

        gamma_list[i] = numerator / denominator

    return np.min(gamma_list)

def decomp_ws(ws_limit, obstacle_list, safety_offset=0.0):
    ''' 
    ws_limit: list of points defining the workspace boundary  
    obstacle_list: list of obstacles (each obstacle is a list of points (vertices of obstacle))  
    safety_offset: margin to grow obstacles (default is 0.0)
    '''
    # Define workspace boundary
    ws_poly = gdspy.Polygon(ws_limit)

    # Define obstacle as polygon object
    obstacle_polygons = [gdspy.Polygon(obs) for obs in obstacle_list]

    # Define obstacle with offset safety
    safety_offset = 0.0
    obstacle_large = [gdspy.offset(obs_poly, safety_offset) for obs_poly in obstacle_polygons]

    # Substract obstacles from workspace to get free space
    poly_with_hole = gdspy.boolean(ws_poly, obstacle_large, "not")

    # Decompose resulting polygon into convex parts
    ws_robot = pd.polygonQuickDecomp(poly_with_hole.polygons[0])

    return ws_robot

def shortest_path_idx_poly(init, goal, ws_robot):
    '''This function return sequence of index of shortest path'''
    n_polyp = len(ws_robot)
    center_list = []

    # Find center of each polytope
    for j in range(n_polyp):
        center_list.append(wsft.center(ws_robot[j]).A1)

    # Find the shortest path
    N = np.empty((0, n_polyp), float)
    N_temp = np.array([])
    for i in range(n_polyp):
        N_temp = []
        for j in range(n_polyp):
            if wsft.check_consecutive_polytopes(ws_robot[i], ws_robot[j]) and (i != j):
                temp = [center_list[i], center_list[j]]
                N_temp.append(LA.norm(temp))
            
            else:
                N_temp.append(0)
        N = np.append(N, [N_temp], axis=0)
    # Dijkstra’s algorithm to find the shortest path
    D, Pr = shortest_path(N, directed=False, method='D', return_predecessors=True)

    init_goal_idx = [wsft.find_polyp(ws_robot, init), wsft.find_polyp(ws_robot, goal)]
    sequence = wsft.get_path(Pr, init_goal_idx[0], init_goal_idx[1])
    print("Find path to go from ", init, "to ", goal)
    print("The index of the sequence of polytopes are: ", sequence)
    return sequence

def path_poly_transition(sq_path, ws_robot):
    '''This function returns enlarged polytope which is merged by original polytope with
      its transition zone formed by two neigboring polytopes'''
    transZone_list = []
    transPolytope_list = []
    # Find the transitioning polytope formed by two neigboring polytopes
    for i in range(len(sq_path)-1):
        IndexPolytopeToCheck = [sq_path[i], sq_path[i+1]]
        (A,B) = wsft.transition_zone(ws_robot[IndexPolytopeToCheck[0]], ws_robot[IndexPolytopeToCheck[1]])
        transZone_k = polytope.Polytope(np.array(A), np.array(B))
        transZone_list.append(transZone_k)
        # convert zone to polytope
        # transZone_k_vertices = wsft.vertices_from_halfspace(np.array(A), np.array(B))
        transZone_k_vertices = wsft.vertices_from_halfspace(transZone_k.A, transZone_k.b)
        trans_k_polytope = polytope.qhull(np.array(transZone_k_vertices))
        transPolytope_list.append(trans_k_polytope)

    # Merging polytopes with their transition zones
    poly_merge_list = []
    for i in range(len(sq_path)):
        poly_k = ws_robot[sq_path[i]]
        poly_k_vertices = np.array(poly_k)
        if i==0:
            trans_k = transPolytope_list[0]
            trans_k_vertices = trans_k.vertices
        elif i==len(sq_path)-1:
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
    for i in range(len(sq_path)):
        poly_k_merge = poly_merge_list[i]
        # Sorting the vertices in clockwise direction (necessary for plotting)
        poly_merge_vertices_sorted = wsft.sort_vertices_clockwise(poly_k_merge.vertices)
        polytope_merge_plot.append(poly_merge_vertices_sorted)
    return poly_merge_list, transPolytope_list, polytope_merge_plot








