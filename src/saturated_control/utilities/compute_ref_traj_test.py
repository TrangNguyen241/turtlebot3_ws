import casadi as cas
import numpy as np

from scipy.interpolate import BSpline
from scipy.integrate import quad
from time import time
from scipy import interpolate
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import ConvexHull, HalfspaceIntersection
import polytope as pc
from scipy.io import savemat
import scipy.io

def bsplineConversionMatrices(num_of_controlPoints: int, k: int, knot):
    # num_of_controlPoints = number of control points
    # polynomial order of spline: k
    # knot vector: knot
    n = num_of_controlPoints - 1
    d = k + 1
    tmp = np.eye(n + 1)
    M = []
    for r in range(d):
        M.append(np.zeros((n + r + 1, n + r + 1 + 1)))
        for i in range(n + r + 1):
            if knot[i + d - r - 1] == knot[i]:
                M[r][i][i] = 0
            else:
                M[r][i][i] = (d - r - 1) / (knot[i + d - r - 1] - knot[i])
            if knot[i + d - r] == knot[i + 1]:
                M[r][i][i + 1] = 0
            else:
                M[r][i][i + 1] = -(d - r - 1) / (knot[i + d - r + 1 - 1] - knot[i + 1])
        tmp = tmp * np.matrix(M[r])
        M[r] = tmp
    return M

def b_spline_basis_functions(num_of_control_points: int, degree_k: int, knot: int) -> list:
    basis_list = []
    for i in range(degree_k + 1):
        list_tmp = []
        for j in range(i + num_of_control_points):
            ctr_point = [0] * (num_of_control_points + i)
            ctr_point[j] = 1.0
            basis_spln_tmp = interpolate.BSpline(knot, ctr_point, degree_k - i)
            list_tmp.append(basis_spln_tmp)
        basis_list.append(list_tmp)
    return basis_list

def knot_vector(polynomial_degree_k, number_of_control_points, *argv):
    # d = k + 1 (polynomial order +1)
    d = polynomial_degree_k + 1
    # n = number_of_control_points - 1
    n = number_of_control_points - 1
    if number_of_control_points < polynomial_degree_k + 1:
        print('Number of control points: {nc} \nDegree of B-spline: {deg_k}'.format(nc=number_of_control_points,
                                                                                    deg_k=polynomial_degree_k))
        raise Exception("The number of control points need to be higher than or at least equal "
                        "to B-spline degree + 1 ")
    if len(argv) == 0:
        t0 = 0
        tf = 1
    else:
        t0 = min(argv[0])
        tf = max(argv[0])
    knot_tmp = np.linspace(t0, tf, n - d + 3, endpoint=True)
    knot_tmp = np.append([t0] * (d - 1), knot_tmp)
    knot_tmp = np.append(knot_tmp, [tf] * (d - 1))
    return knot_tmp
def get_vertices_Ab(A, b, feasible_point):
    # Combine A and -b to form the half-space representation
    Ab = np.column_stack([A, -b])

    # Compute intersections of the half-spaces
    Hs = HalfspaceIntersection(Ab, feasible_point)

    # Extract intersection points
    x, y = zip(*Hs.intersections)
    points = np.stack([x, y], axis=1)

    # Get vertices using ConvexHull
    Pv = ConvexHull(points)

    # Extract vertices in order
    vertices = np.empty((0, 2))
    for n in Pv.vertices:
        vertices = np.vstack((vertices, points[n]))

    return vertices
# def compute_reference_trajectory(num_of_control_points, way_points, k):
# def compute_reference_trajectory():
k = 4
num_of_control_points = 10
# way_points = np.array([[0.15, -0.05, 0,  0.25],
#                         [0, 0.1, 0.3, 0.3]])
way_points = np.array([[0, 0.3, 1, 1.6, 1.9, 1, 0.2, 2],
                       [0, 1.8, 1.4, 1.8, 1, 0.8, 0.5, 0.1]])
time_steps = 1100 # = knot/sampling time
knot = [0,110] # Trajectory will be generated from 0 to 60 seconds
knot = knot_vector(k, num_of_control_points, knot)
tt = np.linspace(min(knot), max(knot)-0.000001, time_steps)
bs_list = b_spline_basis_functions(num_of_control_points, k, knot)
M = bsplineConversionMatrices(num_of_control_points, k, knot)

waypoint_time_stamps = np.linspace(min(knot), max(knot), way_points.shape[1])
ctrl_pts_timestamps = np.linspace(min(knot), max(knot), num_of_control_points)
#****************** OPTIMIZATION PROBLEM ***************************
solver = cas.Opti()
# Control point as optimization variable
P = solver.variable(way_points.shape[0], num_of_control_points)
# Objective function
objective = 0

P1 = cas.mtimes(P, M[0])  # Conversion matrix M
# Objective function
for i in range(num_of_control_points + 1):
    for j in range(num_of_control_points + 1):
        f_lamb = lambda t, it=i, jt=j: bs_list[1][it](t) * bs_list[1][jt](t)
        buff_int = quad(f_lamb, min(knot), max(knot))[0]
        objective = objective + cas.mtimes(cas.transpose(cas.mtimes(buff_int, P1[:, i])), P1[:, j])

# Implementing constraints
for i in range(way_points.shape[1]):
    tmp_bs = np.zeros((len(bs_list[0]), 1))
    for j in range(len(bs_list[0])):
        tmp_bs[j] = bs_list[0][j](waypoint_time_stamps[i])
    # Mathematically, mtimes(P, tmp_bs) = P * tmp_bs
    solver.subject_to(cas.mtimes(P, tmp_bs) == way_points[:, i])
    solver.subject_to(cas.diag(cas.mtimes(P1.T, P1)) < 0.22**2)
    # for i in range(P1.shape[1]):  # Lặp qua từng cột của P1
    #     solver.subject_to(cas.sumsqr(P1[:, i]) <= 0.22**2)

#Solve
solver.minimize(objective)
solver_options = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
solver.solver('ipopt', solver_options)
tic = time()
sol = solver.solve()  # Solve for the control points
toc = time()
Elapsed_time = toc - tic
print('Elapsed time for solving: ', Elapsed_time, '[second]')
# Construct the result curve
P = sol.value(P)
print('=======================================================================================')
print('P=', P)
# Compute the Bspline with the solution of P 
spn = []
if way_points.shape[0] == 1:
    spn.append(BSpline(knot, P, k))
else:
    for i in range(P.shape[0]):
        spn.append(BSpline(knot, P[i], k))
# Advanced ...
# First derivative of the flat output
P1 = np.array(P * M[0])
traj_d = []
for i in range(P1.shape[0]):
    traj_d.append(BSpline(knot, P1[i], k - 1))
# Second derivative of the flat output
P2 = np.array(P * M[1])
traj_dd = []
for i in range(P2.shape[0]):
    traj_dd.append(BSpline(knot, P2[i], k - 2))
# Find set of (U - U_ref = U_e): using matlab to do Pontryagin difference, so we will extract A and b on u_ref_hull in python, bring them to matlab 
# This is the U_ref hull
traj_d_array = np.array([traj_d[0](tt), traj_d[1](tt)]).T
u_ref_hull = ConvexHull(traj_d_array)  # set of reference
A_u_ref_hull = u_ref_hull.equations[:, :-1]
b_u_ref_hull = - u_ref_hull.equations[:, -1]
# Save to matlab file
savemat('/home/nguyehtt/turtlebot3_ws/src/saturated_control/utilities/u_ref_hull.mat', {'A': A_u_ref_hull, 'b': b_u_ref_hull})
print("Saved convex hull representation to 'u_ref_hull.mat'")

# Constraint set of robot
ptsU = []
ru = 0.22
for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
    ptsU.append([ru * np.cos(tta), ru * np.sin(tta)])
ptsU = np.array(ptsU)
u_robot = ConvexHull(ptsU) # set of constraint of robot

# Load data from matlab 
Ue_data = scipy.io.loadmat('/home/nguyehtt/turtlebot3_ws/src/saturated_control/utilities/U_e_input_python.mat')
A_Ue = Ue_data['A_Ue_input']
b_Ue = Ue_data['b_Ue_input']
Ue_input_hull = pc.Polytope(A_Ue, b_Ue)
### Get vertices for plotting
# Define a feasible point
feasible_point = np.array([0, 0])
# Obtain the vertices of the polytope
Ue_vertices = get_vertices_Ab(A_Ue, b_Ue, feasible_point)

# Plotting trajectory and derivative of trajectory
fig1 = plt.figure() 
gs = gridspec.GridSpec(2, 2) 

# Plot reference trajectory of robot
ax1 = fig1.add_subplot(gs[:2, 0])

ax1.set_title('Reference trajectory')
ax1.plot(spn[0](tt), spn[1](tt), 'b',label = "Reference trajectory")
ax1.plot(P[0], P[1], lw=1)
ax1.scatter(way_points[0, :], way_points[1, :], label='waypoints', color='red', lw=5)
ax1.scatter(P[0], P[1], label='Control Points')
ax1.set_xlabel('z1 (m)', fontsize=13)
ax1.set_ylabel('z2 (m)', fontsize=13)
# ax1.axis('tight')
ax1.legend()
ax1.grid(True)
# Plot derivative of trajectory
ax2 = fig1.add_subplot(gs[0, 1])
ax2.set_title('Derivative of reference trajectory')
ax2.plot(tt, traj_d[0](tt), 'm', lw = 2,label = "Derivative of z1")
# ax1.plot(P1[0], P1[1], lw=1)
ax2.set_xlabel('time (m)', fontsize=13)
ax2.set_ylabel('z1_dot', fontsize=13)
# ax1.axis('tight')
ax2.legend()
ax2.grid(True)

ax3 = fig1.add_subplot(gs[1, 1])
ax3.set_title('Derivative of reference trajectory')
ax3.plot(tt, traj_d[1](tt), 'm', lw = 2,label = "Derivative of z2")
# ax1.plot(P1[0], P1[1], lw=1)
ax3.set_xlabel('time (m)', fontsize=13)
ax3.set_ylabel('z2_dot', fontsize=13)
# ax1.axis('tight')
ax3.legend()
ax3.grid(True)
# Draw P1 in constraint set
# Draw circle of constraint set
theta = np.linspace(0, 2 * np.pi, 100)  
ru = 0.22
x = ru * np.cos(theta) 
y = ru * np.sin(theta)  
fig2 = plt.figure() 
plt.title("P1 plot")
plt.plot(x, y,'k', label = 'Input constraint set')
plt.plot(traj_d[0](tt), traj_d[1](tt), 'g', label = "Derivative of z1 and z2")

# Plotting polytope bouding traj_d
for simplex in u_ref_hull.simplices:
    plt.plot(traj_d_array[simplex, 0], traj_d_array[simplex, 1], 'r-')
plt.fill(traj_d_array[u_ref_hull.vertices, 0], traj_d_array[u_ref_hull.vertices, 1], 'r', alpha=0.3, label='Constraint set Uref')

# Plotting polytope contraints of robot
for simplex in u_robot.simplices:
    plt.plot(ptsU[simplex, 0], ptsU[simplex, 1], 'g-')

# Plotting Ue
Ue_polygon = plt.Polygon(Ue_vertices, color='b', alpha=0.4, linewidth=2, label='Constraint set Ue')
plt.gca().add_patch(Ue_polygon)

plt.legend()
plt.axis(True)
plt.show()
