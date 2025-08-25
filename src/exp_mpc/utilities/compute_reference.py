import casadi as cas
import numpy as np

from scipy.interpolate import BSpline
from scipy.integrate import quad
import time
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


def compute_reference_trajectory(num_of_control_points, way_points, k):
    dt = 0.000001 
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
    # Final velocity is zero
    # for i in range(way_points.shape[1]):
    #     tmp_bs = np.zeros((len(bs_list[1]), 1))
    #     for j in range(len(bs_list[1])):
    #         tmp_bs[j] = bs_list[1][j](waypoint_time_stamps[-1] - dt)
    #     solver.subject_to(cas.mtimes(P1, tmp_bs) == 0)

    # for i in range(way_points.shape[1]):
    #     tmp_bs = np.zeros((len(bs_list[1]), 1))
    #     for j in range(len(bs_list[1])):
    #         tmp_bs[j] = bs_list[1][j](waypoint_time_stamps[0] + dt)
    #     solver.subject_to(cas.mtimes(P1, tmp_bs) == 0)

    #Solve
    solver.minimize(objective)
    solver_options = {'ipopt': {'print_level': 0, 'sb': 'yes'}, 'print_time': 0}
    solver.solver('ipopt', solver_options)
    tic = time.time()
    sol = solver.solve()  # Solve for the control points
    toc = time.time()
    Elapsed_time = toc - tic
    print('Elapsed time for solving: ', Elapsed_time, '[second]')
    # Construct the result curve
    P = sol.value(P)
    # print('=======================================================================================')
    # print('P =', P)
    # Compute the Bspline with the solution of P 
    ref_traj = []
    if way_points.shape[0] == 1:
        ref_traj.append(BSpline(knot, P, k))
    else:
        for i in range(P.shape[0]):
            ref_traj.append(BSpline(knot, P[i], k))
    # Convert ref_traj to numpy array
    ref_traj = np.array([ref_traj[0](tt), ref_traj[1](tt)])
    # First derivative of the flat output
    P1 = np.array(P * M[0])
    traj_d = []
    for i in range(P1.shape[0]):
        traj_d.append(BSpline(knot, P1[i], k - 1))
    # Convert ref_traj to numpy array
    traj_d = np.array([traj_d[0](tt), traj_d[1](tt)])
    return ref_traj, traj_d
k = 4
# num_of_control_points = 30 # after
num_of_control_points = 10
way_points = np.array([[0, 0.3, 1, 1.6, 1.9, 1, 0.2, 2],
                       [0, 1.8, 1.4, 1.8, 1, 0.8, 0.5, 0.1]])

a = []
b = []
a, b = compute_reference_trajectory(num_of_control_points, way_points, k)
print("Done")