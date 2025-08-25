import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import polytope as pc
# Import functions
import sys
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/cbf_avoid_colli')

# from utilities_formation import transformations
from utilities.cbf_funcs import *

def khachiyan_algorithm(points, tol=1e-5, max_iter=1000):
    """
    Implements Khachiyan Algorithm to find the minimum volume enclosing ellipse.
    Returns the matrix P and center c such that (x - c)^T P (x - c) <= 1 defines the ellipse.
    
    Args:
        points: ndarray of shape (N, d), N points in d-dimensional space.
        tol: convergence tolerance.
        max_iter: maximum number of iterations.

    Returns:
        P: shape matrix of the ellipse (d x d)
        c: center of the ellipse (d,)
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N)))  # Augmented matrix
    u = np.ones(N) / N  # Uniform initial weights

    for _ in range(max_iter):
        X = Q.T @ np.diag(u) @ Q
        M = np.einsum('ij,jk,ki->i', Q, np.linalg.inv(X), Q.T)  # Mahalanobis distance
        max_idx = np.argmax(M)
        max_M = M[max_idx]

        step_size = (max_M - d - 1) / ((d + 1) * (max_M - 1))
        new_u = (1 - step_size) * u
        new_u[max_idx] += step_size

        if np.linalg.norm(new_u - u) < tol:
            break
        u = new_u

    # Compute center and shape matrix
    center = points.T @ u
    cov = (points - center).T @ np.diag(u) @ (points - center)
    P = np.linalg.inv(cov) / d
    return P, center

def plot_polytope_and_ellipse(points, P, c):
    """
    Plot the polytope defined by vertices and the bounding ellipse.

    Args:
        points: ndarray of shape (N, 2), the polytope vertices
        P: shape matrix of the ellipse
        c: center of the ellipse
    """
    fig, ax = plt.subplots()
    points = np.array(points)

    # Plot polygon (polytope)
    hull = points[np.append(np.arange(len(points)), 0)]
    ax.plot(hull[:, 0], hull[:, 1], 'bo-', label='Polytope vertices')

    # Plot ellipse
    eigvals, eigvecs = np.linalg.eigh(P)
    axes_lengths = 1.0 / np.sqrt(eigvals)
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ellipse = Ellipse(xy=c, width=2*axes_lengths[0], height=2*axes_lengths[1],
                      angle=angle, edgecolor='r', fc='none', lw=2, label='Bounding Ellipse')
    ax.add_patch(ellipse)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Minimum Volume Enclosing Ellipse (Khachiyan Algorithm)")
    plt.grid(True)
    plt.show()

# === Example usage ===
if __name__ == "__main__":
    # vertices = np.array([
    #     [0.5, 0.5],
    #     [1.2, 0.4],
    #     [1.0, 1.3]
    # ])
    # vertices = np.array([
    #     [2.3, 1.],
    #     [1.6, 0.9],
    #     [1.5, 1.5],
    #     [2.2, 2.0]
    # ])
    vertices = np.array([
        [0.2, 1.3],
        [0.5, 1.4],
        [0.4, 1.7],
        [0.3, 1.6]
    ])
    # vertices = np.array([
    #     [1.0, 2.5], 
    #     [1.5, 2.5], 
    #     [2.0, 3.3], 
    #     [0.5, 3.5]
    # ])
    # vertices = np.array([
    #     [3.0, 1.5], 
    #     [2.7, 2.5], 
    #     [3.2, 3.4], 
    #     [3.7, 3.0], 
    #     [3.5, 2.0]
    # ])
    # vertices = np.array([
    #     [2.3, 0.0], 
    #     [2.3, 0.5], 
    #     [2.8, 0.5], 
    #     [2.8, 0.0]
    # ])

    

    P, c = khachiyan_algorithm(vertices)
    plot_polytope_and_ellipse(vertices, P, c)

