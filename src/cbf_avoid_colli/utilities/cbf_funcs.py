import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def ellipse_outer_polytope(points, tol=1e-5, max_iter=1000):
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

def plot_obstacles_with_ellipses(obstacles, list_P_ellipse, list_center_ellipse):
    """
    Plot all obstacles in self.obstacles along with their minimum volume enclosing ellipses.
    Each obstacle must have .V as the array of vertices.
    """
    fig, ax = plt.subplots()

    for i, obs in enumerate(obstacles):
        # points = np.array(obs.vertices)
        # hull = points[np.append(np.arange(len(points)), 0)]
        # ax.plot(hull[:, 0], hull[:, 1], 'bo-')

        # Plot obstacle vertices
        # vertices = obs.vertices
        vertices = obs
        vertices = np.vstack((vertices, vertices[0]))  # Close the polygon
        ax.plot(vertices[:, 0], vertices[:, 1], 'b-', label=f"Obstacle {i+1}" if i == 0 else None)
        ax.fill(vertices[:, 0], vertices[:, 1], 'b', alpha=0.2)

        # Bounding ellipse
        # P, c = ellipse_outer_polytope(points)
        P = list_P_ellipse[i]
        c = list_center_ellipse[i].flatten()

        eigvals, eigvecs = np.linalg.eigh(P)
        axes_lengths = 1.0 / np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = Ellipse(xy=c, width=2*axes_lengths[0], height=2*axes_lengths[1],
                          angle=angle, edgecolor='r', fc='none', lw=2)
        ax.add_patch(ellipse)

    ax.set_aspect('equal')
    ax.set_title("Obstacles with Bounding Ellipses (Khachiyan)")
    plt.grid(True)
    plt.show()