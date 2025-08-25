#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Initial setiing
Ts = 0.1 # sampling time
T = 200 # Simulation time
v_max, w_max = 0.22, 2.84  # giới hạn đầu vào

# Environment setup
np.random.seed(42)
obstacles = np.array([[2.5, 2.5], [3.5, 1.5], [1.0, 3.8]])
obs_radius = 0.4
safe_radius = 0.6 

# Function to check safety
def is_safe(q, obstacles, r_safe):
    for obs in obstacles:
        distance = np.linalg.norm(q-obs)
        if distance < r_safe:
            return False
    return True

# Create Q
x_vals = np.arange(0, 5.01, 0.5)
y_vals = np.arange(0, 5.01, 0.5)
Q_all = np.array([[x,y] for x in x_vals for y in y_vals])
Q = np.array([q for q in Q_all if is_safe(q, obstacles, safe_radius)])

# Compute W(q)
def compute_W(q):
    distances = [np.linalg.norm(q-obs) for obs in obstacles]
    min_dis = min(distances)
    return 0.5*(min_dis - obs_radius)**2

W_dict = {tuple(q): compute_W(q) for q in Q}
# CBF
# def h_q(x,q,Wq):
#     v_lyapunov = 0.5*(x - q)**2
#     h_q = Wq - v_lyapunov
#     return h_q
def h_q(x, q, Wq):
    return Wq - 0.5 * np.linalg.norm(x - q)**2

def dh_q_dx(x, q):
    return -(x - q)

# --- Trajectory lưu ---
trajectory = []
h_traj = []

# --- Trạng thái khởi tạo ---
xi = np.array([0.5, 0.5, 0.0])  # [x, y, theta]
b = 0.3  # khoảng cách tới điểm điều khiển
goal = np.array([4.5, 4.5])

for t in range(int(T / Ts)):
    # --- Tính điểm điều khiển x ---
    x = xi[:2] + b * np.array([np.cos(xi[2]), np.sin(xi[2])])

    # --- Tín hiệu mong muốn ---
    u_ref = -1.0 * (x - goal)  # controller đơn giản
    u_ref = np.clip(u_ref, -v_max, v_max)

    # --- Tìm q tốt nhất ---
    best_q = None
    best_h = -np.inf
    for q in Q:
        h = h_q(x, q, W_dict[tuple(q)])
        if h > best_h:
            best_h = h
            best_q = q

    # --- CBF-QP ---
    u = cp.Variable(2)
    h = h_q(x, best_q, W_dict[tuple(best_q)])
    grad_h = dh_q_dx(x, best_q)
    alpha = 0.5

    constraints = [
    grad_h @ u + alpha * h >= 0,
    cp.abs(u[0]) <= v_max,
    cp.abs(u[1]) <= v_max  # u là u' = \dot{x}, nên nên giới hạn giống v_max
    ]


    objective = cp.Minimize(cp.sum_squares(u - u_ref))
    prob = cp.Problem(objective, constraints)
    prob.solve()

    u_star = u.value if u.value is not None else u_ref

    # --- Biến đổi về (v, w) từ u_star ---
    T_fl = np.array([
        [np.cos(xi[2]), -b * np.sin(xi[2])],
        [np.sin(xi[2]),  b * np.cos(xi[2])]
    ])
    v_omega = np.linalg.pinv(T_fl) @ u_star
    v = np.clip(v_omega[0], -v_max, v_max)
    w = np.clip(v_omega[1], -w_max, w_max)

    # --- Cập nhật trạng thái ---
    xi[0] += v * np.cos(xi[2]) * Ts
    xi[1] += v * np.sin(xi[2]) * Ts
    xi[2] += w * Ts

    # --- Ghi lại ---
    trajectory.append(xi.copy())
    h_traj.append(h)

    if np.linalg.norm(xi[:2] - goal) < 0.1:
        break

# --- Vẽ kết quả ---
trajectory = np.array(trajectory)

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label="trajectory")
ax.plot(goal[0], goal[1], 'go', label="goal")

for obs in obstacles:
    circle = plt.Circle(obs, obs_radius, color='r')
    ax.add_patch(circle)
Q_np = np.array(Q)  # đảm bảo Q là mảng numpy
ax.plot(Q_np[:, 0], Q_np[:, 1], 'kx', label='CBF centers (q)')

ax.legend()
plt.title("TurtleBot path with Lyapunov-based CBF safety filter")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.show()