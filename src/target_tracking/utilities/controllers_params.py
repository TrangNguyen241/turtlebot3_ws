import numpy as np

# Parameters for Saturated Control
saturated_params = {
    "alpha": 0.1, #c1: 0.1
    "gamma": 3.0,
}

# Parameters for LQR + Lyapunov Control
lqr_lyapunov_params = {
    "Q": np.eye(2) * 10,
    "R": np.eye(2) * 1,
    "omega": 0.1,
    "alpha": 0.1
}

# Parameters for Explicit MPC
''' We import exmp_solution.py in the main control file "target_tracking_controllers.py"'''

# Parameters for Implicit MPC
implicit_mpc_params = {
    "Q": np.diag([1.5, 1.5]),
    "R": np.diag([30, 30]),
    "P": np.diag([67.8362318214405, 67.8362318214405]),
    "Npred": 5
}