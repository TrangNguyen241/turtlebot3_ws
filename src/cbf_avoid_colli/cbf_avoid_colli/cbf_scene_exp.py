#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import cvxpy as cp
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from mocap4r2_msgs.msg import RigidBodies
import scipy.io
# Export and import data in .mat
from scipy.io import savemat
import os

import time
import sys
import math
import polytope as pc
import matplotlib.pyplot as plt
import control as ct
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

# Messages for visualization in Rviz
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Import functions
import sys
# sys.path.append('/home/nguyehtt/turtlebot3_ws/src')
# from utilities_formation.callback_states import *
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/cbf_avoid_colli')
# from utilities_formation import transformations
from utilities.callback_states import *
from utilities.cbf_funcs import *

# Import Casadi for Lyapunouv-based Control
import casadi as cas 

class MyCBF(Node):
    def __init__(self):
        super().__init__("cbf_mov_static_obs")
        # Parameters of system
        self.MAX_LIN_VEL = 0.22
        self.MIN_LIN_VEL = -self.MAX_LIN_VEL
        self.MAX_ROT_VEL = 2.84
        self.MIN_ROT_VEL = -self.MAX_ROT_VEL
        self.b = 0.1
        self.Ts = 0.1 #sampling tracking

        # System dynamics in continuous time
        self.A = np.zeros((2,2))
        self.B = np.eye(2)
        self.nx = self.A.shape[1]
        self.nu = self.A.shape[1]
        self.N = 2 # number of robots

        # Parameters of LQR controller
        # 75 and 3 : stable
        self.Q = np.diag(np.ones(self.nx) * 15)
        self.R = np.eye(self.nu) * 1
        self.K,self.P_lqr,_ = ct.lqr(self.A, self.B, self.Q, self.R)
        self.P_lya = np.array([[0.1, 0.0],
                               [0.0, 0.1]])

        # Define safe distance
        self.safe_dis = 0.3
        # self.gamma = 0.2  # rate of control barrier function #0.1 avoid from the large dis, #1 avoid from the small dis
        self.gamma = 0.1
        self.omega = 0.1   # lyapunov inequality #0.05
        self.relax_param = 0.1
        self.std_error = 0.01

        # Constraint set for u1, u2
        self.ru = min(self.b*self.MAX_ROT_VEL, self.MAX_LIN_VEL) 
        ptsU = []
        for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
            ptsU.append([self.ru * np.cos(tta), self.ru * np.sin(tta)])
        ptsU = np.array(ptsU)
        self.U_input = pc.qhull(ptsU) #checked
        self.Up = Polygon(ptsU, facecolor=(1, 1, 0, 0.1),edgecolor=(0, 0, 0, 0.8)) # for plotting U

        # Column ith is the target for robot i
        # self.target = np.array([[-0.9, -0.94],
        #                         [-0.94, 0.6]])   # experiment with 2 tb (no collision)
        # self.target = np.array([[1.12, 1.35],
        #                         [1.2, -1.22]])   # experiment with 2 tb (collision) #scene 1
        # self.target = np.array([[-0.5, -1.0],
        #                         [1.0, 0.5]])   # experiment with 2 tb (collision) # case_2
        # self.target = np.array([[1.0, 0.55],
        #                         [-0.47, -1.0]])   # experiment with 2 tb (collision) # case_2_2
        self.target = np.array([[0.46, 0.94],
                                [1.11, -0.38]])   # experiment with 2 tb (collision) # case_2_2
        # self.target = np.array([[-1.3, -1.33],
        #                         [-0.6, 0.54]])   # experiment with 2 tb (collision)
        
        # Obstacles
        self.obstacles = []
        
        # self.obstacles.append(pc.qhull(np.array([[0.0, -0.3],
        #                                         [0.2, -0.3],
        #                                         [0.2, -0.5],
        #                                         [0.0, -0.5]])))
        # self.obstacles.append(pc.qhull(np.array([[0.0, 0.3],
        #                                         [0.3, 0.2],
        #                                         [0.0, 0.0]])))
        # self.obstacles.append(np.array([[0.9, -0.6],
        #                                 [0.9, -0.9],
        #                                 [0.6, -0.9],
        #                                 [0.6, -0.6]]))
        # Scene 2
        # self.obstacles.append(np.array([[-0.58, -0.39],
        #                                 [-0.87, -0.39],
        #                                 [-0.87, -0.65],
        #                                 [-0.58, -0.65]]))  # scene 2
        # self.obstacles.append(np.array([[0.56, 0.93],
        #                                 [0.43, 0.68],
        #                                 [0.26, 0.92]]))
        # Scene 3
        self.obstacles.append(np.array([[0.5, -0.24],
                                        [0.5, -0.51],
                                        [0.2, -0.51],
                                        [0.2, -0.24]]))  # scene 2
        self.obstacles.append(np.array([[-0.07, 0.91],
                                        [0.14, 0.74],
                                        [-0.13, 0.63]]))

        self.num_obs = len(self.obstacles)
        self.list_P_ellipse = []
        self.list_center_ellipse = []
        # Find the smallest bouding ellipsoid of a polytope
        for i in range(self.num_obs):
            # polytope_k_ver = self.obstacles[i].vertices
            polytope_k_ver = self.obstacles[i]
            P_k, c_k = ellipse_outer_polytope(polytope_k_ver)
            self.list_P_ellipse.append(P_k)
            self.list_center_ellipse.append(c_k)
        plot_obstacles_with_ellipses(self.obstacles, self.list_P_ellipse, self.list_center_ellipse)

        # Variables of state robots
        self.robot_subscriptions = {}
        self.robot_publisher = {}
        self.robot_positions = {i: (0, 0, 0) for i in range(self.N)}
        self.u_real = np.zeros((self.nu, self.N))
        self.u_real_safe = np.zeros((self.nu, self.N))

        # For PLOTTING
        self.xplot_0 = []
        self.yplot_0 = []
        self.linvel_0 = []
        self.angvel_0 = []

        self.xplot_1 = []
        self.yplot_1 = []
        self.linvel_1 = []
        self.angvel_1 = []

        self.num_tracking_steps = 0
        self.time_plot = []

        #------------------------ Run simulation --------------------
        # Subscribe to the topic /odom (given by ROS) of 2 robots to receive their positions
        # Subscribe for robot 0
        # self.robot_subscriptions[0] = self.create_subscription(
        #     Odometry,
        #     '/turtlebot30/odom',
        #     lambda msg: self.odom_callback_wrapper(msg, 0),
        #     qos_profile=qos_profile_sensor_data
        # )
        # # Subscribe for robot 1
        # self.robot_subscriptions[1] = self.create_subscription(
        #     Odometry,
        #     '/turtlebot31/odom',
        #     lambda msg: self.odom_callback_wrapper(msg, 1),
        #     qos_profile=qos_profile_sensor_data
        # )



        #----------------------- Run in experiment ------------------
        # Subscribe to the topic /rigid_bodies (given by Qualisys camera system) of 4 robots to receive their positions.
        self.subscription = self.create_subscription(RigidBodies, 'rigid_bodies', self.rigid_bodies_callback, 10)


        # Create a publisher to publish the linear and angular velocities of the 2 robots in experiment
        self.cmd_msg = Twist()
        # Publisher for robot 0
        self.robot_publisher[0] = self.create_publisher(
            Twist,
            '/turtlebot33/cmd_vel',
            10
        )
        # Publisher for robot 0
        self.robot_publisher[1] = self.create_publisher(
            Twist,
            '/turtlebot30/cmd_vel',
            10
        )
        # Timer for control loop
        self.timer = self.create_timer(self.Ts, self.cbf_avoid_collision)

    def odom_callback_wrapper(self, msg, robot_id):
        x, y, theta = odometry_callback(msg = msg)
        self.robot_positions[robot_id] = (x, y, theta)
        # self.get_logger().info(f"Robot {robot_id} position: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")

    # Function of calculating position of robot includes: x, y, theta (Experiment, from /rigidbodies)
    def rigid_bodies_callback(self, msg):
        # Robot 0
        x_0 = msg.rigidbodies[0].pose.position.x
        y_0 = msg.rigidbodies[0].pose.position.y
        z_0 = msg.rigidbodies[0].pose.position.z

        qx_0 = msg.rigidbodies[0].pose.orientation.x
        qy_0 = msg.rigidbodies[0].pose.orientation.y
        qz_0 = msg.rigidbodies[0].pose.orientation.z
        qw_0 = msg.rigidbodies[0].pose.orientation.w

        angles_0 = quaternion2euler(qx_0, qy_0, qz_0, qw_0)
        theta_0 = angles_0[-1]%(2*math.pi)
        self.robot_positions[0] = (x_0, y_0, theta_0)
        self.get_logger().info("x_0: {}, y_0: {}, theta_0: {}".format(self.robot_positions[0][0], self.robot_positions[0][1], self.robot_positions[0][2]))

        # Robot 1
        x_1 = msg.rigidbodies[1].pose.position.x
        y_1 = msg.rigidbodies[1].pose.position.y
        z_1 = msg.rigidbodies[1].pose.position.z

        qx_1 = msg.rigidbodies[1].pose.orientation.x
        qy_1 = msg.rigidbodies[1].pose.orientation.y
        qz_1 = msg.rigidbodies[1].pose.orientation.z
        qw_1 = msg.rigidbodies[1].pose.orientation.w

        angles_1 = quaternion2euler(qx_1, qy_1, qz_1, qw_1)
        theta_1 = angles_1[-1]%(2*math.pi)
        self.robot_positions[1] = (x_1, y_1, theta_1)
        self.get_logger().info("x_1: {}, y_1: {}, theta_1: {}".format(self.robot_positions[1][0], self.robot_positions[1][1], self.robot_positions[1][2]))

        
    def barrier_func(self, u_d, poses_robot, error_tb):
        solver = cas.Opti()
        options = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 3000} # max_iter: 5000
        solver.solver('ipopt', options)
        # Define variables
        u_safe = solver.variable(self.nu, self.N)
        # Define parameters
        u_unsafe = solver.parameter(self.nu, self.N)
        z_robot = solver.parameter(self.nx, self.N)
        error_z = solver.parameter(self.nx, self.N)
        
        objective = 0
        for i in range(self.N):
            objective += cas.norm_2(u_safe[:, i] - u_unsafe[:, i])**2
            solver.subject_to(cas.mtimes(self.U_input.A, u_safe[:, i]) <= self.U_input.b)
        objective += self.relax_param
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                h_br = cas.norm_2(z_robot[:, i] - z_robot[:, j])**2 - self.safe_dis**2
                grad_h_br = cas.vertcat(2*(z_robot[0, i] - z_robot[0, j]), 
                                        2*(z_robot[1, i] - z_robot[1, j]), 
                                        -2*(z_robot[0, i] - z_robot[0, j]),
                                        -2*(z_robot[1, i] - z_robot[1, j]))
                u_safe_ij = cas.vertcat(u_safe[:, i], u_safe[:, j])
                solver.subject_to(cas.mtimes(grad_h_br.T, u_safe_ij) >= -self.gamma * h_br)
        for i in range(self.N):
            solver.subject_to(cas.mtimes(cas.mtimes(2*cas.transpose(error_z[:, i]), self.P_lya), u_safe[:, i]) 
                              <= - self.omega*cas.mtimes(cas.mtimes(cas.transpose(error_z[:, i]), self.P_lya), error_z[:, i]) + self.relax_param)
        # Barrier function for obstacle avoidance
        for i in range(self.N):
            for j in range(len(self.obstacles)):
                p_ellipse = self.list_P_ellipse[j]
                c_ellipse = self.list_center_ellipse[j].reshape((2,1))
                h_br_obs = cas.mtimes(cas.mtimes(cas.transpose(z_robot[:, i] - c_ellipse), p_ellipse),(z_robot[:, i] - c_ellipse)) - 1 -0.1
                grad_h_br_obs = 2*cas.mtimes(p_ellipse, (z_robot[:, i] - c_ellipse))
                u_safe_i = u_safe[:, i]
                solver.subject_to(cas.mtimes(cas.transpose(grad_h_br_obs), u_safe_i) >= -self.gamma*h_br_obs)

        # Minimize objective
        solver.set_value(u_unsafe, u_d)
        solver.set_value(z_robot, poses_robot)
        solver.set_value(error_z, error_tb)
        
        solver.minimize(objective)
        sol = solver.solve()
        u_br_safe = sol.value(u_safe)
        return u_br_safe
    
    def save_data_export(self, robot_idx, target_tb, x_plot, y_plot, time_plot, lin_v_plot, ang_v_plot):
        # *********Define the base directory for saving data in Matlab **********
        base_dir_mat = "/home/nguyehtt/turtlebot3_ws/src/cbf_avoid_colli/data_plotting/plot_matlab/cbf_exp_scene"
        
        # Ensure the directory exists
        os.makedirs(base_dir_mat, exist_ok=True)
        
        # Define filenames based on controller type
        x_pos_filename = f"tb_{robot_idx}_xPos_cbf.mat"
        y_pos_filename = f"tb_{robot_idx}_yPos_cbf.mat"
        position_filename = f"tb_{robot_idx}_Position_cbf.mat"
        lin_vel_filename = f"tb_{robot_idx}_linVel_cbf.mat"
        ang_vel_filename = f"tb_{robot_idx}_angVel_cbf.mat"
        
        # Save data
        scipy.io.savemat(os.path.join(base_dir_mat, x_pos_filename), {'x': time_plot, 'y': x_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, y_pos_filename), {'x': time_plot, 'y': y_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, position_filename), {'x': x_plot, 'y': y_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, lin_vel_filename), {'x': time_plot, 'y': lin_v_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, ang_vel_filename), {'x': time_plot, 'y': ang_v_plot})

        # **********Saving data to .npz to plot in Python**************
        base_dir_py = "/home/nguyehtt/turtlebot3_ws/src/cbf_avoid_colli/data_plotting/plot_python/cbf_exp_scene"
        os.makedirs(base_dir_py, exist_ok=True)
        npz_filename = os.path.join(base_dir_py, f"tb_{robot_idx}_data_cbf.npz")
        np.savez(npz_filename,
                 target = np.array(target_tb),
                 time = np.array(time_plot),
                 x = np.array(x_plot),
                 y = np.array(y_plot),
                 lin_v = np.array(lin_v_plot),
                 ang_v = np.array(ang_v_plot))
        self.get_logger().info(f"Exported robot {robot_idx} data to .mat and .npz")



    def cbf_avoid_collision(self):
        if not(all(all(coord != 0.0 for coord in value) for value in self.robot_positions.values())):
            self.get_logger().warn("Waiting for ODOMETRY......")
            time.sleep(self.Ts)
            return
        print("===============START=================")
        # Get the poses of the robots (tranform dictionaries to numpy array)
        poses = np.array([[self.robot_positions[i][0], self.robot_positions[i][1], self.robot_positions[i][2]] for i in range(self.N)]).T
        print(f"Pose of robots: {poses}")

        # Convert states of robots to single integrator dynamics
        poses_si = np.vstack((poses[0, :] + self.b * np.cos(poses[2, :]),
                              poses[1, :] + self.b * np.sin(poses[2, :]))) 
        error =  poses_si - self.target
        u_lqr = -self.K @ error
        tic = time.time()
        # u_br = self.barrier_func(u_lqr, poses_si, error)
        u_br = self.barrier_func(u_lqr, poses_si, error)
        toc = time.time()
        denta_time = toc - tic
        print(f"Computation time: {denta_time}")
        # For plotting
        self.num_tracking_steps += 1
        time_series = self.num_tracking_steps * self.Ts
        self.time_plot.append(time_series)
        self.xplot_0.append(poses_si[0, 0])
        self.yplot_0.append(poses_si[1, 0])

        self.xplot_1.append(poses_si[0, 1])
        self.yplot_1.append(poses_si[1, 1])



        ######## Edit publisher according to index of tb in experiment #######
        for i in range(self.N):
            self.u_real[:, i] = np.array([[np.cos(poses[2, i]), np.sin(poses[2, i])],
                               [-np.sin(poses[2, i])/self.b, np.cos(poses[2, i])/self.b]]) @ u_br[:, i]
            self.u_real_safe[0, i] = np.min([self.u_real[0,i], (2.6 - np.abs(self.u_real[1,i]))/2.6 * 0.21])
            self.u_real_safe[1, i] = self.u_real[1, i]

             # Publish command for robots
            self.cmd_msg.linear.x = self.u_real_safe[0, i]
            self.cmd_msg.angular.z = self.u_real_safe[1, i]
            self.robot_publisher[i].publish(self.cmd_msg)
        
        # # For plotting
        self.linvel_0.append(self.u_real_safe[0, 0])
        self.angvel_0.append(self.u_real_safe[1, 0])

        self.linvel_1.append(self.u_real_safe[0, 1])
        self.angvel_1.append(self.u_real_safe[1, 1])

        print(f"u_real: {self.u_real}")
        print(f"u_real_safe: {self.u_real_safe}")

        # Check whether robot reach target or not
        norm_error = np.linalg.norm(error, axis=0)
        if np.all(norm_error < self.std_error):
            self.get_logger().info("All robots reached their targets. Stopping...")
            # Publish velocity 0 for robots
            for i in range(self.N):
                self.cmd_msg.linear.x = 0.0
                self.cmd_msg.angular.z = 0.0
                self.robot_publisher[i].publish(self.cmd_msg)
                # # For plotting
                # self.linvel_0.append(0.0)
                # self.angvel_0.append(0.0)

                # self.linvel_1.append(0.0)
                # self.angvel_1.append(0.0)

            target_0 = self.target[:, 0]
            target_1 = self.target[:, 1]
            self.save_data_export(0, target_0, self.xplot_0, self.yplot_0, self.time_plot, self.linvel_0, self.angvel_0)
            self.save_data_export(1, target_1, self.xplot_1, self.yplot_1, self.time_plot, self.linvel_1, self.angvel_1)
            self.timer.cancel()
            return
        


        

def main(args = None):
    rclpy.init(args=args)
    node = MyCBF()
    try: 
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()