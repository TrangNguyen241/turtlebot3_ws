#!/usr/bin/env python3

# Necessary ROS packages
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# Topics packages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from mocap4r2_msgs.msg import RigidBodies

# Export and import data in .mat
from scipy.io import savemat
import scipy.io
import os

# Basic Python packages
import numpy as np
import time
import sys
import math
import ast  # To safely convert string to list

# Packages for controllers
import polytope as pc
import cvxpy as cp
import control as ct
# Import Casadi for Lyapunouv-based Control
import casadi as cas 

# Plotting packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

# Messages for visualization in Rviz
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Import neccessary functions
import sys
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/target_tracking')
from utilities.exmpc_solution import *
from utilities.controllers_params import (
    saturated_params, lqr_lyapunov_params, implicit_mpc_params
)

class TargetControl(Node):
    def __init__(self):
        #####################################################################
        #                       Initialization                              #
        #####################################################################
        super().__init__('target_tracking_controllers')
        #########################################
        #       Declare parameters of system    # 
        #########################################

        # Parameters of system TurtleBot3
        self.MAX_LIN_VEL = 0.22
        self.MIN_LIN_VEL = -self.MAX_LIN_VEL
        self.MAX_ROT_VEL = 2.84
        self.MIN_ROT_VEL = -self.MAX_ROT_VEL
        # self.Ts = 0.1 # sampling time
        self.Ts = 0.1 # sampling time
        # self.Ts = 0.05 # sampling time
        self.b = 0.1 # the control point in front of robot #0.1
        self.A = np.zeros((2,2)) # matrix A in continuous time
        self.B = np.eye(2) # matrix B in continuos time
        self.Ad = np.eye(2) + self.A * self.Ts # matrix A in discrete time
        self.Bd = self.B * self.Ts # matrix B in discrete time
        self.nx = self.A.shape[1]
        self.nu = self.A.shape[1]
        # Constraint set for u1, u2
        self.ru = min(self.b*self.MAX_ROT_VEL, self.MAX_LIN_VEL) # radius of ball constraint of u
        ptsU = []
        for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
            ptsU.append([self.ru * np.cos(tta), self.ru * np.sin(tta)])
        ptsU = np.array(ptsU) # approximation of constraints set as a polytope
        self.U_input = pc.qhull(ptsU) 
        self.Up = Polygon(ptsU, facecolor=(1, 1, 0, 0.1),edgecolor=(0, 0, 0, 0.8)) # for plotting U_input

        # Compute matrix P of saturated and LQR + Lyapunov control once and store it
        self.P_sat = self.compute_P_sat_lya(saturated_params["alpha"])
        self.P_lya = self.compute_P_sat_lya(lqr_lyapunov_params["alpha"])

        # Import X_f of MPC for model
        self.Xf_mpc = scipy.io.loadmat('/home/nguyehtt/turtlebot3_ws/src/target_tracking/utilities/Xf_target.mat')
        self.Xf_A = self.Xf_mpc['A_Xf']
        self.Xf_b = self.Xf_mpc['b_Xf']

        # States of robot
        self.x = None
        self.y = None
        self.theta = None

        # Input control
        self.u_real = np.zeros((2,1))
        self.u_real_safe = np.zeros((2,1))

        # Flag finish tracking
        self.target_tracking_finish = False

        # Desired error tracking
        self.d_error = 0.1

        # Time variable to evaluate computation time of controller
        self.start_time = 0.0
        self.end_time = 0.0
        self.comp_time = 0.0
        self.total_comp_time = 0.0
        self.avg_comp_time = 0.0

        # Variable store data for plotting
        # Positions of robot
        self.x_plot = []
        self.y_plot = []
        self.target_plot = np.zeros((2,1))
        # For plotting velocity of robot
        self.num_tracking_steps = 0
        self.time_plot = []
        self.lin_v_plot = []
        self.ang_v_plot = []
        self.u_1_plot = []
        self.u_2_plot = []

        ########################################
        #       Treating arguments             #
        ########################################
        
        # Declare ROS 2 parameters for selecting controller type and target
        self.declare_parameter('controller_type', 'saturated')  # Default: saturated
        self.declare_parameter('target', [0.4, 1.0])  # Default target

        ########################################
        #   Subscriptions / Publishers         #
        ########################################
        # Subscribe to /odom topic in simulation
        # self.supscription = self.create_subscription(Odometry,'odom',self.odometry_callback, qos_profile=qos_profile_sensor_data)
        # Subscribe to /rigid_bodies topic in experiment
        self.subscription = self.create_subscription(RigidBodies, 'rigid_bodies', self.rigid_bodies_callback, 10)
        self.publisher = self.create_publisher(Twist,'cmd_vel', 10)
        self.timer = self.create_timer(self.Ts, self.control_callback)
        # Publishers for reference path and actual path for visualization in Rviz
        self.act_path_pub = self.create_publisher(Path, 'actual_path', 10)
        self.target_pub = self.create_publisher(Marker, 'visualization_target', 10)
        # Initialize Path messages
        self.actual_path = Path()
        self.actual_path.header.frame_id = "odom"

    ###########################################
    #               Callbacks                 #
    ###########################################

    def control_callback(self):
        if (self.x is None) or (self.y is None) or (self.theta is None):
            self.get_logger().warn("Waiting for ODOMETRY......")
            time.sleep(self.Ts)
            return
        if not(self.target_tracking_finish):
            # Get current states
            x = self.x
            y = self.y
            theta = self.theta
            # Linearized states of robot
            pose_robot = np.array([[x + self.b * np.cos(theta)], 
                                [y + self.b * np.sin(theta)]]) 
            # Publish actual path in Rviz for visualization
            self.publish_actual_path(pose_robot[0,0], pose_robot[1,0])
            # Get parameters: controller type and target
            controller_type = self.get_parameter('controller_type').value
            target_list = self.get_parameter('target').value
            # Convert target string to NumPy matrix
            target = np.array(target_list).reshape(-1,1)
            # Publish target in Rviz for visualization
            self.publish_target(target[0, 0], target[1, 0])
            # Compute error between current state and target
            error_to_target = np.linalg.norm(pose_robot - target)
            print("ERROR tracking: ", error_to_target)
            # if not(error_to_target <= self.d_error):
            if not(error_to_target <= self.d_error) or not(self.u_real[0,0] < 0.01):
                # Start time 
                tic = time.time()
                # Using chosen control method to compute u in integrator dynamics
                linear_input_u = self.get_control(controller_type, pose_robot, target)
                # End time 
                toc = time.time()
                # Transform u to control input (v and omega) in unicycle dynamics
                self.u_real = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta)/self.b, np.cos(theta)/self.b]]) @ linear_input_u
                # Processing linear velocity fo turtlebot
                self.u_real_safe[0,0] = np.min([self.u_real[0,0], (2.6 - self.u_real[1,0])/2.6 * 0.21])
                self.u_real_safe[1,0] = self.u_real[1,0]
                # Publish command for robots
                cmd_msg = Twist()
                cmd_msg.linear.x = self.u_real_safe[0, 0] # linear velocity
                cmd_msg.angular.z = self.u_real_safe[1, 0] # angular velocity
                print("Linear vel: ", self.u_real[0,0])
                print("Ang vel: ", self.u_real[1,0])
                self.publisher.publish(cmd_msg)
                # Computation time of controller
                self.comp_time = toc - tic
                self.total_comp_time += self.comp_time
                # Save data for plotting in Matplotlib
                self.num_tracking_steps +=1 # count number of tracking steps
                time_series = self.num_tracking_steps * self.Ts
                self.save_data_plot(time=time_series, state=pose_robot, target=target, linear_u=linear_input_u, u_real=self.u_real)
            else:
                self.target_tracking_finish = True  
                # Stop robot by setting linear and angular velocity to 0.0
                cmd_msg = Twist()
                self.u_real[0, 0] = 0.0
                self.u_real[1, 0] = 0.0
                cmd_msg.linear.x = self.u_real[0, 0]
                cmd_msg.angular.z = self.u_real[1, 0]
                self.publisher.publish(cmd_msg)
                print("Robot finished tracking")
                print("Number of tracking steps: ", self.num_tracking_steps)
                print("ERROR tracking: ", error_to_target)
                self.num_tracking_steps +=1 # count number of tracking steps
                self.avg_comp_time = self.total_comp_time / (self.num_tracking_steps - 1) # compute average computation time
                time_series = self.num_tracking_steps * self.Ts
                self.save_data_plot(time=time_series, state=pose_robot, target=target, linear_u=self.u_real, u_real=self.u_real)
                # Save to .mat file to plot in matlab
                self.save_data_export(controller_type, self.x_plot, self.y_plot, self.time_plot, self.lin_v_plot, self.ang_v_plot)
                print(f"Data saved for {controller_type} controller.")
                print(f"Average computation time of controller: {self.avg_comp_time}")
    # Function to get type of desired control   
    def get_control(self, controller_type, state, target):
        '''--- Select and apply the controller ---'''
        controllers = {
            'saturated': self.saturated_control,
            'lqr_lyapunov': self.lqr_lyapunov_control,
            'explicit_mpc': self.explicit_mpc,
            'implicit_mpc': self.implicit_mpc
        }

        if controller_type in controllers:
            return controllers[controller_type](state, target)
        else:
            self.get_logger().warn(f"Unknown controller type: {controller_type}, using 'saturated' as default")
            return self.saturated_control(state, target)
    #********************************
    #       SATURATED CONTROL       *    
    #********************************
    def saturated_control(self, state, target):

        '''----------- Get parameters ------------'''
        gamma = saturated_params["gamma"]

        '''------- Compute the saturated control input --------'''
        # Computing the control signal
        u = -gamma * self.B.T @ self.P_sat @ (state - target)
        # Using saturated function
        if np.all((self.U_input.A @ u) <= self.U_input.b):
            lamda = 1
        else:
            lamda_list = self.U_input.b / (self.U_input.A @ u)
            valid_lamda = lamda_list[(lamda_list >= 0)&(lamda_list<=1)]
            lamda = np.min(valid_lamda)
        print("Lamda: ", lamda)
        u_sat = lamda * u
        return u_sat
    #********************************
    #       LQR+LYAPUNOV CONTROL    *    
    #********************************
    def lqr_lyapunov_control(self, state, target):

        '''------ Get parameters --------------'''
        Q_lqr = lqr_lyapunov_params["Q"]
        R_lqr = lqr_lyapunov_params["R"]
        omega = lqr_lyapunov_params["omega"]
        
        # Find K by using ct package
        K,_,_ = ct.lqr(self.A, self.B, Q_lqr, R_lqr)
        
        '''------- Compute the LQR + Lyapunov control input --------'''
        # LQR control
        u_vir_lqr = -np.dot(K, (state - target))
        #+++ Lyapunouv based control +++
        # Define solver Casadi
        solver = cas.Opti()
        options = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 3000} # max_iter: 5000
        solver.solver('ipopt', options)
        # Define variables
        u_lya = solver.variable(self.nu, 1)
        # Define parameters
        u_lqr = solver.parameter(self.nu, 1)
        x_state = solver.parameter(self.nx, 1)
        # Define objective
        objective = cas.mtimes(cas.transpose(u_lqr - u_lya), (u_lqr - u_lya))
        # Define constraints
        solver.subject_to(cas.mtimes(cas.mtimes(2*cas.transpose(x_state), self.P_lya), u_lya) <= -omega*cas.mtimes(cas.mtimes(cas.transpose(x_state), self.P_lya), x_state))
        solver.subject_to(cas.mtimes(self.U_input.A, u_lya) <= self.U_input.b)
        # Minimize objective
        solver.minimize(objective)
        # Set value for parameters
        solver.set_value(u_lqr, u_vir_lqr)
        error = state - target
        solver.set_value(x_state, error)
        # Solve u 
        sol = solver.solve()
        u_lqr_lya = sol.value(u_lya)
        # Reshape u
        u_lqr_lya = u_lqr_lya.reshape(2,1)
        return u_lqr_lya
    
    #********************************
    #          ExMPC CONTROL        *    
    #********************************
    def explicit_mpc(self, state, target):
        '''----- By computing explicit MPC control input, 
        we import and use explicit solution function calculated by MPT3 in Matlab ----'''
        u = exmpc_solution(state - target)
        u_exmpc = u[:2]
        return u_exmpc

    #********************************
    #          iMPC CONTROL         *    
    #********************************
    def implicit_mpc(self, state, target):

        '''------ Get parameters --------------'''
        Q_mpc = implicit_mpc_params["Q"]
        R_mpc = implicit_mpc_params["R"]
        P_mpc = implicit_mpc_params["P"]
        Npred = implicit_mpc_params["Npred"]

        '''------- Compute the implicit MPC control input --------'''
        # Optimization problem using casadi
        solver_2 = cas.Opti()
        # Define variables
        x = solver_2.variable(self.nx, Npred+1)
        u = solver_2.variable(self.nu, Npred)
        xinit = solver_2.parameter(self.nx, 1)

        # Initialize constrains
        solver_2.subject_to(x[:,0] == xinit)
        for i in range(0, Npred):
            # Dynamic system
            solver_2.subject_to(x[:, i+1] == cas.mtimes(self.Ad, x[:, i]) + cas.mtimes(self.Bd, u[:, i]))
            # Constraint of u
            solver_2.subject_to(cas.mtimes(self.U_input.A, u[:, i]) <= self.U_input.b)
        # Constraint for terminal set
        solver_2.subject_to(cas.mtimes(self.Xf_A, (x[:, Npred] - target)) <= self.Xf_b)
        
        # Initialize objective
        objective = 0
        for i in range(0,Npred):
            objective = objective + cas.mtimes(cas.mtimes(cas.transpose(x[:,i] - target), Q_mpc), x[:,i] - target) +\
                                cas.mtimes(cas.mtimes(cas.transpose(u[:,i]), R_mpc), u[:,i])
        objective = objective + cas.mtimes(cas.mtimes(cas.transpose(x[:, Npred] - target), P_mpc), x[:, Npred] - target)
        ### Them cai constraint cua X_f
        
        solver_2.minimize(objective)
        
        # Define the solver
        options = {'ipopt': {'print_level': 0, 'sb': 'yes'},'print_time':0}
        solver_2.solver('ipopt', options)

        # Solve the problem
        solver_2.set_value(xinit, state)
        sol = solver_2.solve()
        usol = sol.value(u)
        u_impc = usol[:,0].reshape(2,1)
        return u_impc

    #############################################
    #           Other Functions                 #
    #############################################
    def compute_P_sat_lya(self, alpha):
        # Find P via Q by LMI
        Q = cp.Variable((self.nx, self.nx), symmetric = True)
        # LMI constraint
        lmi1 = Q @ self.A.T + self.A @ Q - 2 * self.B @ self.B.T + alpha * Q <= 0
        lmi2 = Q >= 0  
        # Combine the LMIs
        constraints = [lmi1, lmi2]
        # Define the optimization problem
        objective = cp.Minimize(0) # No objective, just looking for feasibility
        problem = cp.Problem(objective, constraints)
        # Solve the problem
        problem.solve(solver = cp.SCS, verbose = False)
        # Check the solution
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("Cannot find matrix Q")
        # Extract the solution
        Q = Q.value
        P = np.linalg.inv(Q)
        return P
    def quaternion2euler(self, qx, qy, qz, qw):
        # roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis rotation)
        sinp = math.sqrt(1 + 2 * (qw * qy - qx * qz))
        cosp = math.sqrt(1 - 2 * (qw * qy - qx * qz))
        pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2

        # yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        angles = np.array([roll, pitch, yaw]) # Yaw is the angle that we want
        return angles
    
    def odometry_callback(self, msg): 
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        q = orientation
        qw = q.w
        qx = q.x
        qy = q.y
        qz = q.z
        angles = self.quaternion2euler(qx, qy, qz, qw)
        theta = angles[-1]

        # Wrapping theta to 2pi
        self.theta = theta%(2*math.pi)
        self.x = position.x
        self.y = position.y
        # self.get_logger().info("x: {}, y: {}, theta: {}".format(self.x, self.y, self.theta))

    def rigid_bodies_callback(self, msg):
        x = msg.rigidbodies[1].pose.position.x
        y = msg.rigidbodies[1].pose.position.y
        z = msg.rigidbodies[1].pose.position.z

        qx = msg.rigidbodies[1].pose.orientation.x
        qy = msg.rigidbodies[1].pose.orientation.y
        qz = msg.rigidbodies[1].pose.orientation.z
        qw = msg.rigidbodies[1].pose.orientation.w

        angles = self.quaternion2euler(qx, qy, qz, qw)
        self.theta = angles[-1]%(2*math.pi)
        self.x = x
        self.y = y
        self.get_logger().info("x_0: {}, y_0: {}, theta_0: {}".format(self.x, self.y, self.theta))

    # Function to publish the target in Rviz
    def publish_target(self, x, y):
        marker = Marker()
        marker.header.frame_id = "odom"  # Change to your reference frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ref_point"
        marker.id = 0
        marker.type = Marker.SPHERE  # A dot
        marker.action = Marker.ADD

        # Set the position of the reference point
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set scale (controls the dot size)
        marker.scale.x = 0.2  # Adjust for visibility
        marker.scale.y = 0.2
        marker.scale.z = 0.1

        # Set color (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully visible

        self.target_pub.publish(marker)

    # Function to publish the actual trajectory in Rviz
    def publish_actual_path(self, x, y):
        self.actual_path.header.stamp = self.get_clock().now().to_msg()

        # Append the current position to the actual path
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0  # Default orientation
        self.actual_path.poses.append(pose)

        self.act_path_pub.publish(self.actual_path)

    def save_data_plot(self, time, state, target, linear_u, u_real):
        '''-------- Append value to plotting variables at each time step --------'''
        self.time_plot.append(time)
        self.x_plot.append(state[0,0])
        self.y_plot.append(state[1,0])
        self.target_plot = target
        self.u_1_plot.append(linear_u[0,0])
        self.u_2_plot.append(linear_u[1,0])
        self.lin_v_plot.append(u_real[0,0])
        self.ang_v_plot.append(u_real[1,0])
        '''------- Visualization in RVIZ --------'''
        self.publish_actual_path(state[0,0], state[1,0])
        self.publish_target(target[0,0], target[1, 0])

    def save_data_export(self, controller_type, x_plot, y_plot, time_plot, lin_v_plot, ang_v_plot):
        # Define the base directory for saving data
        base_dir = "/home/nguyehtt/turtlebot3_ws/src/target_tracking/data_plotting"
        
        # Ensure the directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Define filenames based on controller type
        x_pos_filename = f"{controller_type}_target_xPos.mat"
        y_pos_filename = f"{controller_type}_target_yPos.mat"
        lin_vel_filename = f"{controller_type}_target_linVel.mat"
        ang_vel_filename = f"{controller_type}_target_angVel.mat"
        
        # Save data
        scipy.io.savemat(os.path.join(base_dir, x_pos_filename), {'x': time_plot, 'y': x_plot})
        scipy.io.savemat(os.path.join(base_dir, y_pos_filename), {'x': time_plot, 'y': y_plot})
        scipy.io.savemat(os.path.join(base_dir, lin_vel_filename), {'x': time_plot, 'y': lin_v_plot})
        scipy.io.savemat(os.path.join(base_dir, ang_vel_filename), {'x': time_plot, 'y': ang_v_plot})

        

#################################################################
#                           Main loop                           #
#################################################################
def main(args=None):
    rclpy.init(args=args)
    node = TargetControl()
    try: 
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        plt.rcParams['font.family'] = 'DejaVu Serif'
        fig = plt.figure(figsize=(12, 10)) 
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[2, 2]) 
        
        
        # Plot position of robot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Position of robot')
        ax1.plot(node.x_plot, node.y_plot, 'b',label = "Real position of robot")
        ax1.plot(node.target_plot[0, 0], node.target_plot[1, 0], '*r', label = "Target of robot", markersize=12)
        # ax1.plot(node.target[0,0], node.target[1, 0], 'r', label = "Target of robot", markersize=12)
        ax1.set_xlabel('x (m)', fontsize=13)
        ax1.set_ylabel('y (m)', fontsize=13)
        ax1.axis('tight')
        ax1.legend().set_draggable(True)
        ax1.grid(True)

        # Plot linearized input of robot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Linearized input of robot')
        ax2.add_patch(node.Up)
        ax2.plot(node.u_1_plot, node.u_2_plot, 'm', label = "Trajectory of linearized input")
        ax2.set_xlabel('u1', fontsize = 13)
        ax2.set_ylabel('u2', fontsize = 13)
        ax2.axis('tight')
        ax2.legend().set_draggable(True)
        ax2.grid(True)
       
        # Plot linear velocity of robot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('Linear velocity of robot')
        ax3.plot(node.time_plot, node.lin_v_plot, 'b',lw = 0.6, label = 'Linear velocity')
        ax3.axhline(y=0.22, color='red', linestyle='--', label='Maximum translational velocity: 0.22 (m/s)')
        ax3.axhline(y=-0.22, color='red', linestyle='--', label='Minimum translational velocity: -0.22 (m/s)')
        ax3.set_xlabel('Time (s)', fontsize = 13)
        ax3.set_ylabel('Linear velocity (m/s)', fontsize = 13)
        ax3.axis('tight')
        ax3.legend().set_draggable(True)
        ax3.grid(True)
    
        # Plot angular velocity of robot
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Angular velocity of robot')
        ax4.plot(node.time_plot, node.ang_v_plot, 'k',lw = 0.6, label = 'Angular velocity')
        ax4.axhline(y=2.84, color='red', linestyle='--', label='Maximum rotational velocity: 2.84 (rad/s)')
        ax4.axhline(y=-2.84, color='red', linestyle='--', label='Minimum rotational velocity: -2.84 (rad/s)')
        ax4.set_xlabel('Time (s)', fontsize = 13)
        ax4.set_ylabel('Angular velocity (rad/s)', fontsize = 13)
        ax4.axis('tight')
        ax4.legend().set_draggable(True)
        ax4.grid(True)
    
        plt.tight_layout()
        plt.show()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()  
