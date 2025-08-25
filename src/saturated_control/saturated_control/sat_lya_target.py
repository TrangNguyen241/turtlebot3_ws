#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import cvxpy as cp
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from mocap4r2_msgs.msg import RigidBodies

import time
import sys
import math
import polytope as pc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Messages for visualization in Rviz
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Import necessary functions
import sys
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/saturated_control')
from utilities.checkPolytope import *
from utilities.compute_bspline_curve import *
from utilities.compute_reference import *

# Import Casadi for Lyapunouv-based Control
import casadi as cas 


class MySatLya(Node):
    def __init__(self):
        super().__init__("sat_lya_target")
        
        # Parameters of system
        self.MAX_LIN_VEL = 0.22
        self.MIN_LIN_VEL = -self.MAX_LIN_VEL
        self.MAX_ROT_VEL = 2.84
        self.MIN_ROT_VEL = -self.MAX_ROT_VEL
        self.b = 0.1
        # self.Ts = 0.02 #target tracking
        self.Ts = 0.1 #trajectory tracking
        # COntinuous time
        A = np.zeros((2,2))
        self.B = np.eye(2)
        # Discrete time
        # A = np.eye(2)
        # self.B = np.eye(2)*self.Ts
        self.nx = A.shape[1]
        self.nu = A.shape[1]
        C = np.eye(2)
        # Find Q by LMI
        # alpha = 1.0 #target tracking
        # self.gamma = 1 #target tracking
        alpha = 1.0 #target tracking # default: 1
        self.gamma = 1.5 #target tracking # default: 1
        self.omega = 0.1 #lyapunouv tuning (0.4 => ref:[0.4, 1] || 0.1 =>ref:[1, 2])
        Q = cp.Variable((self.nx, self.nx), symmetric = True)
        # LMI constraint
        lmi1 = Q @ A.T + A @ Q - 2 * self.B @ self.B.T + alpha * Q <= 0
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
        self.P = np.linalg.inv(Q)
        # Use P calculated by Matlab
        # self.P = np.array([[0.1051, 0],[0, 0.1051]])
        # Constraint set for u1, u2
        self.ru = min(self.b*self.MAX_ROT_VEL, self.MAX_LIN_VEL) 
        ptsU = []
        for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
            ptsU.append([self.ru * np.cos(tta), self.ru * np.sin(tta)])
        ptsU = np.array(ptsU)
        self.U_input = pc.qhull(ptsU) #checked
        # Target point
        self.target_point = np.array([[1.0],[2.0]])
        # self.target_point = np.array([[0.4],[1]])
        self.std_error = 0.005
        # States of robot
        self.x = None
        self.y = None
        self.theta = None

        # Variables to create B-spline curve
        self.way_points = np.array([[0.1, 0.3, 1, 1.6, 1.9, 1, 0.2, 2],
                                [0, 1.8, 1.4, 1.8, 1, 0.8, 0.5, 0.1]])
        self.num_of_control_points = 10
        self.k = 4 # bac cua duong bspline
        self.flag_add_start_point = 0
        self.xref = [] # bien de luu path theo x, y
        self.yref = []
        self.iref = 0

        # For plotting positions of robot
        self.x_plot = []
        self.y_plot = []
        # For plotting velocity of robot
        self.lin_v_plot = []
        self.time = []
        self.ang_v_plot = []
        self.u_1_plot = []
        self.u_2_plot = []
        # For plotting u
        self.u1_ref_plot = []
        self.u2_ref_plot = []
        self.u1_vir = []
        self.u2_vir = []
        self.u1_vir_ref = []
        self.u2_vir_ref = []
        self.u1_sat = []
        self.u2_sat = []
        # For plotting Lyapunouv energy
        self.lya_energy_plot = []
        self.lya_limit_plot = []
        self.lya_limit_plot_2 = []
        # Import trajectory
        self.ref_traj, self.u_ref_traj = compute_reference_trajectory(self.num_of_control_points, self.way_points, self.k)
        self.imax = self.ref_traj.shape[1]
        self.target_tracking_finish = False
        self.trajectory_tracking_finish = False
        self.counter = 0
        self.compute_time = 0
        # Import U_e set
        # Load data from matlab 
        Ue_data = scipy.io.loadmat('/home/nguyehtt/turtlebot3_ws/src/saturated_control/utilities/U_e_input_python.mat')
        self.A_Ue = Ue_data['A_Ue_input']
        self.b_Ue = Ue_data['b_Ue_input']
        Ue_input_hull = pc.Polytope(self.A_Ue, self.b_Ue)

        # Subscribe and publish
        self.supscription = self.create_subscription(Odometry,'odom',self.odometry_callback, qos_profile=qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist,'cmd_vel', 10)
        self.timer = self.create_timer(self.Ts, self.saturated_lya_target)
        # Publishers for reference path and actual path
        self.ref_path_pub = self.create_publisher(Path, '/reference_path', 10)
        self.act_path_pub = self.create_publisher(Path, '/actual_path', 10)
        self.target_pub = self.create_publisher(Marker, 'visualization_target', 10)
        # Initialize Path messages
        self.reference_path = Path()
        self.reference_path.header.frame_id = "odom"  # Adjust frame_id as needed

        self.actual_path = Path()
        self.actual_path.header.frame_id = "odom"

    # Function to publish the reference trajectory
    def publish_reference_path(self, xref, yref):
        self.reference_path.header.stamp = self.get_clock().now().to_msg()
        self.reference_path.poses = []  # Reset poses

        for x, y in zip(xref, yref):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "odom"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Default orientation
            self.reference_path.poses.append(pose)

        self.ref_path_pub.publish(self.reference_path)

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

    
    # Function to publish the actual trajectory
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
    def rigid_bodies_callback(self, msg):
        x = msg.rigidbodies[0].pose.position.x
        y = msg.rigidbodies[0].pose.position.y
        z = msg.rigidbodies[0].pose.position.z

        qx = msg.rigidbodies[0].pose.orientation.x
        qy = msg.rigidbodies[0].pose.orientation.y
        qz = msg.rigidbodies[0].pose.orientation.z
        qw = msg.rigidbodies[0].pose.orientation.w

        angles = self.quaternion2euler(qx, qy, qz, qw)
        self.theta = angles[-1]%(2*math.pi)
        self.x = x
        self.y = y
        #  self.get_logger().info("x_0: {}, y_0: {}, theta_0: {}".format(self.x_0, self.y_0, self.theta_0))
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

    ########### Saturated control for robot tracking a target ##############
    def saturated_lya_target(self):
        if (self.x is None) or (self.y is None) or (self.theta is None):
            self.get_logger().warn("Waiting for ODOMETRY......")
            time.sleep(self.Ts)
            return
        if not(self.target_tracking_finish):
            # Linearized states of robot
            x = self.x
            y = self.y
            theta = self.theta
            pose_robot = np.array([[x + self.b * np.cos(theta)], 
                                [y + self.b * np.sin(theta)]])
            self.x_plot.append(pose_robot[0,0])
            self.y_plot.append(pose_robot[1,0])
            self.publish_actual_path(pose_robot[0,0], pose_robot[1,0])
            # Transformation of target point 
            self.ref_point = np.array([[self.target_point[0, 0]],
                                [self.target_point[1, 0]]])
            self.publish_target(self.target_point[0, 0], self.target_point[1, 0])
            error_to_target = np.linalg.norm(pose_robot - self.ref_point)
            error = pose_robot - self.ref_point

            # For Lyapunouv limit
            pose_initial = np.array([[0.0 + self.b * np.cos(theta)], 
                                [0.0 + self.b * np.sin(theta)]])   
            lya_0 = ((pose_initial - self.ref_point).T @ self.P @ (pose_initial - self.ref_point)).flatten()[0]

            # Optimization problem (Lyapunouv based control)
            # Define solver Casadi
            solver = cas.Opti()
            options = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 5000}
            solver.solver('ipopt', options)
            # Define variables
            u_lya = solver.variable(self.nu, 1)
            # Define parameters
            u_sat = solver.parameter(self.nu, 1)
            x_state = solver.parameter(self.nx, 1)
            # Define objective
            objective = cas.mtimes(cas.transpose(u_sat - u_lya), (u_sat - u_lya))
            # Define constraints
            # solver.subject_to(cas.mtimes(cas.mtimes(cas.mtimes(cas.transpose(u_lya), self.P), x_state) +\
            #                   cas.mtimes(cas.mtimes(cas.transpose(x_state), self.P), u_lya), u_lya) <= -self.omega*cas.mtimes(cas.mtimes(cas.transpose(x_state), self.P), x_state))
            solver.subject_to(cas.mtimes(cas.mtimes(2*cas.transpose(x_state), self.P), u_lya) <= -self.omega*cas.mtimes(cas.mtimes(cas.transpose(x_state), self.P), x_state)) 
            solver.subject_to(cas.mtimes(cas.transpose(u_lya), u_lya) <= self.ru**2)
            solver.minimize(objective)

            if (error_to_target > self.std_error):
                start = time.time()
                # Computing the control signal
                u_vir = -self.gamma * self.B.T @ self.P @ (pose_robot - self.ref_point)
                if check_u_inputs(self.ru, u_vir):
                    lamda = 1
                else:
                    # Using U constraints as Polytope
                    # lamda_list = self.U_input.b / (self.U_input.A @ u_vir)
                    # valid_lamda = lamda_list[(lamda_list >= 0)&(lamda_list<=1)]
                    # lamda_min = np.min(valid_lamda)
                    # lamda = lamda_min
                    # Using U constraints as ru
                    lamda = self.ru / np.linalg.norm(u_vir)
                u_vir_sat = lamda * u_vir
                # Lyapunouv based control
                solver.set_value(u_sat, u_vir_sat)
                solver.set_value(x_state, error)
                sol = solver.solve()
                u_sat_lya = sol.value(u_lya)
                # For debug casadi
                print("u_lya cua casadi: ", solver.debug.value(u_lya))
                u_sat_lya = u_sat_lya.reshape(2,1)
                self.u_1_plot.append(u_sat_lya[0, 0])
                self.u_2_plot.append(u_sat_lya[1, 0])
                end = time.time()
                self.u_real = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta)/self.b, np.cos(theta)/self.b]]) @ u_sat_lya
                # u_real[0, 0] = np.clip(u_real[0, 0], self.MIN_LIN_VEL, self.MAX_LIN_VEL)
                # u_real[1, 0] = np.clip(u_real[1, 0], self.MIN_ROT_VEL, self.MAX_ROT_VEL)
                # Publish command for robots
                cmd_msg = Twist()
                cmd_msg.linear.x = self.u_real[0, 0]
                cmd_msg.angular.z = self.u_real[1, 0]
                self.publisher.publish(cmd_msg)
                self.lin_v_plot.append(self.u_real[0, 0])
                self.ang_v_plot.append(self.u_real[1, 0])
                self.counter +=1
                denta_time_compute = end - start
                self.compute_time += denta_time_compute
                print(f'Computation time for one loop: {denta_time_compute} (s)')
                time_series = self.counter *self.Ts
                self.time.append(time_series)
                self.get_logger().info("lin vel: {}, ang vel: {}".format(self.u_real[0, 0], self.u_real[1, 0]))     
                # For plotting ENERGY DECREASE (LYAPUNOUV DECREASE)
                lya_energy = (error.T @ self.P @ error).flatten()[0] 
                lya_limit = lya_0 * np.exp(-self.omega * time_series)
                lya_limit_2 = lya_0 * np.exp(-1.0 * time_series)
                self.lya_energy_plot.append(lya_energy)   
                self.lya_limit_plot.append(lya_limit)
                self.lya_limit_plot_2.append(lya_limit_2)
            else:
                self.target_tracking_finish = True  
                # Stop robot
                cmd_msg = Twist()
                self.u_real[0, 0] = 0.0
                self.u_real[1, 0] = 0.0
                cmd_msg.linear.x = self.u_real[0, 0]
                cmd_msg.angular.z = self.u_real[1, 0]
                self.publisher.publish(cmd_msg)
                self.lin_v_plot.append(self.u_real[0, 0])
                self.ang_v_plot.append(self.u_real[1, 0])
                self.counter +=1
                time_series = self.counter *self.Ts
                self.time.append(time_series)
                print("Robot reaches target!!!")
                print("Computation time in total: ", self.compute_time)
                self.get_logger().info("Number of target tracking steps: {}".format(self.counter))
                lya_energy = (error.T @ self.P @ error).flatten()[0]
                lya_limit = lya_0 * np.exp(-self.omega * time_series)
                lya_limit_2 = lya_0 * np.exp(-1.0 * time_series)
                self.lya_energy_plot.append(lya_energy)
                self.lya_limit_plot.append(lya_limit)
                self.lya_limit_plot_2.append(lya_limit_2)
                # Save to .mat file to plot in matlab
                # scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/saturated_control/saturated_control/lya_sat_target_pos.mat', {'x': self.x_plot, 'y': self.y_plot})
                # scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/saturated_control/saturated_control/lya_sat_target_lin_vel.mat', {'x': self.time, 'y': self.lin_v_plot})
                # scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/saturated_control/saturated_control/lya_sat_target_ang_vel.mat', {'x': self.time, 'y': self.ang_v_plot})
                # scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/saturated_control/saturated_control/lya_omg_0.05.mat', {'x': self.time, 'y': self.lya_energy_plot})

                print("Data saving")
      

def main(args = None):
    rclpy.init(args=args)
    node = MySatLya()
    try: 
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        plt.rcParams['font.family'] = 'DejaVu Serif'
        fig1 = plt.figure(figsize=(12, 10)) 
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[2, 2]) 
        
        
        # Plot position of robot
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.set_title('Position of robot')
        ax1.plot(node.x_plot, node.y_plot, 'b',label = "Real position of robot")
        ax1.plot(node.ref_point[0, 0], node.ref_point[1, 0], '*r', label = "Target of robot", markersize=12) # For target tracking
        # ax1.plot(node.xref, node.yref, 'r', label = "Reference trajectory", markersize=12)  # For trajectory tracking
        ax1.set_xlabel('x (m)', fontsize=13)
        ax1.set_ylabel('y (m)', fontsize=13)
        ax1.axis('tight')
        ax1.legend().set_draggable(True)
        ax1.grid(True)

        # Plot linearized input of robot
        # Draw circle of constraint set
        theta = np.linspace(0, 2 * np.pi, 100)  
        x = node.ru * np.cos(theta) 
        y = node.ru * np.sin(theta)  
        
        ax2 = fig1.add_subplot(gs[0, 1])
        ax2.set_title('Linearized input of robot')
        ax2.plot(x, y, label = 'Input constraint set')
        ax2.plot(node.u_1_plot, node.u_2_plot, 'm', label = "Trajectory of linearized input")
        ax2.set_xlabel('u1', fontsize = 13)
        ax2.set_ylabel('u2', fontsize = 13)
        ax2.axis('tight')
        ax2.legend().set_draggable(True)
        ax2.grid(True)
       
        # Plot linear velocity of robot
        ax3 = fig1.add_subplot(gs[1, 0])
        ax3.set_title('Linear velocity of robot')
        ax3.plot(node.time, node.lin_v_plot, 'b',lw = 0.6, label = 'Linear velocity')
        ax3.axhline(y=0.22, color='red', linestyle='--', label='Maximum translational velocity: 0.22 (m/s)')
        ax3.axhline(y=-0.22, color='red', linestyle='--', label='Minimum translational velocity: -0.22 (m/s)')
        ax3.set_xlabel('Time (s)', fontsize = 13)
        ax3.set_ylabel('Linear velocity (m/s)', fontsize = 13)
        ax3.axis('tight')
        ax3.legend().set_draggable(True)
        ax3.grid(True)
    
        # Plot angular velocity of robot
        ax4 = fig1.add_subplot(gs[1, 1])
        ax4.set_title('Angular velocity of robot')
        ax4.plot(node.time, node.ang_v_plot, 'k',lw = 0.6, label = 'Angular velocity')
        ax4.axhline(y=2.84, color='red', linestyle='--', label='Maximum rotational velocity: 2.84 (rad/s)')
        ax4.axhline(y=-2.84, color='red', linestyle='--', label='Minimum rotational velocity: -2.84 (rad/s)')
        ax4.set_xlabel('Time (s)', fontsize = 13)
        ax4.set_ylabel('Angular velocity (rad/s)', fontsize = 13)
        ax4.axis('tight')
        ax4.legend().set_draggable(True)
        ax4.grid(True)

        # Plotting Lyapunouv
        plt.figure(figsize=(12, 10)) 
        plt.title('Lyapunouv of robot')
        plt.plot(node.time, node.lya_energy_plot, 'b', lw = 0.6, label = 'Lyapunouv')
        plt.plot(node.time, node.lya_limit_plot, 'r', lw = 0.6, label = 'Lyapunouv limit omega = 0.01')
        plt.plot(node.time, node.lya_limit_plot_2, 'g', lw = 0.6, label = 'Lyapunouv limit omega = 1')
        plt.xlabel('Time (s)', fontsize = 13)
        plt.ylabel('V(x)', fontsize = 13)
        plt.legend().set_draggable(True)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    node.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()