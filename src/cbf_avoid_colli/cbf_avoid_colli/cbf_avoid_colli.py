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
sys.path.append('/home/nguyehtt/turtlebot3_ws/src')
# from utilities_formation import transformations
from utilities_formation.transformations import *
from utilities_formation.callback_states import *
from utilities_formation.graph import *
from utilities_formation.barrier_certificate import *

# Import Casadi for Lyapunouv-based Control
import casadi as cas 

class MyCBF(Node):
    def __init__(self):
        super().__init__("cbf_avoid_colli")
        # Parameters of system
        self.MAX_LIN_VEL = 0.21
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
        self.N = 4 # number of robots

        # Parameters of LQR controller
        # 75 and 3 : stable
        self.Q = np.diag(np.ones(self.nx) * 15)
        self.R = np.eye(self.nu) * 1
        self.K,self.P_lqr,_ = ct.lqr(self.A, self.B, self.Q, self.R)

        # Define safe distance
        self.safe_dis = 0.3
        self.gamma = 0.2  # rate of control barrier function #0.1 avoid from the large dis, #1 avoid from the small dis

        # Constraint set for u1, u2
        self.ru = min(self.b*self.MAX_ROT_VEL, self.MAX_LIN_VEL) 
        ptsU = []
        for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
            ptsU.append([self.ru * np.cos(tta), self.ru * np.sin(tta)])
        ptsU = np.array(ptsU)
        self.U_input = pc.qhull(ptsU) #checked
        self.Up = Polygon(ptsU, facecolor=(1, 1, 0, 0.1),edgecolor=(0, 0, 0, 0.8)) # for plotting U

        # Column ith is the target for robot i
        self.target = np.array([[2.0, 0.0, 0.0, 2.0],
                                [-1.0, 0.0, 2.0, 2.0]])   # simulation
        # self.target = np.array([[0.0, -0.67, -0.75],
        #                         [0.0, -0.72, 0.96]])   # experiment with 3 tb
        # self.target = np.array([[-0.9, -0.94],
        #                         [-0.94, 0.6]])   # experiment with 2 tb
        
        # Desired error tracking
        self.std_error = 0.005

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

        self.xplot_2 = []
        self.yplot_2 = []
        self.linvel_2 = []
        self.angvel_2 = []

        self.xplot_3 = []
        self.yplot_3 = []
        self.linvel_3 = []
        self.angvel_3 = []

        self.num_tracking_steps = 0
        self.time_plot = []

        #------------------------ Run simulation --------------------
        # Subscribe to the topic /odom (given by ROS) of 4 robots to receive their positions
        for i in range(self.N):
            self.robot_subscriptions[i] = self.create_subscription(
                Odometry,
                f'/turtlebot3{i}/odom',
                lambda msg, robot_id=i: self.odom_callback_wrapper(msg, robot_id),
                qos_profile=qos_profile_sensor_data
            )
        #----------------------- Run in experiment ------------------
        # Subscribe to the topic /rigid_bodies (given by Qualisys camera system) of 4 robots to receive their positions.
        # self.subscription = self.create_subscription(RigidBodies, 'rigid_bodies', self.rigid_bodies_callback, 10)

        # Create a publisher to publish the linear and angular velocities of the 4 robots in simulation
        # for i in range(self.N):
        #     self.robot_publisher[i] = self.create_publisher(
        #         Twist,
        #         f'/turtlebot3{i}/cmd_vel',
        #         10
        #     )

        # Create a publisher to publish the linear and angular velocities of the 4 robots in experiment
        self.cmd_msg = Twist()
        self.robot_publisher_0 = self.create_publisher(Twist,'/turtlebot30/cmd_vel', 10)
        self.robot_publisher_1 = self.create_publisher(Twist,'/turtlebot31/cmd_vel', 10)
        self.robot_publisher_2 = self.create_publisher(Twist,'/turtlebot32/cmd_vel', 10)
        self.robot_publisher_3 = self.create_publisher(Twist,'/turtlebot33/cmd_vel', 10)
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

        # Robot 2
        # x_2 = msg.rigidbodies[2].pose.position.x
        # y_2 = msg.rigidbodies[2].pose.position.y
        # z_2 = msg.rigidbodies[2].pose.position.z

        # qx_2 = msg.rigidbodies[2].pose.orientation.x
        # qy_2 = msg.rigidbodies[2].pose.orientation.y
        # qz_2 = msg.rigidbodies[2].pose.orientation.z
        # qw_2 = msg.rigidbodies[2].pose.orientation.w

        # angles_2 = quaternion2euler(qx_2, qy_2, qz_2, qw_2)
        # theta_2 = angles_2[-1]%(2*math.pi)
        # self.robot_positions[2] = (x_2, y_2, theta_2)
        # self.get_logger().info("x_2: {}, y_2: {}, theta_2: {}".format(self.robot_positions[2][0], self.robot_positions[2][1], self.robot_positions[2][2]))


        # Robot 3
        # x_3 = msg.rigidbodies[3].pose.position.x
        # y_3 = msg.rigidbodies[3].pose.position.y
        # z_3 = msg.rigidbodies[3].pose.position.z

        # qx_3 = msg.rigidbodies[3].pose.orientation.x
        # qy_3 = msg.rigidbodies[3].pose.orientation.y
        # qz_3 = msg.rigidbodies[3].pose.orientation.z
        # qw_3 = msg.rigidbodies[3].pose.orientation.w

        # angles_3 = quaternion2euler(qx_3, qy_3, qz_3, qw_3)
        # self.theta_3 = angles_3[-1]%(2*math.pi)
        # theta_3 = angles_3[-1]%(2*math.pi)
        # self.x_3 = x_3
        # self.y_3 = y_3
        # self.robot_positions[3][0] = x_3
        # self.robot_positions[3][1] = y_3
        # self.robot_positions[3][2] = theta_3
        # self.robot_positions[3] = (x_3, y_3, theta_3)
        # self.get_logger().info("x_3: {}, y_3: {}, theta_3: {}".format(self.robot_positions[3][0], self.robot_positions[3][1], self.robot_positions[3][2]))

    def publish_command(self, robot_id, linear, angular):
        # Publishing command
        self.cmd_msg.linear.x = linear
        self.cmd_msg.linear.y = 0.0
        self.cmd_msg.linear.z = 0.0

        self.cmd_msg.angular.x = 0.0
        self.cmd_msg.angular.y = 0.0
        self.cmd_msg.angular.z = angular

        # Publishing to the corresponding robot
        getattr(self, f'robot_publisher_{robot_id}').publish(self.cmd_msg)
    # Các hàm publish_command_0, publish_command_1, ... giờ chỉ cần gọi hàm chung này:
    def publish_command_0(self, linear, angular):
        self.publish_command(0, linear, angular)

    def publish_command_1(self, linear, angular):
        self.publish_command(1, linear, angular)

    def publish_command_2(self, linear, angular):
        self.publish_command(2, linear, angular)

    def publish_command_3(self, linear, angular):
        self.publish_command(3, linear, angular)
        
    def barrier_func(self, u_d, poses_robot):
        solver = cas.Opti()
        options = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 3000} # max_iter: 5000
        solver.solver('ipopt', options)
        # Define variables
        u_safe = solver.variable(self.nu, self.N)
        # Define parameters
        u_unsafe = solver.parameter(self.nu, self.N)
        z_robot = solver.parameter(self.nx, self.N)
        objective = 0
        for i in range(self.N):
            objective += cas.norm_2(u_safe[:, i] - u_unsafe[:, i])**2
            solver.subject_to(cas.mtimes(self.U_input.A, u_safe[:, i]) <= self.U_input.b)
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                h_br = cas.norm_2(z_robot[:, i] - z_robot[:, j])**2 - self.safe_dis**2
                grad_h_br = cas.vertcat(2*(z_robot[0, i] - z_robot[0, j]), 
                                        2*(z_robot[1, i] - z_robot[1, j]), 
                                        -2*(z_robot[0, i] - z_robot[0, j]),
                                        -2*(z_robot[1, i] - z_robot[1, j]))
                u_safe_ij = cas.vertcat(u_safe[:, i], u_safe[:, j])
                solver.subject_to(cas.mtimes(grad_h_br.T, u_safe_ij) >= -self.gamma * h_br)
        # Minimize objective
        solver.set_value(u_unsafe, u_d)
        solver.set_value(z_robot, poses_robot)
        solver.minimize(objective)
        sol = solver.solve()
        u_br_safe = sol.value(u_safe)
        return u_br_safe
    
    def save_data_export(self, robot_idx, x_plot, y_plot, time_plot, lin_v_plot, ang_v_plot):
        # Define the base directory for saving data
        base_dir = "/home/nguyehtt/turtlebot3_ws/src/cbf_avoid_colli/data_plotting"
        
        # Ensure the directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Define filenames based on controller type
        x_pos_filename = f"tb_{robot_idx}_xPos_cbf.mat"
        y_pos_filename = f"tb_{robot_idx}_yPos_cbf.mat"
        position_filename = f"tb_{robot_idx}_Position_cbf.mat"
        lin_vel_filename = f"tb_{robot_idx}_linVel_cbf.mat"
        ang_vel_filename = f"tb_{robot_idx}_angVel_cbf.mat"
        
        # Save data
        scipy.io.savemat(os.path.join(base_dir, x_pos_filename), {'x': time_plot, 'y': x_plot})
        scipy.io.savemat(os.path.join(base_dir, y_pos_filename), {'x': time_plot, 'y': y_plot})
        scipy.io.savemat(os.path.join(base_dir, position_filename), {'x': x_plot, 'y': y_plot})
        scipy.io.savemat(os.path.join(base_dir, lin_vel_filename), {'x': time_plot, 'y': lin_v_plot})
        scipy.io.savemat(os.path.join(base_dir, ang_vel_filename), {'x': time_plot, 'y': ang_v_plot})


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
        u_br = self.barrier_func(u_lqr, poses_si)
        toc = time.time()
        denta_time = toc - tic
        print(f"Computation time: {denta_time}")
        # For plotting
        self.num_tracking_steps += 1
        time_series = self.num_tracking_steps * self.Ts
        self.time_plot.append(time_series)
        self.xplot_0.append(poses[0, 0])
        self.yplot_0.append(poses[1, 0])

        self.xplot_1.append(poses[0, 1])
        self.yplot_1.append(poses[1, 1])

        self.xplot_2.append(poses[0, 2])
        self.yplot_2.append(poses[1, 2])

        self.xplot_3.append(poses[0, 3])
        self.yplot_3.append(poses[1, 3])


        ######## Edit publisher according to index of tb in experiment #######
        for i in range(self.N):
            self.u_real[:, i] = np.array([[np.cos(poses[2, i]), np.sin(poses[2, i])],
                               [-np.sin(poses[2, i])/self.b, np.cos(poses[2, i])/self.b]]) @ u_br[:, i]
            # Processing
            self.u_real_safe[0, i] = np.min([self.u_real[0,i], (2.6 - self.u_real[1,i])/2.6 * 0.21])
            self.u_real_safe[1, i] = self.u_real[1, i]

        # Check whether robot reach target or not
        norm_error = np.linalg.norm(error, axis=0)
        if np.all(norm_error < self.std_error):
            self.get_logger().info("All robots reached their targets. Stopping...")
            self.publish_command_0(linear=0.0, angular = 0.0)
            self.publish_command_1(linear=0.0, angular = 0.0)
            self.publish_command_2(linear=0.0, angular = 0.0)
            self.publish_command_3(linear=0.0, angular = 0.0)
            # For plotting
            self.linvel_0.append(0.0)
            self.angvel_0.append(0.0)

            self.linvel_1.append(0.0)
            self.angvel_1.append(0.0)

            self.linvel_2.append(0.0)
            self.angvel_2.append(0.0)

            self.linvel_3.append(0.0)
            self.angvel_3.append(0.0)

            self.save_data_export(0, self.xplot_0, self.yplot_0, self.time_plot, self.linvel_0, self.angvel_0)
            self.save_data_export(1, self.xplot_1, self.yplot_1, self.time_plot, self.linvel_1, self.angvel_1)
            self.save_data_export(2, self.xplot_2, self.yplot_2, self.time_plot, self.linvel_2, self.angvel_2)
            self.save_data_export(3, self.xplot_3, self.yplot_3, self.time_plot, self.linvel_3, self.angvel_3)
            self.timer.cancel()
            return

        # cmd_msg = Twist()
        # # Publish command for [tb3 - index 0] in u_real
        # cmd_msg.linear.x = self.u_real[0, 0]
        # cmd_msg.angular.z = self.u_real[1, 0]
        # self.robot_publisher_3.publish(cmd_msg)

        # # Publish command for [tb0 - index 1] in u_real
        # cmd_msg.linear.x = self.u_real[0, 1]
        # cmd_msg.angular.z = self.u_real[1, 1]
        # self.robot_publisher_0.publish(cmd_msg)
       
        # For experiment with 2 tb
        # self.publish_command_0(linear=self.u_real_safe[0, 1], angular = self.u_real_safe[1, 1])
        # self.publish_command_3(linear=self.u_real_safe[0, 0], angular = self.u_real_safe[1, 0])

        # For simulation with 4 tb
        self.publish_command_0(linear=self.u_real[0, 0], angular = self.u_real[1, 0])
        self.publish_command_1(linear=self.u_real[0, 1], angular = self.u_real[1, 1])
        self.publish_command_2(linear=self.u_real[0, 2], angular = self.u_real[1, 2])
        self.publish_command_3(linear=self.u_real[0, 3], angular = self.u_real[1, 3])

        
        ######### Publish command in simulation
        # for i in range(self.N):
        #     self.u_real[:, i] = np.array([[np.cos(poses[2, i]), np.sin(poses[2, i])],
        #                        [-np.sin(poses[2, i])/self.b, np.cos(poses[2, i])/self.b]]) @ u_br[:, i]
            
        #     # Publish command for robots
        #     cmd_msg = Twist()
        #     cmd_msg.linear.x = self.u_real[0, i]
        #     cmd_msg.angular.z = self.u_real[1, i]
        #     self.robot_publisher[i].publish(cmd_msg)

        # For plotting
        self.linvel_0.append(self.u_real_safe[0, 1])
        self.angvel_0.append(self.u_real_safe[1, 1])

        self.linvel_1.append(self.u_real_safe[0, 1])
        self.angvel_1.append(self.u_real_safe[1, 1])

        self.linvel_2.append(self.u_real_safe[0, 2])
        self.angvel_2.append(self.u_real_safe[1, 2])

        self.linvel_3.append(self.u_real_safe[0, 3])
        self.angvel_3.append(self.u_real_safe[1, 3])
        print("u_real: ", self.u_real)

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