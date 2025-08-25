#!/usr/bin/env python3

# ROS
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

# Topic
from thao_trang_interfaces.msg import Tuple, TupleList, TupleListArray
from nav_msgs.msg import Odometry
from mocap4r2_msgs.msg import RigidBodies
from geometry_msgs.msg import Twist

# Basic python packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import math
import casadi as cas 
import polytope as pc
import scipy.io
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

# Import necessary functions
import sys
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/exp_mpc')
from utilities.compute_reference import *


class MyImpMPC(Node):
#### init ####
    def __init__(self):
        super().__init__('imp_mpc_target')

#### define variable ####

        # Limits of the Turtlebot3: Burger
        self.MAX_LIN_VEL = 0.22
        self.MIN_LIN_VEL = -self.MAX_LIN_VEL
        self.MAX_ROT_VEL = 2.84
        self.MIN_ROT_VEL = -self.MAX_ROT_VEL

        # Sampling time
        self.Ts = 0.1 #0.1
        self.b = 0.1

        # Variables to store commands
        self.command_linear_speed = 0.00
        self.command_angular_speed = 0.00

        # Robot's position updated by odometry_callback or rigid_bodies_callback
        self.x = None
        self.y = None
        self.theta = None

        # Initialize vectors for plot and data saving
        # For plotting positions of robot
        self.x_plot = []
        self.y_plot = []
        # For plotting velocity of robot
        self.time = []
        self.lin_v_plot = []
        self.ang_v_plot = []
        self.u_1_plot = []
        self.u_2_plot = []

        self.counter = 0 # counter for computation steps
        self.compute_time = 0

        # states of system
        self.state_0 = None

        # Maximum errors
        self.std_error = 0.005
        self.total_error_tracking = 0

        # input
        self.u0 = np.array([[0],[0]])

        # Reference 
        # self.target = np.array([[0.2],
        #                         [0.4]])
        self.target = np.array([[0.4],
                                [1.0]])
        self.actual_poses = np.zeros((2,1100))
        
        # Variables to create B-spline curve
        self.way_points = np.array([[0.1, 0.3, 1, 1.6, 1.9, 1, 0.2, 2],
                                [0, 1.8, 1.4, 1.8, 1, 0.8, 0.5, 0.1]])
        self.num_of_control_points = 10
        self.k = 4 # bac cua duong bspline
        self.flag_add_start_point = 0
        self.xref = [] # bien de luu path theo x, y
        self.yref = []
        self.iref = 0
        
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

        # Q, R, P parameters
        self.A = np.zeros(2)
        self.B = np.eye(2)
        self.Ad = np.eye(2) + self.A * self.Ts
        self.Bd = self.B * self.Ts
        # self.Q = np.diag([0.5, 0.5])
        # self.R = np.diag([1, 1])
        self.Q = np.diag([1.5, 1.5])
        self.R = np.diag([30, 30])
        self.P = np.diag([67.8362318214405, 67.8362318214405])

        # Constraint set for u1, u2
        self.ru = min(self.b*self.MAX_ROT_VEL, self.MAX_LIN_VEL) 
        ptsU = []
        for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
            ptsU.append([self.ru * np.cos(tta), self.ru * np.sin(tta)])
        ptsU = np.array(ptsU)
        self.U_input = pc.qhull(ptsU) #checked

        self.target_tracking_finish = False

        # create velocity publisher
        self.publish_vel = self.create_publisher(Twist,'cmd_vel',10)
        # create odometry subscriber
        # self.sub_odom = self.create_subscription(Odometry,'odom',self.odometry_callback, qos_profile=qos_profile_sensor_data)
        self.subscription = self.create_subscription(RigidBodies, 'rigid_bodies', self.rigid_bodies_callback, 10)
        # Publishers for reference path and actual path to visualize in Rviz
        self.ref_path_pub = self.create_publisher(Path, '/reference_path', 10)
        self.act_path_pub = self.create_publisher(Path, '/actual_path', 10)
        # Initialize Path messages
        self.reference_path = Path()
        self.reference_path.header.frame_id = "odom"  # Adjust frame_id as needed

        self.actual_path = Path()
        self.actual_path.header.frame_id = "odom"
        # Initialize timer for controller
        self.timer = self.create_timer(self.Ts, self.imp_mpc_trajectory)
  
#### CALLBACKS ####
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

    def rigid_bodies_callback(self, msg):
            # print(f"{msg}")
        x=msg.rigidbodies[2].pose.position.x
        y=msg.rigidbodies[2].pose.position.y
        z=msg.rigidbodies[2].pose.position.z

        qx = msg.rigidbodies[2].pose.orientation.x
        qy = msg.rigidbodies[2].pose.orientation.y
        qz = msg.rigidbodies[2].pose.orientation.z
        qw = msg.rigidbodies[2].pose.orientation.w

        angles = self.quaternion2euler(qx, qy, qz, qw)
        self.theta = angles[-1]%(2*math.pi)
        self.x = x
        self.y = y

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
    
    def pub_vel(self, command_linear_speed, command_angular_speed):
        msg = Twist()
        msg.linear.x = command_linear_speed
        msg.angular.z = command_angular_speed
        self.publish_vel.publish(msg)

    #### MPC Controller ####
    def imp_mpc_trajectory(self):
        if (self.x is None) or (self.y is None) or (self.theta is None):
            self.get_logger().warn("Waiting for ODOMETRY......")
            time.sleep(self.Ts)
            return
        if not(self.trajectory_tracking_finish):
            if self.iref < self.imax:
                #### initializing start position ####     
                x = self.x
                y = self.y
                theta  = self.theta
                pose_robot  = np.array([[x + self.b*np.cos(theta)],
                                    [y + self.b*np.sin(theta)]])  
                self.x_plot.append(pose_robot[0, 0])
                self.y_plot.append(pose_robot[1, 0]) 
                # Get reference 
                ref_point = self.ref_traj[:, self.iref].reshape(2,1)
                self.xref.append(ref_point[0])
                self.yref.append(ref_point[1])
                # Publishing reference path to Rviz (visualization)
                self.publish_reference_path(self.ref_traj[0, :], self.ref_traj[1, :])
                # Publishing positions of robot to Rviz
                self.publish_actual_path(pose_robot[0, 0], pose_robot[1, 0])
                self.actual_poses[:, self.iref] = pose_robot.reshape(2,)
                error_to_target = np.linalg.norm(pose_robot - ref_point)

                self.total_error_tracking += error_to_target
                self.get_logger().info("Error_to_target: {}".format(error_to_target))
                Npred = 5  # number of predictions
                start = time.time()
                    
                #### model dimension ####
                dx, du = np.shape(self.Ad)
                # Optimization problem using casadi
                solver_2 = cas.Opti()
                # define variables
                x = solver_2.variable(dx, Npred+1)
                u = solver_2.variable(du, Npred)
                xinit = solver_2.parameter(dx, 1)

                # initialize constrains
                solver_2.subject_to(x[:,0] == xinit)
                for i in range(0, Npred):
                    # dynamic
                    solver_2.subject_to(x[:, i+1] == cas.mtimes(self.Ad, x[:, i]) + cas.mtimes(self.Bd, u[:, i]))
                    solver_2.subject_to(cas.mtimes(self.A_Ue, u[:, i]) <= self.b_Ue)
                
                # Initialize objective
                objective = 0
                for i in range(0,Npred):
                    objective = objective + cas.mtimes(cas.mtimes(cas.transpose(x[:,i] - ref_point), self.Q), x[:,i] - ref_point) +\
                                        cas.mtimes(cas.mtimes(cas.transpose(u[:,i]), self.R), u[:,i])
                objective = objective + cas.mtimes(cas.mtimes(cas.transpose(x[:, Npred] - ref_point), self.P), x[:, Npred] - ref_point)
                solver_2.minimize(objective)
                
                # Define the solver
                options = {'ipopt': {'print_level': 0, 'sb': 'yes'},'print_time':0}
                solver_2.solver('ipopt', options)

                # Solve the problem
                solver_2.set_value(xinit, pose_robot)

                sol = solver_2.solve()
                usol = sol.value(u)
                self.u0 = usol[:,0]
                self.u0 = self.u0.reshape(2,1)
                # u = u_imp + u_ref
                u_vir = self.u0 + self.u_ref_traj[:, self.iref].reshape(2,1)
                
                self.u_1_plot.append(u_vir[0,0])
                self.u_2_plot.append(u_vir[1,0])
                self.u_real = np.array([[np.cos(theta), np.sin(theta)],
                                        [-np.sin(theta)/self.b, np.cos(theta)/self.b]]) @ u_vir
                # Publishing control signal
                cmd_lin_vel = self.u_real[0,0]
                cmd_ang_vel = self.u_real[1,0]
        
                print(f'linear speed: {cmd_lin_vel},  angular speed: {cmd_ang_vel}')
                self.pub_vel(cmd_lin_vel, cmd_ang_vel)
                end = time.time()
                denta_time_compute = end - start
                self.compute_time += denta_time_compute
                print(f'Computation time for one loop: {denta_time_compute} (s)')
                self.lin_v_plot.append(self.u_real[0,0])
                self.ang_v_plot.append(self.u_real[1,0])
                self.counter +=1
                time_series = self.counter *self.Ts
                self.time.append(time_series)
                self.iref = self.iref + 1
            else: 
                self.u_real[0,0] = 0.0
                self.u_real[1,0] = 0.0
                self.trajectory_tracking_finish = True  
                self.pub_vel(self.u_real[0,0],self.u_real[1,0])
                # self.lin_v_plot.append(self.u_real[0])
                # self.ang_v_plot.append(self.u_real[1])
                avg_time = self.compute_time / self.counter
                # RMS tracking error
                squared_errors = (self.ref_traj - self.actual_poses) ** 2 
                squared_distances = np.sum(squared_errors, axis=0)
                rmse = np.sqrt(np.mean(squared_distances))

                # avg_er_tracking = self.total_error_tracking / self.counter
                self.counter +=1
                # time_series = self.counter *self.Ts
                # self.time.append(time_series)
                print("Robot completed tracking target")
                print("Computation time in average: ", avg_time)
                # print(f"Avg tracking error: {avg_er_tracking}")
                print(f"RMS tracking error: {rmse}")
                self.get_logger().info("Number of target tracking steps: {}".format(self.counter))
                # Save to .mat file to plot in matlab
                scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/imp_mpc/imp_mpc/imp_traj_pos.mat', {'x': self.x_plot, 'y': self.y_plot})
                scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/imp_mpc/imp_mpc/imp_traj_lin_vel.mat', {'x': self.time, 'y': self.lin_v_plot})
                scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/imp_mpc/imp_mpc/imp_traj_ang_vel.mat', {'x': self.time, 'y': self.ang_v_plot})
                scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/imp_mpc/imp_mpc/ref_traj.mat', {'x': self.xref, 'y': self.yref})
                print("Data saving")


    

def main(args = None):
    rclpy.init(args=args)
    node = MyImpMPC()
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
        ax1.plot(node.xref, node.yref, 'r', label = "Reference trajectory", markersize=12) 
        # ax1.plot(node.target[0,0], node.target[1, 0], 'r', label = "Target of robot", markersize=12)
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
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Linearized input of robot')
        ax2.plot(x, y, label = 'Input constraint set')
        ax2.plot(node.u_1_plot, node.u_2_plot, 'm', label = "Trajectory of linearized input")
        ax2.set_xlabel('u1', fontsize = 13)
        ax2.set_ylabel('u2', fontsize = 13)
        ax2.axis('tight')
        ax2.legend().set_draggable(True)
        ax2.grid(True)
       
        # Plot linear velocity of robot
        ax3 = fig.add_subplot(gs[1, 0])
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
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Angular velocity of robot')
        ax4.plot(node.time, node.ang_v_plot, 'k',lw = 0.6, label = 'Angular velocity')
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



