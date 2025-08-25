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

import scipy.io

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import sys
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/exp_mpc')
from utilities.exmpc_target_2 import *
from utilities.compute_bspline_curve import *
class MyExpMPC(Node):
    def __init__(self):
        super().__init__("exp_mpc_target")
        self.Ts = 0.1 # sampling time
        # self.Ts = 0.02 # sampling time
        self.b = 0.1 #tracking point of robot (it is placed at distance b in front of robot)
        self.MAX_LIN_VEL = 0.22
        self.MIN_LIN_VEL = -self.MAX_LIN_VEL
        self.MAX_ROT_VEL = 2.84
        self.MIN_ROT_VEL = -self.MAX_ROT_VEL
        
        # States of robot
        self.x = None
        self.y = None
        self.theta = None
        # Constraint set for u1, u2
        self.ru = min(self.b*self.MAX_ROT_VEL, self.MAX_LIN_VEL) 

        # Reference 
        # self.target = np.array([[0.2],
        #                         [0.4]])
        self.target = np.array([[0.4],
                                [1.0]])
        self.std_error = 0.005

        # For plotting positions of robot
        self.x_plot = []
        self.y_plot = []
        # For plotting velocity of robot
        self.lin_v_plot = []
        self.time = []
        self.ang_v_plot = []
        self.u_1_plot = []
        self.u_2_plot = []

        self.counter = 0 # counter for computation steps
        self.target_tracking_finish = False
        self.compute_time = 0
 
        # Subscribe and publish
        self.supscription = self.create_subscription(Odometry,'odom',self.odometry_callback, qos_profile=qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist,'cmd_vel', 10)
        # Publishers for reference path and actual path
        self.ref_path_pub = self.create_publisher(Path, '/reference_path', 10)
        self.act_path_pub = self.create_publisher(Path, '/actual_path', 10)
        # Initialize Path messages
        self.reference_path = Path()
        self.reference_path.header.frame_id = "odom"  # Adjust frame_id as needed

        self.actual_path = Path()
        self.actual_path.header.frame_id = "odom"

        self.timer = self.create_timer(self.Ts, self.exp_mpc_control_loop)


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


    def exp_mpc_control_loop(self):
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
            self.publish_actual_path(pose_robot[0, 0], pose_robot[1, 0])
            error_to_target = np.linalg.norm(pose_robot - self.target)
            if (error_to_target > self.std_error):
                # Computing the control signal
                start_time_compute = time.time()
                u_vir = exmpc_target_2(pose_robot - self.target)
                end_time_compute = time.time()
                denta_time_compute = end_time_compute - start_time_compute
                self.compute_time = self.compute_time + denta_time_compute
                print("Computation time of one loop expl MPC: ", denta_time_compute)
                self.u_1_plot.append(u_vir[0, 0])
                self.u_2_plot.append(u_vir[1, 0])
                T_fl = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta)/self.b, np.cos(theta)/self.b]])
                self.u_real = T_fl @ u_vir[:2]
                self.get_logger().info("lin vel: {}, ang vel: {}".format(self.u_real[0, 0], self.u_real[1, 0]))

                # Publishing control signal for robot
                cmd_msg = Twist()
                cmd_msg.linear.x = self.u_real[0, 0]
                cmd_msg.angular.z = self.u_real[1, 0]
                self.publisher.publish(cmd_msg)
                self.lin_v_plot.append(self.u_real[0, 0])
                self.ang_v_plot.append(self.u_real[1, 0])
                self.counter +=1
                time_series = self.counter *self.Ts
                self.time.append(time_series)
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
                avg_time = self.compute_time / (self.counter-1)
                print("Computation time in average: ", avg_time)
                self.get_logger().info("Number of target tracking steps: {}".format(self.counter))
                # Save data
                # np.savez('/home/nguyehtt/turtlebot3_ws/src/exp_mpc/exp_mpc/data_position_expmpc.mpy', x=self.x_plot, y=self.y_plot)
                scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/exp_mpc/exp_mpc/exp_target_pos.mat', {'x': self.x_plot, 'y': self.y_plot})
                scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/exp_mpc/exp_mpc/exp_target_lin_vel.mat', {'x': self.time, 'y': self.lin_v_plot})
                scipy.io.savemat('/home/nguyehtt/turtlebot3_ws/src/exp_mpc/exp_mpc/exp_target_ang_vel.mat', {'x': self.time, 'y': self.ang_v_plot})
                print("Data saving")


def main(args = None):
    rclpy.init(args=args)
    node = MyExpMPC()
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
        ax1.plot(node.target[0, 0], node.target[1, 0], '*r', label = "Target of robot", markersize=12)
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
if __name__ == "__main__":
    main()