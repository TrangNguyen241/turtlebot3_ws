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
# Import functions
sys.path.append('/home/nguyehtt/turtlebot3_ws/src')
from utilities_formation.callback_states import *
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/saturated_control')
from utilities.checkPolytope import *
class MyMultipleSaturated(Node):
    def __init__(self):
        super().__init__("saturated_multiple")
        
        # Parameters of system
        self.MAX_LIN_VEL = 0.22
        self.MIN_LIN_VEL = -self.MAX_LIN_VEL
        self.MAX_ROT_VEL = 2.84
        self.MIN_ROT_VEL = -self.MAX_ROT_VEL
        self.b = 0.05
        A = np.zeros((2,2))
        self.B = np.eye(2)
        self.Ts = 0.02
        nx = A.shape[1]
        nu = A.shape[1]
        C = np.eye(2)
        # Find Q by LMI
        alpha = 0.5
        self.gamma = 5
        Q = cp.Variable((nx, nx), symmetric = True)
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
        # self.P = np.array([[0.1051, 0],[0, 0.1051]])
        # Contraints of Inputs (u1, u2)
        R = 33e-3
        D = 160e-3
        Omega_bar = 6.67
        ru = 2*Omega_bar*self.b*R/np.sqrt(4*self.b**2 + D**2)
        ptsU = []
        for tta in np.linspace(0, 2 * np.pi - 1e-4, 20):
            ptsU.append([ru * np.cos(tta), ru * np.sin(tta)])
        ptsU = np.array(ptsU)
        self.U_input = pc.qhull(ptsU) #checked
        # Number of robots
        self.N = 3
        # Target point
        self.ref_point = np.array([[1, 4, -3],
                                [-4, 4, 4]])
        self.std_error = 0.01
        # Control signal 
        self.u_vir = np.zeros((2, self.N)) 
        self.u_real = np.zeros((2, self.N)) 
        self.error_to_target = np.zeros((1, self.N))
        # States of robot
        self.robot_positions = {i: (0, 0, 0) for i in range(self.N)}
        
        self.robot_subscriptions = {}
        self.robot_publisher = {}
        # Subscribe to the topic /odom (given by ROS) of 4 robots to receive their positions
        for i in range(self.N):
            self.robot_subscriptions[i] = self.create_subscription(
                Odometry,
                f'/turtlebot3{i}/odom',
                lambda msg, robot_id=i: self.odom_callback_wrapper(msg, robot_id),
                qos_profile=qos_profile_sensor_data
            )
        # Create a publisher to publish the linear and angular velocities of the 4 robots
        for i in range(self.N):
            self.robot_publisher[i] = self.create_publisher(
                Twist,
                f'/turtlebot3{i}/cmd_vel',
                10
            )
        self.timer = self.create_timer(self.Ts, self.saturated_control)
    def odom_callback_wrapper(self, msg, robot_id):
        x, y, theta = odometry_callback(msg = msg)
        self.robot_positions[robot_id] = (x, y, theta)
        self.get_logger().info(f"Robot {robot_id} position: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")

    def uni_si_states(self, state):
        for i in range(state.shape[1]): 
            state[0, i] = state[0, i] + self.b*np.cos(state[2, i])
            state[1, i] = state[1, i] + self.b*np.sin(state[2, i])
        return state

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
    def saturated_control(self):
        if all(all(coord == 0.0 for coord in value) for value in self.robot_positions.values()):
            self.get_logger().warn("Waiting for ODOMETRY......")
            time.sleep(self.Ts)
            return
        # Linearized states of robot
        poses_uni = np.array([[self.robot_positions[i][0], self.robot_positions[i][1], self.robot_positions[i][2]] for i in range(self.N)]).T
        poses_robot = self.uni_si_states(poses_uni)
        for i in range(self.N):
        # for i in range(poses_robot.shape[1]):
            self.error_to_target[0, i] = np.linalg.norm(poses_robot[:2, i] - self.ref_point[:, i])
            self.get_logger().info(f"@@@@@@ Error {i}: {self.error_to_target[0, i]} @@@@@@@@@")

        # for i in range(self.N):
            if self.error_to_target[0, i] > self.std_error:
                self.u_vir[:,i] = -self.gamma * self.B.T @ self.P @ (poses_robot[:2, i] - self.ref_point[:, i])
                if check_u_inputs(self.U_input, self.u_vir[:, i]):
                    lamda = 1
                else:
                    lamda_list = self.U_input.b / (self.U_input.A @ self.u_vir[:,i])
                    valid_lamda = lamda_list[(lamda_list >= 0)&(lamda_list<=1)]
                    lamda_min = np.min(valid_lamda)
                    lamda = lamda_min
                u_vir_sat = lamda * self.u_vir[:,i]
                self.u_real[:,i] = np.array([[np.cos(poses_robot[2, i]), np.sin(poses_robot[2, i])],
                               [-np.sin(poses_robot[2, i])/self.b, np.cos(poses_robot[2, i])/self.b]]) @ u_vir_sat
                self.get_logger().info(f"##### Robot {i} lin vel: {self.u_real[0, i]}, ang vel: {self.u_real[1, i]}####")
            else:
                self.u_real[0, i] = 0.0
                self.u_real[1, i] = 0.0
        # for i in range(self.N):
            cmd_msg = Twist()
            cmd_msg.linear.x = np.clip(self.u_real[0, i], self.MIN_LIN_VEL, self.MAX_LIN_VEL)
            cmd_msg.angular.z = np.clip(self.u_real[1, i], self.MIN_ROT_VEL, self.MAX_ROT_VEL)
            self.robot_publisher[i].publish(cmd_msg)
            self.get_logger().info("lin vel_{}: {}, ang vel_{}: {}".format(i, self.u_real[0, i],i, self.u_real[1, i]))
        

def main(args = None):
    rclpy.init(args=args)
    node = MyMultipleSaturated()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == "__main__":
    main()