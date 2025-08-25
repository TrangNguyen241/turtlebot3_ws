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
import copy
# Plotting packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

# Messages for visualization in Rviz
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Import neccessary functions
import sys
sys.path.append('/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/utilities')
from rg_funcs import *
# from utilities.poly_decomp import *
# from utilities import * 
# from poly_decomp import poly_decomp as pd

class RGAvoidObs(Node):
    def __init__(self):
        #####################################################################
        #                       Initialization                              #
        #####################################################################
        super().__init__('ref_gov_target')
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
        self.b = 0.1 # the control point in front of robot
        self.A = np.zeros((2,2)) # matrix A in continuous time
        self.B = np.eye(2) # matrix B in continuos time
        self.C = np.eye(2)
        self.D = np.zeros((2, 2))
        self.Ad = np.eye(2) + self.A * self.Ts # matrix A in discrete time
        self.Bd = self.B * self.Ts # matrix B in discrete time
        self.nx = self.A.shape[1]
        self.nu = self.A.shape[1]
        self.nr = self.C.shape[0]
        self.Mc = np.block([[self.A, self.B], [self.C, self.D]])
        self.Mc_inv = np.linalg.inv(self.Mc)
        self.Kxr = self.Mc_inv[:self.nx, -self.nu:]
        # Constraint set for u1, u2
        self.ru = min(self.b*self.MAX_ROT_VEL, self.MAX_LIN_VEL) # radius of ball constraint of u
        ptsU = []
        for tta in np.linspace(0, 2 * np.pi - 1e-4, 10):
            ptsU.append([self.ru * np.cos(tta), self.ru * np.sin(tta)])
        ptsU = np.array(ptsU) # approximation of constraints set as a polytope
        self.U_input = pc.qhull(ptsU) 
        self.Up = Polygon(ptsU, facecolor=(1, 1, 0, 0.1),edgecolor=(0, 0, 0, 0.8)) # for plotting U_input
        # Parameters for control input (conmputed in Matlab)
        self.Q = np.array([[10.0, 0], [0, 10.0]])
        # self.R = np.array([[-5.125, 0], [0, -5.125]]) # beta = 0.025
        # self.R = np.array([[-7.5003, 0], [0, -7.5003]]) # beta = 0.5
        self.R = np.array([[-10, 0], [0, -10]]) # beta = 1.0
        # self.R = np.array([[-15.0, 0], [0, -15.0]]) # beta = 2.0
        self.P = np.linalg.inv(self.Q)
        self.K = np.dot(self.R, self.P)
        self.list_gamma = []

        # Find Q by LMI
        # self.beta = 0.025
        # epsilon = 1e-3
        # Q = cp.Variable((self.nx, self.nx), symmetric = True)
        # R = cp.Variable((self.nu, self.nx))
        # # LMI constraints
        # lmi1 = Q @ self.A.T  + self.A @ Q + self.B @ R + R.T @ self.B.T + self.beta*Q 
        # # lmi2 = Q >= 0
        # constraints = [lmi1 <= 0, Q >= 0]
        # # Define the optimization problem
        # objective = cp.Minimize(0) # No objective, just looking for feasibility
        # problem = cp.Problem(objective, constraints)
        # # Solve the problem
        # problem.solve(solver = cp.SCS, verbose = False)
        # # Check the solution
        # if problem.status not in ["optimal", "optimal_inaccurate"]:
        #     raise ValueError("Cannot find matrix Q")
        
        # Extract the solution
        # self.Q = Q.value
        # self.Q = self.Q + 1e-6 * np.eye(self.nx)
        # self.R = R.value
        # self.P = np.linalg.inv(self.Q)
        # self.K = self.R @ self.P

        # Constraints of state (moving space for robot)
        ws_limit = [[0, 0], [4, 0], [4, 4], [0, 4]]
        init = [0.26, 3.0]  # starting point => Map 2
        goal = [2.3, 0.8] # dep cho map 2
        # Define obstacle list
        obstacles = [
            [[0.5, 0.5], [1.2, 0.4], [1., 1.3]],                                      # obstacle1
            [[2.3, 1.], [1.6, 0.9], [1.5, 1.5], [2.2, 2.0]],                          # obstacle2
            [[0.2, 1.3], [0.5, 1.4], [0.4, 1.7], [0.3, 1.6]],                         # obstacle3
            [[1.0, 2.5], [1.5, 2.5], [2.0, 3.3], [0.5, 3.5]],                         # obstacle4
            [[3.0, 1.5], [2.7, 2.5], [3.2, 3.4], [3.7, 3.0], [3.5, 2.0]],             # obstacle5
            [[2.3, 0.0], [2.3, 0.5], [2.8, 0.5], [2.8, 0.0]]                          # obstacle6 
        ]
        tic = time.time()
        safe_ws_rb = decomp_ws(ws_limit, obstacles)
        sequence_path = shortest_path_idx_poly(init, goal, safe_ws_rb)
        self.rb_path, self.trans_list, self.poly_merge_plot = path_poly_transition(sequence_path, safe_ws_rb)
        toc = time.time()
        self.comp_time = toc - tic
        self.get_logger().info("--------COMPUTATION TIME-----: {}".format(self.comp_time))

        # Get the center of transition zone as a temporary reference
        self.ref_list = []
        self.ref_list_trans = []
        for i in range(len(self.trans_list)):
            vertices_trans = self.trans_list[i].vertices
            center_trans = np.mean(vertices_trans, axis=0)
            self.ref_list.append(center_trans)
            self.ref_list_trans.append(center_trans)
        self.desired_target = copy.deepcopy(goal)
        self.desired_target = np.array(self.desired_target)
        self.ref_list.append(self.desired_target)
        self.reach_target = False
        self.std_error = 0.1

        # For storing data to be able to plot in Matlab
        rb_vertices = []
        for poly in self.poly_merge_plot:
            rb_vertices.append(poly)

        # Converto to obejct array (cell array in matlab)
        rb_vertices_obj = np.empty(len(rb_vertices), dtype=object)
        rb_vertices_obj[:] = rb_vertices

        # Save to .mat file
        dir_path = "/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/utilities"
        file_name = f"rb_path_vertices_plotting.mat"
        savemat(os.path.join(dir_path, file_name), {"rb_path_plotting": rb_vertices_obj})

        # For storing to export to Matlab
        # Convert to a single 2D numpy array of shape (N,2)
        ref_array = np.stack(self.ref_list_trans)
        # Save to .mat file
        savemat(os.path.join("/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/utilities", f"list_cen_trans.mat"), {"ref_list_trans": ref_array})


        self.pose = np.zeros((3,1))
        self.x = None
        self.y = None
        self.theta = None
        init_point = copy.deepcopy(init)
        init_pose_np = np.array(init_point).reshape(-1,1)
        init_pose = np.array([[init_pose_np[0,0]],[init_pose_np[1,0]],[np.pi/2]]) 

        self.z_init = np.array((2,1))
        self.ureal = np.zeros((self.nu, 1))
        self.target_idx = 0
        self.flag = 1
        self.kappa=350.0
        self.eta = 0.075
        self.AllConstraint = {
            "A": np.vstack([
                np.hstack([self.U_input.A @ self.K, -self.U_input.A @ self.K @ self.Kxr]),
                np.hstack([self.rb_path[self.target_idx].A, np.zeros((self.rb_path[self.target_idx].A.shape[0], self.nr))])
            ]),
            "b": np.concatenate((self.U_input.b, self.rb_path[self.target_idx].b))
        }
        self.Ref = self.ref_list[self.target_idx]
        self.filteredRef = np.array((2,1))

        # For plotting
        self.xplot = []
        self.yplot = []
        self.linvel = []
        self.angvel = []
        self.time_plot = []
        self.counter = 0

        # Subscription and publish:
        self.supscription = self.create_subscription(Odometry,'odom',self.odometry_callback, qos_profile=qos_profile_sensor_data)
        # Subscribe to the topic /rigid_bodies (given by Qualisys camera system) of robot to receive position
        # self.subscription = self.create_subscription(RigidBodies, 'rigid_bodies', self.rigid_bodies_callback, 10)
        self.publisher = self.create_publisher(Twist,'cmd_vel', 10)
        self.timer = self.create_timer(self.Ts, self.control_loop)

        # Publishers for reference targets and actual path
        self.act_path_pub = self.create_publisher(Path, 'actual_path', 10)
        self.target_pub = self.create_publisher(MarkerArray, "visualization_marker_array", 10)
        # Initialize Path messages

        self.actual_path = Path()
        self.actual_path.header.frame_id = "odom"
        
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
           # Robot 0
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

    def publish_targets(self):
        marker_array = MarkerArray()

        for i, ref in enumerate(self.target, start=1):
            marker = Marker()
            marker.header.frame_id = "odom"  # Change to your reference frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "ref_points"
            marker.id = i  # Mỗi điểm có ID khác nhau
            marker.type = Marker.SPHERE  # A dot
            marker.action = Marker.ADD

            # Lấy tọa độ x, y từ ref
            x, y = ref[0, 0], ref[1, 0]
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

            # Thêm Marker vào danh sách
            marker_array.markers.append(marker)

        # Xuất tất cả điểm lên RViz
        self.target_pub.publish(marker_array)

    def save_data_export(self, target_tb, x_plot, y_plot, time_plot, lin_v_plot, ang_v_plot):
        # *********Define the base directory for saving data in Matlab **********
        base_dir_mat = "/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/data_plotting/plot_matlab"
        
        # Ensure the directory exists
        os.makedirs(base_dir_mat, exist_ok=True)
        
        # Define filenames based on controller type
        x_pos_filename = f"tb_xPos_rg.mat"
        y_pos_filename = f"tb_yPos_rg.mat"
        position_filename = f"tb_Position_rg.mat"
        lin_vel_filename = f"tb_linVel_rg.mat"
        ang_vel_filename = f"tb_angVel_rg.mat"
        
        # Save data
        scipy.io.savemat(os.path.join(base_dir_mat, x_pos_filename), {'x': time_plot, 'y': x_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, y_pos_filename), {'x': time_plot, 'y': y_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, position_filename), {'x': x_plot, 'y': y_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, lin_vel_filename), {'x': time_plot, 'y': lin_v_plot})
        scipy.io.savemat(os.path.join(base_dir_mat, ang_vel_filename), {'x': time_plot, 'y': ang_v_plot})

        # **********Saving data to .npz to plot in Python**************
        base_dir_py = "/home/nguyehtt/turtlebot3_ws/src/obs_avoid/obs_avoid/data_plotting/plot_python"
        os.makedirs(base_dir_py, exist_ok=True)
        npz_filename = os.path.join(base_dir_py, f"tb_data_rg.npz")
        np.savez(npz_filename,
                 target = np.array(target_tb),
                 time = np.array(time_plot),
                 x = np.array(x_plot),
                 y = np.array(y_plot),
                 lin_v = np.array(lin_v_plot),
                 ang_v = np.array(ang_v_plot))
        self.get_logger().info(f"Exported robot data to .mat and .npz")
        
    def erg_robot(self):
        self.get_logger().info("x: {}, y: {}, theta: {}".format(self.x, self.y, self.theta))
        if (self.x is None) or (self.y is None) or (self.theta is None):
            self.get_logger().warn("Waiting for ODOMETRY......")
            time.sleep(self.Ts)
            # Return default values when odometry is not available
            return Twist(), float('inf')  # Default twist and a large distance
        if (self.flag == 1):
            self.pose = np.array([[self.x], [self.y], [self.theta]])
            tta_tmp = self.pose[2, 0]
            self.z_init = np.array([
            [self.pose[0,0] + self.b * np.cos(tta_tmp)],
            [self.pose[1,0] + self.b * np.sin(tta_tmp)]])
            self.filteredRef = self.z_init
            self.flag = 0
        self.pose = np.array([[self.x], [self.y], [self.theta]])
        
        # self.pose = np.array([[3.47], [2.6], [np.pi/2]]) # for debugging
        tta_tmp = self.pose[2, 0]
        z = np.array([
            [self.pose[0,0] + self.b * np.cos(tta_tmp)],
            [self.pose[1,0] + self.b * np.sin(tta_tmp)]
        ])
        self.xplot.append(z[0, 0]) 
        self.yplot.append(z[1, 0])
        # self.publish_actual_path(z[0, 0], z[1, 0])
        # self.publish_targets()

        # distance = sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
        error_to_target = np.linalg.norm(z - self.desired_target.reshape(2,1), axis = 0)
        print(f"Error to target: {error_to_target}")

        # self.filteredRef = z # for debugging
        # self.get_logger().info("z: {}".format(z))
        if self.target_idx < len(self.rb_path) - 1:
            if check_corridor(self.rb_path[self.target_idx], z) and \
               check_corridor(self.rb_path[self.target_idx + 1], z):
                self.target_idx += 1
                self.AllConstraint["A"] = np.vstack([
                    np.hstack([self.U_input.A @ self.K, -self.U_input.A @ self.K @ self.Kxr]),
                    np.hstack([self.rb_path[self.target_idx].A,
                                np.zeros((self.rb_path[self.target_idx].A.shape[0], self.nr))])
                ])
                self.AllConstraint["b"] = np.vstack((
                    self.U_input.b.reshape(-1, 1), self.rb_path[self.target_idx].b.reshape(-1,1)
                ))
                self.Ref = self.ref_list[self.target_idx]
                print("********************Change set-point!**************************")
        # Compute feedback and virtual controls

        print(f"z of robot: {z}")
        xfb = z
        xv = self.Kxr @ self.filteredRef
        
        # Dynamic safety margin
        Gamma = RGThreshold(self.AllConstraint["A"], self.AllConstraint["b"], self.P, xv, self.filteredRef)
        self.list_gamma.append(Gamma)
        
        Delta = self.kappa * (Gamma - (xfb - xv).T @ self.P @ (xfb - xv))
        
        # Navigation field
        self.Ref = self.Ref.reshape(2,1)
        rho = (self.Ref - self.filteredRef) / max(np.linalg.norm(self.Ref - self.filteredRef), self.eta)
        self.filteredRef = self.filteredRef + self.Ts * Delta * rho
        
        xv = self.Kxr @ self.filteredRef
        
        # Compute the virtual control input and transform to real inputs
        u_vir = self.K @ (xfb - xv)

        matrix_transform = np.array([[np.cos(tta_tmp), np.sin(tta_tmp)],
                               [-np.sin(tta_tmp)/self.b, np.cos(tta_tmp)/self.b]])
        
        self.ureal = matrix_transform @ u_vir
        print(f"Linear Vel: {self.ureal[0, 0]}")
        print(f"Angular Vel: {self.ureal[1, 0]}")

        # Publish command for robots
        # cmd_msg = Twist()
        # cmd_msg.linear.x = self.ureal[0, 0]
        # cmd_msg.angular.z = self.ureal[1, 0]
        # self.publisher.publish(cmd_msg)
        # self.get_logger().info("lin vel: {}, ang vel: {}".format(self.ureal[0], self.ureal[1]))  

        twist = Twist()
        twist.linear.x = self.ureal[0,0]
        twist.angular.z = self.ureal[1,0]

        self.linvel.append(self.ureal[0,0])
        self.angvel.append(self.ureal[1,0])

        # Export data for plotting
        self.counter += 1
        time_series = self.counter * self.Ts
        self.time_plot.append(time_series)
        

        return twist, error_to_target


    def control_loop(self):
        if self.reach_target:
            return
        
        twist, error_to_target = self.erg_robot()
        self.publisher.publish(twist)
        if (error_to_target <= self.std_error) and (self.ureal[0,0] <= 0.01):
            self.reach_target = True
            self.publisher.publish(Twist())
            self.save_data_export(self.desired_target, self.xplot, self.yplot, self.time_plot, self.linvel, self.angvel)
            self.get_logger().info('Target reached!')
            self.get_logger().info("Completion time: {} ".format(self.counter*self.Ts))
            # Compute length of trajectory
            x = np.array(self.xplot)
            y = np.array(self.yplot)
            dx = np.diff(x)
            dy = np.diff(y)
            distance = np.sqrt(dx**2 + dy**2)
            total_distance = np.sum(distance)
            self.get_logger().info("Trajectory length: {}".format(total_distance))
            self.get_logger().info("Tracking error: {}".format(error_to_target))
            # Compute min and max Gamma (safety margin)
            min_gamma = min(self.list_gamma)
            max_gamma = max(self.list_gamma)
            self.get_logger().info("Min of Gamma: {}".format(min_gamma))
            self.get_logger().info("Max of Gamma: {}".format(max_gamma))
            # Print computation time
            self.get_logger().info("Computation time offline: {}".format(self.comp_time))

            


        
        

def main(args = None):
    rclpy.init(args=args)
    node = RGAvoidObs()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == "__main__":
    main()



        


        



        

        

        
