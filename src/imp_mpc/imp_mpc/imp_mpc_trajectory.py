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
import time
import math
import casadi as cas 
import threading
from scipy.interpolate import BSpline

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class MPC_controller(Node):
#### init ####
    def __init__(self):
        super().__init__('MPC_controller')

#### define variable ####

        # Limits of the Turtlebot3: Burger
        self.MAX_LINEAR_SPEED = 0.22
        self.MIN_LINEAR_SPEED = -0.22
        self.MAX_ROTATION_SPEED = 2.82
        self.MIN_ROTATION_SPEED = -2.82

        self.min_input = np.array([[-0.22],[-2.82]])
        self.max_input = np.array([[0.22],[2.82]])
        self.delta_min_input = np.array([[-0.015],[-0.15]])
        self.delta_max_input = np.array([[0.015],[0.15]])

        # Sampling time
        self.Ts = 0.4

        # Time variables
        self.start_time = time.time()
        self.end_time = None

        # Variables to store commands
        self.command_linear_speed = 0.00
        self.command_angular_speed = 0.00

        # Robot's position updated by odometry_callback or rigid_bodies_callback
        self.x = None
        self.y = None
        self.z = None
        self.theta = None

        # Initialize vectors for plot and data saving
        self.x_plot = []
        self.y_plot = []

        # states of system
        self.state_0 = None

        # Maximum errors
        # self.distance_error = 0.02
        self.distance_error_1 = 0.2
        self.angle_error = 0.05

        # input
        self.u0 = np.array([[0],[0]])

        # output_desired
        
        self.out_desired = None

        # create velocity publisher
        self.publish_vel = self.create_publisher(Twist,'cmd_vel',10)
        # create odometry subscriber
        self.sub_odom = self.create_subscription(Odometry,'odom',self.odometry_callback, qos_profile=qos_profile_sensor_data)
        # self.subscription = self.create_subscription(RigidBodies, 'rigid_bodies', self.rigid_bodies_callback, 10)
        # Publish data to plot
        self.pub_x_ref = self.create_publisher(Tuple,'x_ref',10)
        self.pub_y_ref = self.create_publisher(Tuple,'y_ref',10)
        self.pub_control_point = self.create_publisher(TupleList,'control_point',10)
        # Publishers for reference path and actual path
        self.ref_path_pub = self.create_publisher(Path, '/reference_path', 10)
        self.act_path_pub = self.create_publisher(Path, '/actual_path', 10)
        # Initialize Path messages
        self.reference_path = Path()
        self.reference_path.header.frame_id = "odom"  # Adjust frame_id as needed

        self.actual_path = Path()
        self.actual_path.header.frame_id = "odom"


#### MPC Controller ####
    def mpc_controller(self):
        while (self.x is None) or (self.y is None) or (self.theta is None):
            self.get_logger().warn("Waiting for ODOMETRY......")
            time.sleep(self.Ts)

        #### initializing start position ####     
        x_0 = self.x
        y_0 = self.y
        theta_0 = self.theta

        # create trajectory

        # control_point = [(x_0,y_0),(1,1),(3,2),(4,1),(5,3),(6,2),(4,0)]  
        # control_point = [(x_0,y_0),(1,1),(2,5),(4,-2),(-4,4),(0,2),(5,5)]  
        # control_point = [(x_0,y_0),(1,1),(3,2),(4,1),(5,3),(6,2),(5,5)] 
        # control_point = [(x_0,y_0),(1,1),(3,1.5),(4,0),(5,1),(7,2),(8,0.5)] 
        # control_point = [(0.0,0.0),(1.0,0.5),(0.0,1.0),(-1.0,0.5),
        #             (0.0,0.0),(1.0,-0.5),(0.0,-1.0),(-1.0,-0.5),
        #             (0.0,0.0),(0.5,0.25),(1.0,0.0),(0.5,-0.25),
        #             (0.0,0.0),(-0.5,0.25),(-1.0,0.0),(-0.5,-0.25),(0.0,0.0)]
        control_point = [(x_0,y_0),(0.5,0.5),(1,-0.5),(1.5,0),(1.75,-0.3),(2,0.4),(2.2,0.3),(3,0.6)] 
        
        num_control_point = len(control_point)
        d = 3
        t = np.linspace(0, 1, num_control_point + d + 1 - 2*d)
        knots = np.concatenate(([0]*d, t, [1]*d))
        bspline = BSpline(knots, control_point, d)
        t_eval = np.linspace(0, 1, 70)  #### you can seperate points here
        self.spline_points = bspline(t_eval)
        self.x_ref = self.spline_points[:,0] 
        self.y_ref = self.spline_points[:,1]
        self.x_ref = np.append(self.x_ref, [self.x_ref[-1]])
        self.y_ref = np.append(self.y_ref,[self.y_ref[-1]])

        
        self.control_point_x_plot = []
        self.control_point_y_plot = []
        for i in control_point:
            self.control_point_x_plot.append(i[0])
            self.control_point_y_plot.append(i[1])

        #### create desired points matrix
        desired_ref = []
        
        for k in range(len(self.x_ref) - 1):
            angle = math.atan2((self.y_ref[k+1] - self.y_ref[k]),(self.x_ref[k+1] - self.x_ref[k]))
            if len(desired_ref) == 0:
                theta_0 = self.theta
                unwrapped_angles = np.unwrap([angle, theta_0])
                angle = unwrapped_angles[0]
                desired_ref.append([[self.x_ref[k]],[self.y_ref[k]],[angle]])
                last_angle = angle
            else:
                unwrapped_angles = np.unwrap([angle, last_angle])
                angle = unwrapped_angles[0]
                desired_ref.append([[self.x_ref[k]],[self.y_ref[k]],[angle]])
                last_angle = angle
       
                    
        Npred = 15  # number of predictions

        last_value = desired_ref[-1]
        for k in range(Npred):
            desired_ref.append(last_value)
        
        desired_ref = np.array(desired_ref)
        desired_ref = desired_ref.reshape(desired_ref.shape[0], desired_ref.shape[1]).T

        print("kich thuoc cua desired_ref: ", desired_ref.shape[1])
        print(desired_ref[:,3])
        print(desired_ref[:,3].shape)
       
        #### tracking trajectory
        for k in range(len(self.x_ref)-1):
            # initializing start position ####
            x_0 = self.x
            y_0 = self.y
            theta_0 = self.theta
            # desired position without taking into account angle
            x_desired = self.x_ref[k]
            y_desired = self.y_ref[k]
            # Publishing reference path to Rviz (visualization)
            self.publish_reference_path(self.x_ref, self.y_ref)
            
            # compute the position error
            delta_x = x_desired - x_0
            delta_y = y_desired - y_0
            position_error = math.sqrt(delta_x ** 2 + delta_y ** 2) # position error 
            angle_desired = math.atan2((self.y_ref[k+1] - self.y_ref[k]),(self.x_ref[k+1] - self.x_ref[k]))
            unwrapped_angles = np.unwrap([angle_desired, theta_0])
            angle_desired = unwrapped_angles[0]
            theta_0 = unwrapped_angles[1]

            self.out_desired = np.array([[x_desired],[y_desired],[angle_desired]])

            while position_error > self.distance_error_1:
                start = time.time()
                x_0 = self.x
                y_0 = self.y
                theta_0 = self.theta
                self.state_0 = np.array([[x_0],[y_0],[theta_0]])
                A = np.array([[math.cos(theta_0), 0],[math.sin(theta_0), 0],[0, 1]])
                #### model dimension ####
                dx, du = np.shape(A)
                # Define weighting matrices
                w_x = 1000
                w_y = 1000
                w_theta = 0.00011
                Q = np.array([[w_x, 0, 0],
                              [0, w_y, 0],
                              [0, 0, w_theta]])
                R = 100
                # Optimization problem using casadi
                solver_2 = cas.Opti()
                # define variables
                x = solver_2.variable(dx, Npred+1)
                u = solver_2.variable(du, Npred)
                xinit = solver_2.parameter(dx, 1)
                uinit = solver_2.parameter(du,1)
                A = [solver_2.variable(dx, du) for _ in range(Npred)]
                # out_desired = solver_2.parameter(dx, Npred+1)
                out_desired = solver_2.parameter(dx,1)

                # initialize constrains
                solver_2.subject_to(x[:,0] == xinit)
                for i in range(0, Npred):
                    # dynamic
                    theta_k = x[2, i]
                    A_k = cas.vertcat(
                        cas.horzcat(cas.cos(theta_k), 0),
                        cas.horzcat(cas.sin(theta_k), 0),
                        cas.horzcat(0, 1)
                    )
                    solver_2.subject_to(A[i] == A_k)
                    solver_2.subject_to(x[:,i+1] == x[:,i] + cas.mtimes(A[i], u[:,i]) * self.Ts)
                    # input constraints
                    solver_2.subject_to(self.min_input <= u[:,i])
                    solver_2.subject_to(u[:,i] <= self.max_input)
                    if i == 0:
                        solver_2.subject_to(self.delta_min_input <= u[:,i] - uinit)
                        solver_2.subject_to(self.delta_max_input >= u[:,i] - uinit)
                    else:
                        solver_2.subject_to(self.delta_min_input <= u[:,i] - u[:,i-1])
                        solver_2.subject_to(self.delta_max_input >= u[:,i] - u[:,i-1])
                
                # Initialize objective
                objective = 0
                for i in range(0,Npred):
                    if i == 0:
                        objective = objective + cas.mtimes(cas.mtimes(cas.transpose(x[:,i] - out_desired), Q), x[:,i] - out_desired) +\
                                        cas.mtimes(cas.mtimes(cas.transpose(u[:,i]), R), u[:,i])
                    else:
                        objective = objective + cas.mtimes(cas.mtimes(cas.transpose(x[:,i] - out_desired), Q), x[:,i] - out_desired) +\
                                        cas.mtimes(cas.mtimes(cas.transpose(u[:,i] ), R), u[:,i] )
                solver_2.minimize(objective)
                
                # Define the solver
                options = {'ipopt': {'print_level': 0, 'sb': 'yes'},'print_time':0}
                solver_2.solver('ipopt', options)

                # Solve the problem
                solver_2.set_value(xinit, self.state_0)
                solver_2.set_value(uinit, self.u0)
                solver_2.set_value(out_desired, self.out_desired)

                sol = solver_2.solve()
                usol = sol.value(u)
                self.u0 = usol[:,0]
                self.command_linear_speed = usol[:,0][0]
                self.command_angular_speed = usol[:,0][1]
                print(f'linear speed: {self.command_linear_speed},  angular speed: {self.command_angular_speed}')
                self.pub_vel(linear=self.command_linear_speed, angular=self.command_angular_speed)

                end = time.time()
                time_elapsed = end - start
                print(f'Computation time: {time_elapsed} (s)')
                if (time_elapsed < self.Ts and time_elapsed > 0):
                    time.sleep(self.Ts - time_elapsed)

                # measure the position againt
                x_0 = self.x
                y_0 = self.y
                theta_0 = self.theta
                delta_x = self.out_desired[0] - x_0
                delta_y = self.out_desired[1] - y_0
                position_error = math.sqrt(delta_x ** 2 + delta_y ** 2) # position error
                # print(f'position error: {position_error}, angular_error: {angle_error}')
                print("gia tri cua k hien tai la: ",k)
                # save to plot
                self.x_plot.append(x_0)
                self.y_plot.append(y_0)
                # Publishing positions of robot to Rviz
                self.publish_actual_path(x_0, y_0)


                # publish data to plot
                x_ref_msg = Tuple()
                x_ref_msg.tuple = self.x_ref.tolist()
                self.pub_x_ref.publish(x_ref_msg)

                y_ref_msg = Tuple()
                y_ref_msg.tuple = self.y_ref.tolist()
                self.pub_y_ref.publish(y_ref_msg)

                control_point_msg = Tuple()
                list_control_point_msg = TupleList()
                for i in control_point:
                    control_point_msg.tuple.append(i[0])
                    control_point_msg.tuple.append(i[1])
                    list_control_point_msg.tuplelist.append(control_point_msg)
                    control_point_msg = Tuple()
                self.pub_control_point.publish(list_control_point_msg)


        self.pub_vel(linear=0.0, angular=0.0)

        
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
        x=msg.rigidbodies[4].pose.position.x
        y=msg.rigidbodies[4].pose.position.y
        z=msg.rigidbodies[4].pose.position.z

        qx = msg.rigidbodies[4].pose.orientation.x
        qy = msg.rigidbodies[4].pose.orientation.y
        qz = msg.rigidbodies[4].pose.orientation.z
        qw = msg.rigidbodies[4].pose.orientation.w

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
    
    def pub_vel(self, linear, angular):
        msg = Twist()
        command_linear_speed = min(max(linear, self.MIN_LINEAR_SPEED), self.MAX_LINEAR_SPEED)
        command_angular_speed = min(max(angular, self.MIN_ROTATION_SPEED), self.MAX_ROTATION_SPEED)
        msg.linear.x = command_linear_speed
        msg.angular.z = command_angular_speed
        self.publish_vel.publish(msg)

    

def main(args = None):
    rclpy.init(args=args)
    node = MPC_controller()
    try:
        controller_thread = threading.Thread(target=node.mpc_controller)
        controller_thread.daemon = True
        controller_thread.start()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        plt.title("MPC tracking trajectory")
        plt.plot(node.x_ref, node.y_ref,lw = 2,color= 'g', label = 'reference')
        # plt.plot(node.control_point_x_plot, node.control_point_y_plot, '-+b', label = 'control point')
        plt.plot(node.x_plot, node.y_plot, '.r', label = 'real_trajectory')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



