##### Code nay do con Chatbot tao ra #####
# TurtleBot PID Controller
# This script implements a PID controller for a TurtleBot to navigate towards a target position.
# It subscribes to the odometry data and publishes velocity commands to the /cmd_vel topic.
# The PID controller adjusts the linear and angular velocities based on the distance and angle to the target.
# It also includes a simple stopping condition when the robot is close enough to the target.
# Import necessary libraries

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import atan2, sqrt 


class TurtleBotPID(Node):
    def __init__(self):
        super().__init__('turtlebot_pid_controller')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        # PID parameters
        self.kp_linear = 1.0
        self.ki_linear = 0.0
        self.kd_linear = 0.0

        self.kp_angular = 4.0
        self.ki_angular = 0.0
        self.kd_angular = 0.0

        # Error terms
        self.prev_linear_error = 0.0
        self.prev_angular_error = 0.0
        self.integral_linear = 0.0
        self.integral_angular = 0.0

        # Current position
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Target position
        self.target_x = 5.0
        self.target_y = 5.0

        # Control flag
        self.reached_goal = False

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        self.theta = 2 * atan2(orientation_q.z, orientation_q.w)

    def compute_pid(self):
        twist = Twist()

        # Compute distance and angle to the target
        distance = sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
        angle_to_target = atan2(self.target_y - self.y, self.target_x - self.x)
        angular_error = angle_to_target - self.theta

        # Linear PID control
        linear_error = distance
        self.integral_linear += linear_error
        derivative_linear = linear_error - self.prev_linear_error
        twist.linear.x = (self.kp_linear * linear_error +
                          self.ki_linear * self.integral_linear +
                          self.kd_linear * derivative_linear)
        self.prev_linear_error = linear_error

        # Angular PID control
        self.integral_angular += angular_error
        derivative_angular = angular_error - self.prev_angular_error
        twist.angular.z = (self.kp_angular * angular_error +
                           self.ki_angular * self.integral_angular +
                           self.kd_angular * derivative_angular)
        self.prev_angular_error = angular_error

        # Apply velocity limits
        max_linear_velocity = 0.22  # Maximum linear velocity
        max_angular_velocity = 2.6  # Maximum angular velocity

        twist.linear.x = max(min(twist.linear.x, max_linear_velocity), -max_linear_velocity)
        twist.angular.z = max(min(twist.angular.z, max_angular_velocity), -max_angular_velocity)

        return twist, distance

    def control_loop(self):
        if self.reached_goal:
            return

        twist, distance = self.compute_pid()
        self.pub.publish(twist)

        # Stop if the robot is close enough to the target
        if distance < 0.1:
            self.reached_goal = True
            self.pub.publish(Twist())  # Stop the robot
            self.get_logger().info('Goal reached!')

def main(args=None):
    rclpy.init(args=args)
    controller = TurtleBotPID()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
