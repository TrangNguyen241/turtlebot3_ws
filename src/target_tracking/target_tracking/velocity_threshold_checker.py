import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class VelocityThresholdChecker(Node):
    def __init__(self):
        super().__init__('velocity_threshold_checker')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.linear_threshold = 0.005  # Ngưỡng phát hiện robot di chuyển (m/s)
        self.v_current = 0.01          # Bắt đầu với 0.01 m/s
        self.v_max = 0.1
        self.v_step = 0.005
        self.detected = False

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('🔍 Checking theshold velocity...')

    def odom_callback(self, msg):
        v_actual = msg.twist.twist.linear.x
        if abs(v_actual) > self.linear_threshold and not self.detected:
            self.get_logger().info(f'✅ Robot start to move at velocity: {self.v_current:.3f} m/s')
            self.detected = True
            self.stop_robot()

    def timer_callback(self):
        if self.v_current > self.v_max or self.detected:
            self.stop_robot()
            rclpy.shutdown()
            return

        twist = Twist()
        twist.linear.x = self.v_current
        self.cmd_pub.publish(twist)
        self.get_logger().info(f'🚀 Applying velocity: {self.v_current:.3f} m/s')
        self.v_current += self.v_step

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)
        self.get_logger().info('🛑 Stopped robot')

def main(args=None):
    rclpy.init(args=args)
    node = VelocityThresholdChecker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
