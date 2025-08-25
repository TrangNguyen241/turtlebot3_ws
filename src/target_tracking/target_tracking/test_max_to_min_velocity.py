import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import csv
from datetime import datetime
import os

class VelocityLogger(Node):
    def __init__(self):
        super().__init__('velocity_logger')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.velocities = [round(v, 3) for v in self._frange(0.22, 0.00, -0.005)]
        self.index = 0
        self.latest_odom = 0.0

        self.csv_file = self.create_log_file()
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('🚀 Bắt đầu gửi vận tốc và ghi log...')

    def _frange(self, start, stop, step):
        while start >= stop:
            yield start
            start += step

    def create_log_file(self):
        folder = os.path.expanduser('/home/nguyehtt/turtlebot3_ws/src/target_tracking/target_tracking/velocity_logs')
        os.makedirs(folder, exist_ok=True)
        filename = f'velocity_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        filepath = os.path.join(folder, filename)
        csv_file = open(filepath, mode='w', newline='')
        self.csv_writer = csv.writer(csv_file)
        self.csv_writer.writerow(['Time (s)', 'Commanded Velocity (m/s)', 'Measured Velocity (m/s)'])
        self.get_logger().info(f'📄 Ghi log vào: {filepath}')
        return csv_file

    def odom_callback(self, msg):
        self.latest_odom = msg.twist.twist.linear.x

    def timer_callback(self):
        if self.index >= len(self.velocities):
            self.stop_robot()
            self.csv_file.close()
            self.get_logger().info('✅ Hoàn tất ghi log và dừng robot.')
            rclpy.shutdown()
            return

        v_cmd = self.velocities[self.index]
        msg = Twist()
        msg.linear.x = v_cmd
        self.cmd_pub.publish(msg)

        v_measured = self.latest_odom
        timestamp = self.get_clock().now().seconds_nanoseconds()[0]
        self.csv_writer.writerow([timestamp, v_cmd, v_measured])

        self.get_logger().info(f'⚙️ Gửi: {v_cmd:.3f} m/s | Đo: {v_measured:.3f} m/s')
        self.index += 1

    def stop_robot(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VelocityLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
