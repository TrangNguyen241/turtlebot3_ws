import math
import numpy as np

# Function of calculating angle of robot from data of position from odom/rigidbod
def quaternion2euler(qx, qy, qz, qw):
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

def odometry_callback(msg):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    q = orientation
    qw, qx, qy, qz = q.w, q.x, q.y, q.z
    angles = quaternion2euler(qx, qy, qz, qw)
    theta = (angles[-1])% (2 * math.pi)
    x = position.x
    y = position.y
    return x, y, theta


    