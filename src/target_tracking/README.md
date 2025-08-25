# Target Tracking using Different Methods

**Author:** Huynh Thao Trang NGUYEN | LCIS (Valence, FR), March 2025  

This project implements and evaluates several control strategies for **target tracking** with Turtlebot3 in simulation (ROS2 + Gazebo) and in real experiments (Qualisys camera system).  

---

## Platform
- **Simulation:** ROS2 Humble + Gazebo + RViz2  
- **Robot:** Turtlebot3 Burger  
- **Experiment:** Qualisys Motion Capture Camera System  

---

## Objectives
- Design and evaluate different controllers for target tracking:
  - Saturated control  
  - LQR + Lyapunov control  
  - Implicit MPC  
  - Explicit MPC  

---

## Prerequisites
- ROS2 Humble installed on your computer.  
- Install **Turtlebot3 simulation packages**:  
  - [Turtlebot3 Overview](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)  
  - [Quick start & simulation setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)  

---

## Simulation in Gazebo and RViz

### Step 1: Launch Gazebo environment
```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

### Step 2: Run controller code with desired method & target
```bash
ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=<controller> -p target:="<x, y>"
```
**Example (implicit MPC, target = [0.4, 1.0]):**
```bash
ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=implicit_mpc -p target:="[0.4, 1.0]"
```

**Available controllers:**
- `saturated` → Saturated Control  
- `lqr_lyapunov` → LQR + Lyapunov Control  
- `explicit_mpc` → Explicit MPC  
- `implicit_mpc` → Implicit MPC  

### Step 3: Visualize in RViz
```bash
rviz2
```
- Set **Fixed Frame** → `odom`  
- Add displays:
  - **RobotModel** → `/robot_description` (visualize Turtlebot3 model)  
  - **Path** → `/actual_path` (visualize robot trajectory)  
  - **Marker** → target point  

---

## Experimental Test with Qualisys

### Step 1: Launch Qualisys driver
```bash
ros2 launch qualisys_driver qualisys.launch.py
```

### Step 2: Configure WiFi hotspot on your laptop
```bash
nmcli con add type wifi ifname <wireless_interface> con-name <hotspot_name> autoconnect yes ssid <SSID> ap ipv4.method shared
```
**Example:**
```bash
nmcli con add type wifi ifname wlp0s20f3 con-name Co4Sys autoconnect yes ssid Co4Sys 802-11-wireless.mode ap ipv4.method shared
```
Turn on the hotspot:
```bash
nmcli con up Co4Sys
```

### Step 3: Check Turtlebot3 connection
```bash
ip n
```
⚠️ Turtlebot3 must be **reachable**, otherwise wait until connection is established.  

### Step 4: Connect to Turtlebot3 via SSH
```bash
ssh ubuntu@<turtlebot_ip>
ros2 launch turtlebot3 <bringup_launch_file>
```
**Example:**
```bash
ssh ubuntu@10.42.0.136
ros2 launch turtlebot32 turtlebot32_bringup.launch.py
```

### Step 5: Run controller on your laptop
```bash
ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=<controller> -p target:="<x, y>"
```

---

## Repository
Code is available at: [turtlebot3_ws](https://github.com/TrangNguyen241/turtlebot3_ws)

---

## Citation (IEEE style)
> TrangNguyen241, “turtlebot3_ws,” GitHub repository, https://github.com/TrangNguyen241/turtlebot3_ws, 2025.

---

## Notes
- This repository ignores build artifacts by default:
  - `build/`, `install/`, `log/`, `*.zip`
- After building, remember to source:
```bash
source install/setup.bash
```
