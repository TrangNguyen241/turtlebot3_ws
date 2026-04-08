# Cost-effective control design for wheeled robots with safety and stability guarantees

**Author:** Huynh Thao Trang NGUYEN | LCIS (Valence, FR), March 2025  

This project implements and evaluates several control strategies for **target tracking** and **trajectory tracking** with Turtlebot3 in simulation (ROS2 + Gazebo) and in real experiments (Qualisys camera system). It also includes **obstacle avoidance** controllers using **CLF-CBF** and **CLF-ERG**.

---

## Platform
- **Simulation:** ROS2 Humble + Gazebo + RViz2  
- **Robot:** Turtlebot3 Burger  
- **Experiment:** Qualisys Motion Capture Camera System  

---

## Objectives
- Design and evaluate different controllers for target and trajectory tracking:
  - Saturated control  
  - CLF-based control  
  - Implicit MPC  
  - Explicit MPC  
- Implement obstacle avoidance with:
  - CLF-CBF (Control Lyapunov Function & Control Barrier Functions)
  - CLF-ERG (Control Lyapunov Function & Explicit Reference Governor)

---

## Prerequisites
- ROS2 Humble installed on your computer.  
- Install **Turtlebot3 simulation packages**:  
  - [Turtlebot3 Overview](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)  
  - [Quick start & simulation setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)  

---

# Target Tracking using Different Methods

## Simulation in Gazebo and RViz

### Step 1: Launch Gazebo environment
```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

### Step 2: Run controller code with desired method & target (Target Tracking)
```bash
ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=<controller> -p target:="<x, y>"
```
**Example (implicit MPC, target = [0.4, 1.0]):**
```bash
ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=implicit_mpc -p target:="[0.4, 1.0]"
```

**Available controllers:**
- `saturated` → Saturated Control
- `sat_lyapunov` → CLF-based Control (Nominal control $u_d$: Saturated control) 
- `lqr_lyapunov` → CLF-based Control (Nominal control $u_d$: LQR)
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

# Trajectory Tracking 

After launching Gazebo (Step 1), you can run trajectory tracking examples for Turtlebot3 using each controller’s script.

> Workspace layout referenced below: `turtlebot3_ws/src/<package>/<exec>`

### Saturated Control
```bash
ros2 run saturated_control saturated_traj
```
Runs `saturated_traj.py` in `turtlebot3_ws/src/saturated_control/saturated_control`.

### CLF-based Control
```bash
ros2 run lya_lqr_control lya_lqr_traj
```
Runs `lya_lqr_traj.py` in `turtlebot3_ws/src/lya_lqr_control/lya_lqr_control`.

### Explicit MPC
```bash
ros2 run exp_mpc exp_mpc_trajectory
```
Runs `exp_mpc_trajectory.py` in `turtlebot3_ws/src/exp_mpc/exp_mpc`.

### Implicit MPC
```bash
ros2 run imp_mpc imp_mpc_traj
```
Runs `imp_mpc_traj.py` in `turtlebot3_ws/src/imp_mpc/imp_mpc`.

---

## Obstacle Avoidance (CLF-CBF and CLF-ERG)

### Step 1: Launch Gazebo environment with Turtlebot3 and obstacles
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Step 2A: Reach a target while avoiding obstacles with **CLF-CBF**
```bash
ros2 run cbf_avoid_colli cbf_avoid_colli
```
Runs `cbf_avoid_colli.py` in `turtlebot3_ws/src/cbf_avoid_colli/cbf_avoid_colli`.

### Step 2B: Reach a target while avoiding obstacles with **CLF-ERG**
```bash
ros2 run obs_avoid ref_gov_target
```
Runs `ref_gov_target.py` in `turtlebot3_ws/src/obs_avoid/obs_avoid`.

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
