### TARGET TRACKING USING DIFFERENT METHODS ###
* Author: Huynh Thao Trang NGUYEN | LCIS (Valence, FR), March, 2025 *

# Platform used: 
This task is simualated and operated by ROS2, Gazebo. Robot used is Turtlebot3. For experimental test, using Qualisys camera system. 

# Objectives:
++ Designing a saturated control, LQR + Lyapunov control, implicit MPC, explicit MPC for target tracking

# Prerequisites
++ To run the code, you need to have ROS2 HUMBLE in your computer. 
++ Read and understand how the Turtlebot3 works via this link: https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/, and install simulation package following this link step by step. 
++ For experimental test, follow the guide of setting up for Turtlebot3 by this link: https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/

# SIMULATION IN GAZEBO AND RVIZ
- Step 1: In one terminal, run the Gazebo environment:
    $ ros2 launch turtlebot3_gazebo empty_world.launch.py 

- Step 2: In another terminal, run the control code with desired type of controller and target:
    $ ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=<name of controller> -p target:="<target>"
    For example: In case of using implicit MPC and target = [0.4, 1.0]:
    $ ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=implicit_mpc -p target:="[0.4, 1.0]"

    Different controllers are repensented by these parameters:
        + Saturated_control: 'saturated'
        + LQR + Lyapunov control: 'lqr_lyapunov'
        + Explicit MPC: 'explicit_mpc'
        + Implicit MPC: 'implicit_mpc'

- Step 3: We can also observe how robot tracks the target in Rviz:
    $ rviz2
    In "Displays" table, click on "Add", and choose "RobotModel", "Path", "Marker". Set "Fixed Frame" is "odom".
        + In "RobotModel": choose /robot_description at "Description Topic" --> visualize Turtlebot3 mdoel
        + In "Path": choose /actual_path at "Topic" --> visualize the trajectory of robot
        + In "Marker": be sure to be chosen --> visualize target point

# EXPERIMENTAL TEST WITH QUALISYS CAMERA SYSTEM:
- Step 1: Launch Qualisys camera system node to receive position of 4 robots (topic /rigid_bodies)
    $ ros2 launch qualisys_driver qualisys.launch.py
- Step 2: We will communicate with Turtlebot3 via Wifi (SSH connection). Configure a Wifi hotspot in your computer:
    $ nmcli con add type wifi ifname <ip of hotspot wireless interface> con-name <name> autoconnect yes ssid <name SSID> ap ipv4.method shared
    In my case:
    $ nmcli con add type wifi ifname wlp0s20f3 con-name Co4Sys autoconnect yes ssid Co4Sys 802-11-wireless.mode ap ipv4.method shared
- Step 2: Turn on the hotspot we created before:
    $ nmcli con up Co4Sys
- Step 3: Check connection between your laptop and turtlebot:
    $ ip n
    !Attention: Turtlebot3 must be "reachable", otherwise you need to wait
- Step 4: Connect Turtlebot3 by running:
    $ ssh ubuntu@<ID of turtlebot>
    $ ros2 launch turtlebot3 <bringup launch file>
    In my case:
    $ ssh ubuntu@10.42.0.136
    $ ros2 launch turtlebot32 turtlebot32_bringup.launch.py
- Step 5: Run the controller node by running this command in another terminal:
    $ ros2 run target_tracking target_tracking_controllers --ros-args -p controller_type:=<name of controller> -p target:="<target>"
