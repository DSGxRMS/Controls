from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package="eufs_sim", executable="eufs_launcher"),
        Node(package="control", executable="path_planner_node", output="screen"),
        Node(package="control", executable="control_loop", output="screen"),
        Node(package="control", executable="controller_plotter", output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", "local_path.rviz"])
    ])

#For launching all nodes

#ros2 launch your_pkg full_system.launch.py


