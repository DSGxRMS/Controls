#!/usr/bin/env python3
"""
Launch file for the improved control system with various configurations.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for control system."""
    
    # Declare launch arguments
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='Odometry topic name'
    )
    
    cmd_topic_arg = DeclareLaunchArgument(
        'cmd_topic',
        default_value='/cmd',
        description='Command topic name'
    )
    
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='ackermann',
        description='Control mode: ackermann or twist'
    )
    
    path_csv_arg = DeclareLaunchArgument(
        'path_csv',
        default_value='/root/rosws/ctrl_main/src/control/control/pathpoints.csv',
        description='Path to CSV file containing waypoints'
    )
    
    scaling_factor_arg = DeclareLaunchArgument(
        'scaling_factor',
        default_value='1.0',
        description='Scaling factor for path coordinates'
    )
    
    loop_arg = DeclareLaunchArgument(
        'loop',
        default_value='false',
        description='Whether the path is a loop'
    )
    
    hz_arg = DeclareLaunchArgument(
        'hz',
        default_value='50.0',
        description='Control loop frequency'
    )
    
    path_offset_x_arg = DeclareLaunchArgument(
        'path_offset_x',
        default_value='0.0',
        description='Path offset in X direction'
    )
    
    path_offset_y_arg = DeclareLaunchArgument(
        'path_offset_y',
        default_value='0.0',
        description='Path offset in Y direction'
    )
    
    enable_startup_mode_arg = DeclareLaunchArgument(
        'enable_startup_mode',
        default_value='true',
        description='Enable startup assistance mode'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug logging'
    )
    
    # Control node
    control_node = Node(
        package='control',
        executable='run_control',
        name='control_node',
        parameters=[{
            'odom_topic': LaunchConfiguration('odom_topic'),
            'cmd_topic': LaunchConfiguration('cmd_topic'),
            'mode': LaunchConfiguration('mode'),
            'path_csv': LaunchConfiguration('path_csv'),
            'scaling_factor': LaunchConfiguration('scaling_factor'),
            'loop': LaunchConfiguration('loop'),
            'hz': LaunchConfiguration('hz'),
            'path_offset_x': LaunchConfiguration('path_offset_x'),
            'path_offset_y': LaunchConfiguration('path_offset_y'),
            'enable_startup_mode': LaunchConfiguration('enable_startup_mode'),
        }],
        output='screen',
        arguments=['--ros-args', '--log-level', 'DEBUG'] if LaunchConfiguration('debug') else []
    )
    
    return LaunchDescription([
        odom_topic_arg,
        cmd_topic_arg,
        mode_arg,
        path_csv_arg,
        scaling_factor_arg,
        loop_arg,
        hz_arg,
        path_offset_x_arg,
        path_offset_y_arg,
        enable_startup_mode_arg,
        debug_arg,
        control_node
    ])


if __name__ == '__main__':
    generate_launch_description()