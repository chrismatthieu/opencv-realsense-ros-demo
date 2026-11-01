"""RealSense, OpenCV marker eye, motion skill, and the virtual robot."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    demo_share = get_package_share_directory('physical_ai_demo')
    rviz_config = os.path.join(demo_share, 'rviz', 'demo.rviz')

    marker_size = LaunchConfiguration('marker_size_m')
    target_distance = LaunchConfiguration('target_distance_m')

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('marker_size_m', default_value='0.12'),
        DeclareLaunchArgument('target_distance_m', default_value='1.0'),
        DeclareLaunchArgument(
            'serial_no',
            default_value='_541522320147',
            description='RealSense serial. Leading underscore keeps it a string.',
        ),
        Node(
            package='realsense2_camera',
            namespace='camera',
            name='camera',
            executable='realsense2_camera_node',
            output='screen',
            parameters=[{
                'serial_no': LaunchConfiguration('serial_no'),
                'rgb_camera.color_profile': '640,480,30',
                'rgb_camera.color_format': 'RGB8',
                'depth_module.depth_profile': '640,480,30',
                'enable_infra1': False,
                'enable_infra2': False,
                'align_depth.enable': True,
                'enable_sync': False,
                'pointcloud.enable': False,
                'initial_reset': False,
            }],
        ),
        Node(
            package='physical_ai_demo',
            executable='aruco_eye',
            name='aruco_eye',
            output='screen',
            parameters=[{
                'marker_size_m': ParameterValue(marker_size, value_type=float),
                'target_id': 23,
            }],
        ),
        Node(
            package='physical_ai_demo',
            executable='marker_skill',
            name='marker_skill',
            output='screen',
            parameters=[{
                'target_distance_m': ParameterValue(target_distance, value_type=float),
            }],
        ),
        Node(
            package='physical_ai_demo',
            executable='stage_base',
            name='stage_base',
            output='screen',
            parameters=[{
                'target_distance_m': ParameterValue(target_distance, value_type=float),
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ])
