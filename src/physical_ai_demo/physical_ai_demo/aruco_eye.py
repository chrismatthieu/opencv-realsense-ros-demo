"""Publish an OpenCV view of ArUco marker 23, plus its pose and range."""

import json
import os

# OpenCV's Qt build has no Wayland plugin. Force the X11 backend before Qt starts.
os.environ.pop('WAYLAND_DISPLAY', None)
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from physical_ai_demo.control import normalize_mode
from physical_ai_demo.vision import (
    draw_menu_bar,
    make_detector,
    measure_frame,
    ros_image_to_array,
)


def bgr_to_ros_image(image, header):
    image = np.ascontiguousarray(image)
    msg = Image()
    msg.header = header
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = int(msg.width * 3)
    msg.data = image.tobytes()
    return msg


def _stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class ArucoEye(Node):
    def __init__(self):
        super().__init__('aruco_eye')
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('raw_depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('target_id', 23)
        self.declare_parameter('marker_size_m', 0.12)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('depth_stamp_slack_sec', 0.20)

        _dictionary, self.detector = make_detector()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth_slots = {}
        self.latest = {
            'found': False,
            'id': None,
            'bearing': None,
            'distance_m': None,
            'depth_m': None,
            'pnp_z_m': None,
            'range_source': 'none',
            'x_m': None,
            'y_m': None,
            'z_m': None,
            'frame_id': '',
            'center_depth_m': None,
            'depth_aligned': False,
        }
        self.latest_pose = None
        self.color_frame = ''
        self.warned_info = False
        self.mode = 'report'
        self.pending_mode = None
        self.gui = False

        self.image_pub = self.create_publisher(Image, '/demo/debug_image', 1)
        self.mode_pub = self.create_publisher(String, '/demo/mode', 10)
        self.create_subscription(String, '/demo/mode', self._on_mode, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/demo/marker_pose', 10)
        self.marker_pub = self.create_publisher(String, '/demo/marker', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # This D585 driver publishes reliable images. Best-effort subscribers
        # connect, but matching the publisher avoids dropped depth frames.
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            Image, self.get_parameter('color_topic').value, self._on_color, image_qos
        )
        self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            lambda msg: self._store_depth('aligned', msg),
            image_qos,
        )
        self.create_subscription(
            Image,
            self.get_parameter('raw_depth_topic').value,
            lambda msg: self._store_depth('raw', msg),
            image_qos,
        )
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self._on_info, image_qos
        )
        self.create_timer(0.2, self._publish_json)
        self._open_menu()
        self.get_logger().info('aruco_eye watching for marker %d' % self.get_parameter('target_id').value)

    def _open_menu(self):
        if not os.environ.get('DISPLAY'):
            self.get_logger().warn('no DISPLAY; menu clicks are unavailable')
            return
        try:
            import tkinter as tk
        except ImportError as exc:
            self.get_logger().warn('menu window unavailable: %s' % exc)
            return
        self.root = tk.Tk()
        self.root.title('Demo menu')
        self.root.attributes('-topmost', True)
        self.root.geometry('480x200+40+40')
        self.root.configure(bg='#181818')
        self.detect_button = tk.Button(
            self.root, command=lambda: self._queue_mode('report'),
            font=('sans-serif', 16), fg='white', relief='flat',
        )
        self.follow_button = tk.Button(
            self.root, command=lambda: self._queue_mode('follow'),
            font=('sans-serif', 16), fg='white', relief='flat',
        )
        self.detect_button.pack(fill='both', expand=True, padx=8, pady=(8, 4))
        self.follow_button.pack(fill='both', expand=True, padx=8, pady=(4, 8))
        self.root.bind('1', lambda _event: self._queue_mode('report'))
        self.root.bind('2', lambda _event: self._queue_mode('follow'))
        self.gui = True

    def _queue_mode(self, mode):
        self.pending_mode = mode

    def _on_mode(self, msg):
        mode = normalize_mode(msg.data)
        if mode in ('report', 'follow', 'approach'):
            self.mode = mode

    def _select_mode(self, mode):
        if mode not in ('report', 'follow') or mode == self.mode:
            return
        self.mode = mode
        message = String()
        message.data = mode
        self.mode_pub.publish(message)
        self.get_logger().info('menu selected %s' % mode)

    def _on_info(self, msg):
        import numpy as np

        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def _store_depth(self, key, msg):
        try:
            image = ros_image_to_array(msg)
        except RuntimeError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return
        self.depth_slots[key] = (image, msg.header.stamp, msg.encoding)

    def _depth_for(self, color_stamp, color_shape):
        slack = float(self.get_parameter('depth_stamp_slack_sec').value)
        scale_param = float(self.get_parameter('depth_scale').value)
        # The D585 ROS align filter advertises a topic and then publishes
        # nothing. Prefer it when frames actually arrive, and otherwise use
        # the rectified depth image at the same pixel.
        for key in ('aligned', 'raw'):
            slot = self.depth_slots.get(key)
            if slot is None:
                continue
            image, stamp, encoding = slot
            if abs(_stamp_seconds(color_stamp) - _stamp_seconds(stamp)) > slack:
                continue
            if image.shape[:2] != color_shape[:2]:
                image = cv2.resize(
                    image,
                    (color_shape[1], color_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            scale = 1.0 if encoding == '32FC1' else scale_param
            return image, scale, key == 'aligned'
        return None, scale_param, False

    def _on_color(self, msg):
        bgr = ros_image_to_array(msg)
        if self.camera_matrix is None and not self.warned_info:
            self.get_logger().warn('waiting for camera_info; drawing corners without a pose')
            self.warned_info = True
        elif self.camera_matrix is not None and self.warned_info:
            self.get_logger().info('camera_info received')
            self.warned_info = False
        depth, scale, depth_aligned = self._depth_for(msg.header.stamp, bgr.shape)
        annotated, measured = measure_frame(
            bgr,
            self.detector,
            int(self.get_parameter('target_id').value),
            float(self.get_parameter('marker_size_m').value),
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            depth=depth,
            depth_scale=scale,
        )
        self.color_frame = msg.header.frame_id
        self.latest = {
            'found': measured['found'],
            'id': measured['id'],
            'bearing': measured['bearing'],
            'distance_m': measured['distance_m'],
            'depth_m': measured['depth_m'],
            'pnp_z_m': measured['pnp_z_m'],
            'range_source': measured['range_source'],
            'x_m': measured['x_m'],
            'y_m': measured['y_m'],
            'z_m': measured['z_m'],
            'frame_id': msg.header.frame_id,
            'center_depth_m': measured.get('center_depth_m'),
            'depth_aligned': depth_aligned,
        }
        self.latest_pose = measured if measured['found'] and measured['quat_xyzw'] else None
        if self.latest_pose is not None:
            self._publish_pose(msg.header)

        draw_menu_bar(annotated, self.mode, measured['found'], measured['distance_m'])
        self._pump_menu(measured)
        self.image_pub.publish(bgr_to_ros_image(annotated, msg.header))
        self._publish_json()

    def _pump_menu(self, measured):
        if not self.gui:
            return
        distance = measured['distance_m']
        distance_text = (
            f'{distance:.2f} m' if measured['found'] and distance is not None else 'hold up marker 23'
        )
        self.detect_button.configure(
            text=f'1   DETECT\n{distance_text}',
            bg='#2e7d32' if self.mode == 'report' else '#303030',
        )
        self.follow_button.configure(
            text='2   FOLLOW\nrobot turns and drives to 1 m',
            bg='#2e7d32' if self.mode == 'follow' else '#303030',
        )
        self.root.update_idletasks()
        self.root.update()
        pending = self.pending_mode
        self.pending_mode = None
        if pending:
            self._select_mode(pending)

    def _publish_pose(self, header):
        quat = self.latest_pose['quat_xyzw']
        pose = PoseStamped()
        pose.header = header
        pose.pose.position.x = self.latest_pose['x_m']
        pose.pose.position.y = self.latest_pose['y_m']
        pose.pose.position.z = self.latest_pose['z_m']
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose)

        transform = TransformStamped()
        transform.header = header
        transform.child_frame_id = 'marker_23'
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z
        transform.transform.rotation = pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def _publish_json(self):
        payload = dict(self.latest)
        payload['stamp_sec'] = self.get_clock().now().nanoseconds / 1e9
        message = String()
        message.data = json.dumps(payload)
        self.marker_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoEye()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
