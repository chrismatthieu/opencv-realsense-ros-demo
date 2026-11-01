"""Integrate /cmd_vel into a visible robot. A burst of zero twists latches a stop."""

import json
import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def _yaw_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def _marker(ns, marker_id, marker_type, frame):
    marker = Marker()
    marker.header.frame_id = frame
    marker.ns = ns
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


class StageBase(Node):
    def __init__(self):
        super().__init__('stage_base')
        self.declare_parameter('target_distance_m', 1.0)
        self.declare_parameter('lateral_scale_m', 0.6)
        self.declare_parameter('command_timeout_sec', 0.5)
        self.declare_parameter('estop_zeros', 4)
        self.declare_parameter('estop_window_sec', 0.04)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear = 0.0
        self.angular = 0.0
        self.last_cmd = None
        self.last_tick = None
        self.zero_times = []
        self.last_nonzero = None
        self.halted = False
        self.mode = 'report'
        self.latched = False
        self.found = False
        self.bearing = None
        self.distance_m = None

        self.tf_broadcaster = TransformBroadcaster(self)
        self.marker_pub = self.create_publisher(MarkerArray, '/demo/markers', 10)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 20)
        self.create_subscription(String, '/demo/mode', self._on_mode, 10)
        self.create_subscription(String, '/demo/status', self._on_status, 10)
        self.create_service(Empty, '/demo/reset_pose', self._on_reset)
        self.create_timer(0.05, self._tick)
        self.get_logger().info('stage robot ready in the map frame')

    def _on_reset(self, _request, response):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.linear = 0.0
        self.angular = 0.0
        self.halted = False
        self.get_logger().info('robot pose reset to the map origin')
        return response

    def _on_mode(self, msg):
        if self.halted:
            self.get_logger().info('estop cleared by a new mode command')
        self.halted = False
        self.zero_times = []

    def _on_status(self, msg):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.mode = status.get('mode') or 'report'
        self.latched = bool(status.get('latched'))
        self.found = bool(status.get('found'))
        self.bearing = status.get('bearing')
        self.distance_m = status.get('distance_m')

    def _on_cmd(self, msg):
        now = time.monotonic()
        stopped = abs(msg.linear.x) < 1e-4 and abs(msg.angular.z) < 1e-4
        window = float(self.get_parameter('estop_window_sec').value)
        needed = int(self.get_parameter('estop_zeros').value)
        if stopped:
            self.zero_times.append(now)
            self.zero_times = [stamp for stamp in self.zero_times if now - stamp <= window]
            moving_recently = (
                self.last_nonzero is not None and (now - self.last_nonzero) < 0.5
            )
            if moving_recently and len(self.zero_times) >= needed and not self.halted:
                self.halted = True
                self.linear = 0.0
                self.angular = 0.0
                self.get_logger().warn('estop latched — publish /demo/mode to move again')
                return
        else:
            self.zero_times = []
            self.last_nonzero = now
        if self.halted:
            return
        self.linear = msg.linear.x
        self.angular = msg.angular.z
        self.last_cmd = now

    def _body_color(self):
        if self.halted:
            return (0.85, 0.15, 0.15)
        if self.latched:
            return (0.95, 0.75, 0.15)
        if self.mode == 'follow':
            return (0.10, 0.65, 0.35)
        if self.mode == 'approach':
            return (0.90, 0.50, 0.10)
        return (0.15, 0.40, 0.85)

    def _label(self):
        if self.halted:
            return 'HALTED'
        if self.latched:
            return 'STOPPED AT 1 m'
        if self.mode == 'preview':
            return 'COMMAND'
        return self.mode.upper()

    def _goal_in_map(self):
        if self.latched:
            return self.x, self.y, True
        if not self.found or self.mode not in ('follow', 'approach'):
            return None
        if self.distance_m is None or self.bearing is None:
            return None
        ahead = float(self.distance_m) - float(self.get_parameter('target_distance_m').value)
        lateral = -float(self.bearing) * float(self.get_parameter('lateral_scale_m').value)
        gx = self.x + ahead * math.cos(self.yaw) - lateral * math.sin(self.yaw)
        gy = self.y + ahead * math.sin(self.yaw) + lateral * math.cos(self.yaw)
        return gx, gy, False

    def _tick(self):
        now = time.monotonic()
        if self.last_tick is None:
            dt = 0.05
        else:
            dt = min(0.1, max(0.0, now - self.last_tick))
        self.last_tick = now

        moving = self.last_cmd is not None and (now - self.last_cmd) <= float(
            self.get_parameter('command_timeout_sec').value
        )
        linear = self.linear if moving and not self.halted else 0.0
        angular = self.angular if moving and not self.halted else 0.0
        self.yaw += angular * dt
        self.x += linear * math.cos(self.yaw) * dt
        self.y += linear * math.sin(self.yaw) * dt

        stamp = self.get_clock().now().to_msg()
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        qx, qy, qz, qw = _yaw_quaternion(self.yaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)
        self.marker_pub.publish(self._markers(stamp))

    def _markers(self, stamp):
        red, green, blue = self._body_color()
        body = _marker('robot', 0, Marker.CUBE, 'base_link')
        body.header.stamp = stamp
        body.pose.position.x = 0.05
        body.scale.x = 0.42
        body.scale.y = 0.28
        body.scale.z = 0.12
        body.color.r = red
        body.color.g = green
        body.color.b = blue
        body.color.a = 1.0

        nose = _marker('robot', 1, Marker.ARROW, 'base_link')
        nose.header.stamp = stamp
        nose.pose.position.x = 0.16
        nose.scale.x = 0.28
        nose.scale.y = 0.06
        nose.scale.z = 0.06
        nose.color.r = 1.0
        nose.color.g = 1.0
        nose.color.b = 1.0
        nose.color.a = 1.0

        label = _marker('robot', 2, Marker.TEXT_VIEW_FACING, 'base_link')
        label.header.stamp = stamp
        label.pose.position.z = 0.35
        label.scale.z = 0.18
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = self._label()

        goal_spec = self._goal_in_map()
        goal = _marker('goal', 3, Marker.CYLINDER, 'map')
        goal.header.stamp = stamp
        if goal_spec is None:
            goal.action = Marker.DELETE
        else:
            gx, gy, arrived = goal_spec
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.position.z = 0.02
            goal.scale.x = 0.22
            goal.scale.y = 0.22
            goal.scale.z = 0.04
            goal.color.a = 0.9
            if arrived:
                goal.color.r = 0.95
                goal.color.g = 0.75
                goal.color.b = 0.15
            else:
                goal.color.r = 0.15
                goal.color.g = 0.85
                goal.color.b = 0.45

        array = MarkerArray()
        array.markers = [body, nose, label, goal]
        return array


def main(args=None):
    rclpy.init(args=args)
    node = StageBase()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
