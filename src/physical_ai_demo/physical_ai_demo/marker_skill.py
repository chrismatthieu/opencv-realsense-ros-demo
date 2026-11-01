"""Turn a marker measurement into /cmd_vel. The agent only selects the mode."""

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from physical_ai_demo.control import compute_command, normalize_mode, slew


class MarkerSkill(Node):
    def __init__(self):
        super().__init__('marker_skill')
        self.declare_parameter('target_distance_m', 1.0)
        self.declare_parameter('max_linear', 0.25)
        self.declare_parameter('max_angular', 0.6)
        self.declare_parameter('track_hold_sec', 0.35)
        self.declare_parameter('stale_sec', 0.5)
        self.declare_parameter('rate_hz', 20.0)

        self.mode = 'report'
        self.latched = False
        self.sample = None
        self.sample_time = None
        self.track = None
        self.track_time = None
        self.linear = 0.0
        self.angular = 0.0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/demo/status', 10)
        self.create_subscription(String, '/demo/marker', self._on_marker, 10)
        self.create_subscription(String, '/demo/mode', self._on_mode, 10)
        period = 1.0 / float(self.get_parameter('rate_hz').value)
        self.create_timer(period, self._tick)
        self.get_logger().info('marker_skill in report mode')

    def _on_mode(self, msg):
        mode = normalize_mode(msg.data)
        if mode is None:
            self.get_logger().warn('unknown mode %r (use report, follow, or approach)' % msg.data)
            return
        if mode != self.mode:
            self.get_logger().info('mode %s -> %s' % (self.mode, mode))
            self.latched = False
        self.mode = mode

    def _on_marker(self, msg):
        try:
            sample = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn('ignored non-JSON marker message')
            return
        self.sample = sample
        self.sample_time = self.get_clock().now()
        if sample.get('found'):
            self.track = sample
            self.track_time = self.sample_time

    def _tracking(self, now):
        if self.sample is None or self.sample_time is None:
            return False, None, None
        age = (now - self.sample_time).nanoseconds / 1e9
        if age > float(self.get_parameter('stale_sec').value):
            return False, None, None
        hold = float(self.get_parameter('track_hold_sec').value)
        if self.track is not None and self.track_time is not None:
            held = (now - self.track_time).nanoseconds / 1e9
            if held <= hold:
                return True, _number(self.track.get('bearing')), _number(self.track.get('distance_m'))
        return False, None, None

    def _tick(self):
        now = self.get_clock().now()
        found, bearing, distance = self._tracking(now)
        dt = 1.0 / float(self.get_parameter('rate_hz').value)
        linear, angular, self.latched = compute_command(
            self.mode,
            found,
            bearing,
            distance,
            latched=self.latched,
            target_distance=float(self.get_parameter('target_distance_m').value),
            max_linear=float(self.get_parameter('max_linear').value),
            max_angular=float(self.get_parameter('max_angular').value),
        )
        self.linear = slew(self.linear, linear, 0.8 * dt)
        self.angular = slew(self.angular, angular, 2.0 * dt)
        if abs(self.linear) < 1e-3:
            self.linear = 0.0
        if abs(self.angular) < 1e-3:
            self.angular = 0.0

        twist = Twist()
        twist.linear.x = self.linear
        twist.angular.z = self.angular
        self.cmd_pub.publish(twist)

        marker_id = None if self.sample is None else self.sample.get('id')
        status = {
            'mode': self.mode,
            'latched': self.latched,
            'found': found,
            'id': marker_id,
            'bearing': bearing,
            'distance_m': distance,
            'linear_x': self.linear,
            'angular_z': self.angular,
        }
        message = String()
        message.data = json.dumps(status)
        self.status_pub.publish(message)


def _number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(args=None):
    rclpy.init(args=args)
    node = MarkerSkill()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
