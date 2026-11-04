"""Turn a marker measurement into /cmd_vel. The agent only selects the mode."""

import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from physical_ai_demo.control import (
    approach_phase,
    approach_should_resume,
    compute_command,
    normalize_mode,
    slew,
)


class MarkerSkill(Node):
    def __init__(self):
        super().__init__('marker_skill')
        self.declare_parameter('target_distance_m', 1.0)
        self.declare_parameter('max_linear', 0.25)
        self.declare_parameter('max_angular', 0.6)
        self.declare_parameter('track_hold_sec', 0.35)
        self.declare_parameter('approach_track_hold_sec', 2.0)
        self.declare_parameter('stale_sec', 0.5)
        self.declare_parameter('approach_stale_sec', 2.5)
        self.declare_parameter('approach_resume_sec', 5.0)
        self.declare_parameter('rate_hz', 20.0)

        self.mode = 'report'
        self.latched = False
        self.latch_since = None
        self.approach_armed = True
        self.approach_cycle = 0
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

    def _reset_approach_loop(self):
        self.latched = False
        self.latch_since = None
        self.approach_armed = True

    def _on_mode(self, msg):
        mode = normalize_mode(msg.data)
        if mode is None:
            self.get_logger().warn('unknown mode %r (use report, preview, follow, or approach)' % msg.data)
            return
        if mode != self.mode:
            self.get_logger().info('mode %s -> %s' % (self.mode, mode))
        # Re-publishing approach always restarts the hunt (clears a stale latch).
        if mode == 'approach':
            self._reset_approach_loop()
        elif mode != self.mode:
            self._reset_approach_loop()
            self.approach_cycle = 0
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
        # Act 4 waits for the operator to raise the marker; keep the last pose
        # longer so a brief occlusion does not look like the mission gave up.
        if self.mode == 'approach':
            stale = float(self.get_parameter('approach_stale_sec').value)
            hold = float(self.get_parameter('approach_track_hold_sec').value)
        else:
            stale = float(self.get_parameter('stale_sec').value)
            hold = float(self.get_parameter('track_hold_sec').value)
        if age > stale:
            return False, None, None
        if self.track is not None and self.track_time is not None:
            held = (now - self.track_time).nanoseconds / 1e9
            if held <= hold:
                return True, _number(self.track.get('bearing')), _number(self.track.get('distance_m'))
        return False, None, None

    def _tick(self):
        now = self.get_clock().now()
        found, bearing, distance = self._tracking(now)
        resume_after = float(self.get_parameter('approach_resume_sec').value)
        resume_in_sec = None

        if self.mode == 'approach' and self.latched:
            if self.latch_since is None:
                self.latch_since = now
            latch_age = (now - self.latch_since).nanoseconds / 1e9
            resume_in_sec = max(0.0, resume_after - latch_age)
            if approach_should_resume(latch_age, resume_after):
                self.latched = False
                self.latch_since = None
                self.approach_armed = False
                self.approach_cycle += 1
                resume_in_sec = None
                self.get_logger().info(
                    'approach cycle %d — hold done, looking for marker again' % self.approach_cycle
                )

        dt = 1.0 / float(self.get_parameter('rate_hz').value)
        linear, angular, latched = compute_command(
            self.mode,
            found,
            bearing,
            distance,
            latched=self.latched,
            target_distance=float(self.get_parameter('target_distance_m').value),
            max_linear=float(self.get_parameter('max_linear').value),
            max_angular=float(self.get_parameter('max_angular').value),
        )
        # After a hold, stay unlatched until the marker leaves the frame so the
        # next show-and-approach is a fresh surprise for the audience.
        if self.mode == 'approach' and not self.approach_armed:
            latched = False
            if not found:
                self.approach_armed = True
                self.get_logger().info('approach re-armed — waiting for marker 23')
        self.latched = latched
        if self.latched and self.latch_since is None:
            self.latch_since = now
        if not self.latched and self.approach_armed:
            self.latch_since = None
            resume_in_sec = None

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
        phase = approach_phase(
            found, self.latched, armed=self.approach_armed, resume_in_sec=resume_in_sec,
        )
        status = {
            'mode': self.mode,
            'phase': phase,
            'latched': self.latched,
            'armed': self.approach_armed,
            'cycle': self.approach_cycle,
            'resume_in_sec': None if resume_in_sec is None else round(resume_in_sec, 2),
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
