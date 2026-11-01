"""OpenCV 5 ArUco path, including the solvePnP backup range."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physical_ai_demo.vision import (
    choose_range,
    make_detector,
    measure_frame,
    menu_choice_at,
    ros_image_to_array,
)


class _Msg:
    def __init__(self, image, encoding):
        self.encoding = encoding
        self.height, self.width = image.shape[:2]
        self.data = image.tobytes()
        channels = image.shape[2] if image.ndim == 3 else 1
        self.step = self.width * channels * image.dtype.itemsize


def _scene():
    dictionary, detector = make_detector()
    marker = cv2.aruco.generateImageMarker(dictionary, 23, 200)
    image = np.full((480, 640, 3), 255, np.uint8)
    image[140:340, 220:420] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    camera = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    return detector, image, camera


def test_centered_marker_reports_id_pose_and_range():
    detector, image, camera = _scene()
    _annotated, measured = measure_frame(
        image, detector, 23, 0.12, camera_matrix=camera, dist_coeffs=np.zeros(5)
    )
    assert measured['found'] is True
    assert measured['id'] == 23
    assert abs(measured['bearing']) < 0.05
    # z = fx * marker_size / pixel_size = 600 * 0.12 / 200
    assert abs(measured['pnp_z_m'] - 0.36) < 0.03
    assert measured['range_source'] == 'pnp'
    assert measured['quat_xyzw'] is not None


def test_consistent_depth_is_the_range():
    detector, image, camera = _scene()
    depth = np.full((480, 640), 400, np.uint16)  # 0.40 m, close to the synthetic pose
    _annotated, measured = measure_frame(
        image,
        detector,
        23,
        0.12,
        camera_matrix=camera,
        dist_coeffs=np.zeros(5),
        depth=depth,
        depth_scale=0.001,
    )
    assert measured['range_source'] == 'depth'
    assert abs(measured['distance_m'] - 0.40) < 1e-6
    assert abs(measured['z_m'] - 0.40) < 0.05
    assert abs(measured['pnp_z_m'] - 0.36) < 0.03


def test_background_depth_falls_back_to_solvepnp():
    detector, image, camera = _scene()
    depth = np.full((480, 640), 5100, np.uint16)  # wall behind a blank page
    _annotated, measured = measure_frame(
        image,
        detector,
        23,
        0.12,
        camera_matrix=camera,
        dist_coeffs=np.zeros(5),
        depth=depth,
        depth_scale=0.001,
    )
    assert measured['range_source'] == 'pnp'
    assert abs(measured['distance_m'] - measured['pnp_z_m']) < 1e-6
    assert abs(measured['depth_m'] - 5.1) < 1e-6
    assert choose_range(5.1, 1.35) == (1.35, 'pnp')
    assert choose_range(1.40, 1.35)[1] == 'depth'


def test_menu_rows():
    assert menu_choice_at(20) == 'report'
    assert menu_choice_at(170) == 'follow'


def test_ros_image_round_trip_for_color_and_depth():
    color = np.zeros((3, 4, 3), np.uint8)
    color[1, 2] = (10, 20, 30)  # stored RGB; the decoder swaps to BGR
    decoded = ros_image_to_array(_Msg(color, 'rgb8'))
    assert decoded.shape == (3, 4, 3)
    assert decoded[1, 2].tolist() == [30, 20, 10]
    depth = np.array([[1000, 0], [1500, 2000]], np.uint16)
    decoded_depth = ros_image_to_array(_Msg(depth, '16UC1'))
    assert decoded_depth.tolist() == depth.tolist()


def test_other_ids_are_not_the_target():
    dictionary, detector = make_detector()
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 200)
    image = np.full((480, 640, 3), 255, np.uint8)
    image[140:340, 220:420] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    _annotated, measured = measure_frame(image, detector, 23, 0.12)
    assert measured['found'] is False
