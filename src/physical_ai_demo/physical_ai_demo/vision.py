"""ArUco measurement on one color frame, with an optional aligned depth image."""

import cv2
import numpy as np


def ros_image_to_array(msg):
    """Decode a ROS image without cv_bridge.

    The Jazzy cv_bridge package on this Jetson was built against an older
    OpenCV than the OpenCV 5 this demo uses, and cv2_to_imgmsg raises KeyError.
    """
    encoding = msg.encoding.lower()
    if encoding == '16uc1':
        dtype, channels = np.uint16, 1
    elif encoding == '32fc1':
        dtype, channels = np.float32, 1
    elif encoding in ('bgr8', 'rgb8'):
        dtype, channels = np.uint8, 3
    elif encoding in ('bgra8', 'rgba8'):
        dtype, channels = np.uint8, 4
    else:
        raise RuntimeError('unsupported image encoding %s' % msg.encoding)

    itemsize = np.dtype(dtype).itemsize
    if msg.step % itemsize != 0:
        raise RuntimeError('image step %d is not a multiple of %d' % (msg.step, itemsize))
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    raw = raw[:msg.height * msg.step].reshape(msg.height, msg.step)
    typed = raw.view(dtype)
    image = typed[:, :msg.width * channels]
    if channels > 1:
        image = image.reshape(msg.height, msg.width, channels)
    else:
        image = image.reshape(msg.height, msg.width)
    image = np.array(image, copy=True)
    if encoding == 'rgb8':
        image = image[:, :, ::-1]
    elif encoding == 'rgba8':
        image = image[:, :, [2, 1, 0]]
    elif encoding == 'bgra8':
        image = image[:, :, :3]
    return image

# OpenCV corner order: top-left, top-right, bottom-right, bottom-left.
_OBJECT_CORNERS = np.array(
    [[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]],
    dtype=np.float32,
)


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return dictionary, cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def rotation_matrix_to_quaternion(rotation):
    """Return x, y, z, w from a 3x3 rotation matrix."""
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / scale
        x = 0.25 * scale
        y = (rotation[0, 1] + rotation[1, 0]) / scale
        z = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / scale
        x = (rotation[0, 1] + rotation[1, 0]) / scale
        y = 0.25 * scale
        z = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / scale
        x = (rotation[0, 2] + rotation[2, 0]) / scale
        y = (rotation[1, 2] + rotation[2, 1]) / scale
        z = 0.25 * scale
    return float(x), float(y), float(z), float(w)


def choose_range(depth_m, pnp_z_m):
    """Pick the range to act on.

    A blank page often returns no stereo depth, so the depth window lands on
    the wall behind it. When the two estimates disagree, keep solvePnP.
    """
    if depth_m is not None and pnp_z_m is not None and pnp_z_m > 0.05:
        if abs(depth_m - pnp_z_m) <= max(0.40, 0.45 * pnp_z_m):
            return depth_m, 'depth'
        return pnp_z_m, 'pnp'
    if depth_m is not None:
        return depth_m, 'depth'
    if pnp_z_m is not None and pnp_z_m > 0.05:
        return pnp_z_m, 'pnp'
    return None, 'none'


MENU_WIDTH = 640
MENU_HEIGHT = 270


def menu_choice_at(y, height=MENU_HEIGHT):
    """Three rows: detect, command preview, follow."""
    band = max(1, height) / 3.0
    if y < band:
        return 'report'
    if y < band * 2:
        return 'preview'
    return 'follow'


def render_menu_panel(mode, found, distance_m):
    """Clickable three-row menu. Row order matches menu_choice_at."""
    panel = np.full((MENU_HEIGHT, MENU_WIDTH, 3), 24, np.uint8)
    distance = f'{distance_m:.2f} m' if found and distance_m is not None else 'hold marker 23 up to the camera'
    _menu_row(panel, 0, mode == 'report', '1   DETECT', distance)
    _menu_row(panel, 1, mode == 'preview', '2   COMMAND', 'show the twist, do not drive')
    _menu_row(panel, 2, mode == 'follow', '3   FOLLOW', 'robot turns and drives to 1 m')
    return panel


def _menu_row(panel, index, active, title, detail):
    row_h = MENU_HEIGHT // 3
    y0 = index * row_h
    color = (40, 110, 60) if active else (48, 48, 48)
    cv2.rectangle(panel, (8, y0 + 8), (MENU_WIDTH - 8, y0 + row_h - 8), color, -1)
    mark = '>' if active else ' '
    cv2.putText(
        panel, f'{mark}  {title}', (24, y0 + 38),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        panel, detail, (28, y0 + 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA,
    )


def draw_menu_bar(image, mode, found, distance_m):
    """Paint the three choices along the bottom of the camera view."""
    height, width = image.shape[:2]
    bar_h = 64
    y0 = height - bar_h
    cv2.rectangle(image, (0, y0), (width, height), (16, 16, 16), -1)
    distance = f'{distance_m:.2f} m' if found and distance_m is not None else '--'
    rows = (
        (mode == 'report', f'1 DETECT  {distance}'),
        (mode == 'preview', '2 COMMAND'),
        (mode == 'follow', '3 FOLLOW'),
    )
    for index, (active, text) in enumerate(rows):
        color = (80, 220, 120) if active else (170, 170, 170)
        cv2.putText(
            image, text, (12 + index * (width // 3), y0 + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )


def draw_command_preview(image, u, v, linear_x, angular_z, bearing, distance_m):
    """Draw the twist Follow would send. The robot is not commanded."""
    height, width = image.shape[:2]
    center_x = width // 2
    top = 130
    bottom = height - 72
    cv2.line(image, (center_x, top), (center_x, bottom), (80, 220, 255), 2)
    if u is not None and v is not None:
        arrow_y = int(np.clip(v, top, bottom))
        arrow_x = int(np.clip(u, 0, width - 1))
        cv2.arrowedLine(
            image, (center_x, arrow_y), (arrow_x, arrow_y), (80, 220, 255), 2, tipLength=0.18,
        )
    if bearing is None:
        lines = ['no marker', 'command would be 0']
    else:
        if distance_m is None:
            range_line = 'range unknown -> linear.x +0.00'
        else:
            range_line = f'range {distance_m:.2f} m -> linear.x {linear_x:+.2f}'
        lines = [
            f'bearing {bearing:+.2f} -> angular.z {angular_z:+.2f}',
            range_line,
            'not sent - robot stays still',
        ]
    text_x = 12 if u is None or u >= center_x else center_x + 16
    y = top + 28
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (text_x - 6, y - text_h - 6), (text_x + text_w + 6, y + 8), (0, 0, 0), -1)
        cv2.putText(image, line, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 2, cv2.LINE_AA)
        y += text_h + 16


def _median_depth(depth, u, v, radius, scale):
    if depth is None:
        return None
    height, width = depth.shape[:2]
    u_int = int(round(u))
    v_int = int(round(v))
    u0 = max(0, u_int - radius)
    u1 = min(width, u_int + radius + 1)
    v0 = max(0, v_int - radius)
    v1 = min(height, v_int + radius + 1)
    window = depth[v0:v1, u0:u1].astype(np.float32) * scale
    valid = window[np.isfinite(window) & (window > 0.15) & (window < 8.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _draw_hud(image, lines):
    y = 34
    for line in lines:
        (width, height), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (8, y - height - 8), (20 + width, y + 8), (0, 0, 0), -1)
        cv2.putText(
            image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += height + 18


def measure_frame(
    bgr,
    detector,
    target_id,
    marker_size_m,
    camera_matrix=None,
    dist_coeffs=None,
    depth=None,
    depth_scale=0.001,
):
    """Detect marker target_id and return (annotated_bgr, measurement).

    measurement always contains found, id, bearing, distance_m, depth_m,
    pnp_z_m, range_source, and pose fields (None when unknown).
    """
    annotated = bgr.copy()
    height, width = bgr.shape[:2]
    center_depth = None
    if depth is not None and depth.shape[:2] == (height, width):
        center_depth = _median_depth(depth, width / 2.0, height / 2.0, 8, depth_scale)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)
    empty = {
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
        'quat_xyzw': None,
        'rvec': None,
        'tvec': None,
        'center_depth_m': None,
        'u': None,
        'v': None,
    }
    empty['center_depth_m'] = center_depth
    if ids is None or len(ids) == 0:
        line = f'searching for marker {target_id}'
        if center_depth is not None:
            line += f'    center {center_depth:.2f} m'
        _draw_hud(annotated, [line])
        return annotated, empty

    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    chosen = None
    for index, marker_id in enumerate(ids.flatten()):
        if int(marker_id) == int(target_id):
            chosen = index
            break
    if chosen is None:
        line = f'searching for marker {target_id}'
        if center_depth is not None:
            line += f'    center {center_depth:.2f} m'
        _draw_hud(annotated, [line])
        return annotated, empty

    image_corners = corners[chosen].reshape(-1, 2).astype(np.float32)
    cv2.polylines(annotated, [image_corners.astype(np.int32)], True, (0, 220, 0), 2)
    center = image_corners.mean(axis=0)
    height, width = bgr.shape[:2]
    bearing = float(np.clip((center[0] - (width / 2.0)) / (width / 2.0), -1.0, 1.0))
    depth_m = _median_depth(depth, center[0], center[1], 4, depth_scale)

    result = dict(empty)
    result.update({
        'found': True,
        'id': int(target_id),
        'bearing': bearing,
        'depth_m': depth_m,
        'u': float(center[0]),
        'v': float(center[1]),
    })

    pose_ok = False
    if camera_matrix is not None:
        dist = np.zeros(5, dtype=np.float64) if dist_coeffs is None else dist_coeffs
        object_points = _OBJECT_CORNERS * float(marker_size_m)
        pose_ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_corners,
            camera_matrix,
            dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if pose_ok:
            tvec = tvec.reshape(3).astype(np.float64)
            rvec = rvec.reshape(3).astype(np.float64)
            pnp_z = float(tvec[2])
            result['pnp_z_m'] = pnp_z
            distance_m, source = choose_range(depth_m, pnp_z)
            result['distance_m'] = distance_m
            result['range_source'] = source
            if source == 'depth' and pnp_z > 0.05:
                tvec = tvec * (distance_m / pnp_z)
            rotation, _ = cv2.Rodrigues(rvec)
            result['x_m'] = float(tvec[0])
            result['y_m'] = float(tvec[1])
            result['z_m'] = float(tvec[2])
            result['quat_xyzw'] = rotation_matrix_to_quaternion(rotation)
            result['rvec'] = rvec
            result['tvec'] = tvec
            cv2.drawFrameAxes(
                annotated, camera_matrix, dist, rvec, tvec, float(marker_size_m) * 0.5
            )

    if result['distance_m'] is None and depth_m is not None:
        result['distance_m'], result['range_source'] = choose_range(depth_m, None)
        result['z_m'] = result['distance_m']

    distance_text = 'range unknown' if result['distance_m'] is None else f"{result['distance_m']:.2f} m ({result['range_source']})"
    depth_text = '' if result['depth_m'] is None else f"depth {result['depth_m']:.2f} m"
    pnp_text = '' if result['pnp_z_m'] is None else f"solvePnP {result['pnp_z_m']:.2f} m"
    detail = '   '.join(part for part in (depth_text, pnp_text) if part)
    _draw_hud(
        annotated,
        [
            'OpenCV ArUco  DICT_4X4_50',
            f"marker {target_id}   {distance_text}",
            detail or 'range unknown',
        ],
    )
    if not pose_ok:
        cv2.circle(annotated, (int(center[0]), int(center[1])), 6, (0, 220, 0), -1)
    return annotated, result
