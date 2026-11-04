# From computer vision to a body

A live demo for the OpenCV community. A RealSense D585 and OpenCV find a printed ArUco marker. A deterministic skill turns that bearing and range into `/cmd_vel`. A virtual robot in RViz shows the motion. Corebrum Synapse only chooses the mode.

The camera is a fixed stage camera. The robot is not localized against it. Left and right in the image yaw the robot. Range drives it forward or back. One meter is the setpoint.

## Screen

RViz shows two things:

- **OpenCV** — marker corners, coordinate axes, id, RealSense distance, and the `solvePnP` distance.
- **Map** — the robot, a label (`REPORT`, `FOLLOW`, `APPROACH`, `STOPPED AT 1 m`), and a green goal puck. The view follows the robot.

## Before Thursday

Print [src/physical_ai_demo/assets/marker_23.png](src/physical_ai_demo/assets/marker_23.png) so the black square is **12 cm** on a side. Leave the white margin. Tape it to something rigid. Depth does not use that size. `solvePnP` does. If the print comes out another size:

```bash
ros2 launch physical_ai_demo demo.launch.py marker_size_m:=0.10
```

Build once:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/nvidia/Projects/opencv
colcon build --symlink-install
```

## Run of show

```bash
source /opt/ros/jazzy/setup.bash
source /home/nvidia/Projects/opencv/install/setup.bash
ros2 launch physical_ai_demo demo.launch.py
```

A window titled **Demo menu** sits on top of the other windows. Click a row, or press 1, 2, or 3 while that window is focused. The same choices are drawn on the camera picture.

1. **DETECT** shows marker 23 and its distance. The robot stays put.
2. **COMMAND** draws the image center, an arrow to the marker, and the twist Follow would send: bearing becomes `angular.z`, and distance minus one meter becomes `linear.x`. The robot stays blue and still.
3. **FOLLOW** sends that same twist. The robot turns green and drives to stay about one meter away.

The distance used for motion is RealSense depth when it agrees with `solvePnP`. A blank page often makes the depth window read the wall behind it; in that case the menu shows the `solvePnP` distance and still prints both numbers.

Click **DETECT** again to stop. `ros2_estop` also stops the robot, and it stays stopped until the next menu choice.

Park the robot back at the origin between takes:

```bash
ros2 service call /demo/reset_pose std_srvs/srv/Empty
```

## Act 4 — Synapse mission (surprise)

Acts 1–3 stay on the Demo menu. Act 4 is chat-only: Corebrum Synapse publishes `approach` on `/demo/mode`. The same skill drives to about one meter, holds for five seconds, then hunts for the marker again until you stop it. Do not add a fourth menu button.

Choreography:

1. End **FOLLOW**, click **DETECT** to stop.
2. Reset the stage robot:

```bash
ros2 service call /demo/reset_pose std_srvs/srv/Empty
```

3. Hold marker 23 in frame. Switch to Cursor.
4. Say: *“Approach marker 23 and stop at one meter.”*

Audience cue: as soon as `/demo/mode` is `approach`, the Demo menu window withdraws entirely and the camera overlay drops its HUD so the stage is Cursor (Synapse) + RViz. RViz goes `APPROACH` (orange), then `STOPPED AT 1 m` (yellow) for five seconds, then hunts again after the marker leaves the frame. The agent should keep printing Synapse `/demo/status` updates in Cursor (`phase`, distance, cycle, resume countdown). Publish `approach` even if the marker is not up yet — waiting for `found` first leaves the three-button menu on screen.

Recovery: ask the agent to stop, Synapse `ros2_estop`, or publish `report` on `/demo/mode` (the Demo menu returns when mode leaves approach).

Rehearsal without the agent:

```bash
ros2 topic pub --once /demo/mode std_msgs/msg/String "{data: approach}"
```

Dry run: launch → DETECT → COMMAND → FOLLOW → DETECT → reset → pub `approach` → confirm latch in RViz → DETECT.

## What the agent is allowed to do

Read `/demo/marker`. Publish `report`, `follow`, or `approach` on `/demo/mode`. The skill owns the control loop — never invent a parallel `/cmd_vel` controller for this demo. These are the fields in `/demo/marker`:

| Field | Meaning |
| --- | --- |
| `id` | 23 when that marker is in frame |
| `bearing` | -1 is the left edge, 0 is center, +1 is the right edge |
| `distance_m` | Range used for motion |
| `depth_m` | Median RealSense depth inside the marker |
| `center_depth_m` | Depth at the middle of the picture when marker 23 is absent |
| `depth_aligned` | True when the sample came from the aligned depth topic |
| `pnp_z_m` | Range from `solvePnP` and the printed size |
| `range_source` | `depth`, `pnp`, or `none` |
| `x_m`, `y_m`, `z_m` | Marker origin in the camera optical frame (x right, y down, z forward) |

`/demo/status` is the skill's own copy: mode, whether approach has latched, and the twist it is publishing.

## If the picture is empty

On this D585 the driver's align filter advertises `/camera/camera/aligned_depth_to_color/image_raw` and then publishes no frames. The eye uses that topic when it is alive, and otherwise reads `/camera/camera/depth/image_rect_raw` at the marker pixel. `depth_aligned` in `/demo/marker` says which one. `solvePnP` is still printed beside it. Check that the streams are alive:

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /demo/debug_image
```

Launch without RViz while checking the graph:

```bash
ros2 launch physical_ai_demo demo.launch.py rviz:=false
```
