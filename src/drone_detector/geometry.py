"""Pixel bounding-box  ->  ENU bearing vector.

This is the BRIDGE between the ML model and the rest of the system. The model
says "there is a drone at pixel (u, v) in camera C"; System 2 needs a direction
in world space to triangulate. These functions convert a pixel + the camera's
pose into a unit `[E, N, U]` bearing vector — exactly the `bearing_vector` field
in System 2's `/events` contract.

This module is the CANONICAL reference for the bearing math. The production copy
runs inside `system1-detection-agent`; keep the two in sync (the unit tests in
`tests/test_geometry.py` pin the conventions).

Conventions:
    ENU world frame: +E east, +N north, +U up   (right-handed)
    azimuth: degrees clockwise from North (the way the camera faces)
    elevation: degrees above the horizon (negative = looking down)
"""

import math


def bbox_center_to_bearing(
    u: float,
    v: float,
    img_width: int,
    img_height: int,
    azimuth_deg: float,
    elevation_deg: float,
    hfov_deg: float,
    vfov_deg: float,
) -> tuple[float, float, float]:
    """RTSP / real-camera path: pixel + camera orientation -> ENU unit bearing.

    Uses a flat-field linear approximation; valid for hFOV < ~120 degrees.
    A detection at the image center returns the camera's own azimuth/elevation
    direction; offsets are mapped linearly onto the field of view.
    """
    dx = (u - img_width / 2.0) / img_width
    dy = -(v - img_height / 2.0) / img_height  # image y down -> elevation up

    alpha = math.radians(azimuth_deg + dx * hfov_deg)
    phi = math.radians(elevation_deg + dy * vfov_deg)

    E = math.cos(phi) * math.sin(alpha)
    N = math.cos(phi) * math.cos(alpha)
    U = math.sin(phi)

    mag = math.sqrt(E * E + N * N + U * U)
    return (E / mag, N / mag, U / mag)


def _quat_rotate(q: list[float], v: list[float]) -> list[float]:
    """Rotate vector v by unit quaternion q = [x, y, z, w]. Returns [x, y, z]."""
    qx, qy, qz, qw = q
    vx, vy, vz = v

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]


def sim_bearing(
    center_px: list[float],
    K: list[float],
    rot_q: list[float],
) -> tuple[float, float, float]:
    """Simulation path: pixel + Unity intrinsics K + world quaternion -> ENU bearing.

    Args:
        center_px: [u, v] pixel coordinates (origin top-left, v down)
        K: flattened row-major 3x3 pinhole matrix [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        rot_q: Unity world rotation quaternion [x, y, z, w]

    Frames:
        Unity camera local: +X right, +Y up, +Z forward
        Unity world:        +X east, +Y up, +Z north (left-handed)
        ENU:                +E east, +N north, +U up (right-handed)
    """
    u, v = center_px
    fx, cx = K[0], K[2]
    fy, cy = K[4], K[5]

    # Back-project pixel to camera-local direction. Image v is down, Unity +Y is
    # up, so negate y.
    d_cam = [(u - cx) / fx, -(v - cy) / fy, 1.0]
    mag = math.sqrt(sum(c * c for c in d_cam))
    d_cam = [c / mag for c in d_cam]

    d_world = _quat_rotate(rot_q, d_cam)

    # Unity world -> ENU axis remap: Unity +X->E, +Y->U, +Z->N
    E, U, N = d_world[0], d_world[1], d_world[2]

    mag2 = math.sqrt(E * E + N * N + U * U)
    return (E / mag2, N / mag2, U / mag2)
