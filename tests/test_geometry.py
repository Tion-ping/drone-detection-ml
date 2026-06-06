"""Tests for the bearing-vector math — the most safety-critical code in the system.

A wrong bearing produces a wrong GPS position downstream with NO error from
System 2. These tests pin the coordinate-frame conventions so a future change or
Unity-format mismatch fails here instead of silently corrupting positions.

Run (no pytest needed):
    python tests/test_geometry.py        # from repo root
or:
    pytest tests/
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from drone_detector.geometry import bbox_center_to_bearing, sim_bearing, _quat_rotate

TOL = 1e-6


def _approx(a, b, tol=TOL):
    return abs(a - b) <= tol


def _is_unit(vec, tol=1e-9):
    return _approx(math.sqrt(sum(c * c for c in vec)), 1.0, tol)


K_ID = [623.54, 0.0, 640.0, 0.0, 623.54, 360.0, 0.0, 0.0, 1.0]
CENTRE = [640.0, 360.0]
Q_ID = [0.0, 0.0, 0.0, 1.0]


# --- sim_bearing (Unity path, used for the demo) ---------------------------- #

def test_sim_identity_centre_is_north():
    e, n, u = sim_bearing(CENTRE, K_ID, Q_ID)
    assert _approx(e, 0.0) and _approx(n, 1.0) and _approx(u, 0.0), (e, n, u)


def test_sim_output_is_unit_vector():
    for px in ([640, 360], [0, 0], [1279, 719], [900, 200]):
        assert _is_unit(sim_bearing(px, K_ID, Q_ID))


def test_sim_pixel_right_points_east():
    e, n, u = sim_bearing([900, 360], K_ID, Q_ID)
    assert e > 0.0 and _approx(u, 0.0)


def test_sim_pixel_left_points_west():
    e, _, _ = sim_bearing([380, 360], K_ID, Q_ID)
    assert e < 0.0


def test_sim_pixel_above_centre_points_up():
    _, _, u = sim_bearing([640, 100], K_ID, Q_ID)
    assert u > 0.0


def test_sim_pixel_below_centre_points_down():
    _, _, u = sim_bearing([640, 600], K_ID, Q_ID)
    assert u < 0.0


def test_sim_yaw_90_faces_east():
    q = [0.0, math.sin(math.radians(45)), 0.0, math.cos(math.radians(45))]
    e, n, u = sim_bearing(CENTRE, K_ID, q)
    assert _approx(e, 1.0, 1e-6) and _approx(n, 0.0, 1e-6) and _approx(u, 0.0, 1e-6)


def test_sim_pitch_up_90_faces_zenith():
    q = [math.sin(math.radians(-45)), 0.0, 0.0, math.cos(math.radians(-45))]
    _, _, u = sim_bearing(CENTRE, K_ID, q)
    assert _approx(u, 1.0, 1e-6)


def test_quat_rotate_identity():
    assert _quat_rotate(Q_ID, [1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]


# --- bbox_center_to_bearing (RTSP / real-camera path) ----------------------- #

def test_rtsp_centre_matches_orientation():
    az, el = 45.0, 20.0
    e, n, u = bbox_center_to_bearing(640, 360, 1280, 720, az, el, 90.0, 60.0)
    exp_e = math.cos(math.radians(el)) * math.sin(math.radians(az))
    exp_n = math.cos(math.radians(el)) * math.cos(math.radians(az))
    exp_u = math.sin(math.radians(el))
    assert _approx(e, exp_e) and _approx(n, exp_n) and _approx(u, exp_u)


def test_rtsp_output_is_unit_vector():
    for u_px, v_px in ([640, 360], [0, 0], [1279, 719]):
        assert _is_unit(bbox_center_to_bearing(u_px, v_px, 1280, 720, 0.0, 0.0, 90.0, 60.0))


def test_rtsp_north_facing_centre_is_north():
    e, n, u = bbox_center_to_bearing(640, 360, 1280, 720, 0.0, 0.0, 90.0, 60.0)
    assert _approx(e, 0.0) and _approx(n, 1.0) and _approx(u, 0.0)


def test_rtsp_right_pixel_points_east():
    e, _, _ = bbox_center_to_bearing(1000, 360, 1280, 720, 0.0, 0.0, 90.0, 60.0)
    assert e > 0.0


def test_rtsp_high_pixel_increases_elevation():
    _, _, u_c = bbox_center_to_bearing(640, 360, 1280, 720, 0.0, 0.0, 90.0, 60.0)
    _, _, u_h = bbox_center_to_bearing(640, 50, 1280, 720, 0.0, 0.0, 90.0, 60.0)
    assert u_h > u_c


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"FAIL  {t.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)
