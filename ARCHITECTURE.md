# Architecture — How the ML fits into the whole system

> Audience: teammates and AI agents working on any part of the drone
> airspace-monitoring system. This explains what the ML component does, the exact
> contracts at its boundaries, and how data flows end-to-end.

---

## 1. The whole system at a glance

Four components, each its own repo, plus the Unity simulator:

```
                         ┌─────────────────────────────────────────────┐
                         │  this repo: drone-detection-ml               │
                         │  the model + inference + bearing geometry     │
                         └───────────────────┬─────────────────────────┘
                                             │ imported by
                                             ▼
 Unity sim ──UDP:9870──►  ┌──────────────────────────┐  POST /events  ┌────────────────────────┐
 (Marius)   labels        │ system1-detection-agent  │ ─────────────► │ system2-positioning-   │
                          │ (Cristiana)              │  bearing vec   │ engine (Filipp)        │
 Real cam ──RTSP─────────►│  detect → bearing → POST │                │  ray triangulation     │
                          └──────────────────────────┘                │  → GPS position        │
                                                                       └───────────┬────────────┘
                                                                                   │ positions table
                                                                                   ▼
                                                                       ┌────────────────────────┐
                                                                       │ System 3 dashboard     │
                                                                       │ (Bogdan): map, zones,  │
                                                                       │ alerts                 │
                                                                       └────────────────────────┘
```

| Component | Repo | Owner | Responsibility |
|---|---|---|---|
| **ML model** | `drone-detection-ml` (this) | Cristiana | Detect drones in a frame; convert pixel → ENU bearing |
| System 1 — detection agent | `system1-detection-agent` | Cristiana | Runtime: ingest frames, call ML, POST events to System 2 |
| System 2 — positioning | `system2-positioning-engine` | Filipp | Triangulate bearings from ≥2 cameras → GPS position → DB |
| System 3 — dashboard | (tbd) | Bogdan | Map of HK, forbidden zones, show drones, alerts |
| Simulator | Unity | Marius | Render cameras; broadcast ground-truth labels over UDP |

**This repo is the "brain".** It does not open cameras, does not talk to the
network, and does not own any threads. It exposes two things that
`system1-detection-agent` imports:
1. `DroneDetector` — frame → detections (the model).
2. `bbox_center_to_bearing` / `sim_bearing` — pixel → ENU bearing (the bridge to System 2).

---

## 2. The two perception paths

The system runs in either of two modes; the ML repo supports both.

### A. `sim` path (Unity — used for the hackathon demo)

Unity already knows where every drone is, so **no neural network runs**. Unity
broadcasts one UDP datagram per frame containing, per camera: the intrinsic matrix
`K`, the world rotation quaternion `rot_q`, and each detection's pixel center.
System 1 calls `sim_bearing(center_px, K, rot_q)` to turn that into an ENU bearing.

`score = 1.0` (ground truth). The `DroneDetector` (YOLO) is **not** used here.

### B. `rtsp` path (real cameras)

For physical cameras there are no labels, so the model does the work:

```
frame ─► DroneDetector.detect() ─► [Detection(cx, cy, score, cls, bbox), ...]
                                          │
                                          ▼  bbox_center_to_bearing(cx, cy, ...)
                                    ENU bearing [E, N, U]
```

Bearing here comes from the camera's surveyed `azimuth/elevation/FOV`
(in System 1's `cameras.yaml`), not from a quaternion.

---

## 3. The output contract (ML/System 1 → System 2)

Everything the ML produces is ultimately shaped into this POST. **This is the one
contract that must never drift.**

`POST {SYSTEM2_URL}/events`

```json
{
  "cam_id": "cam_01",
  "timestamp": "2026-06-06T14:23:00.456Z",
  "detections": [
    { "bearing_vector": [0.342, 0.876, -0.340], "score": 0.97 }
  ]
}
```

| Field | Meaning | Hard rule |
|---|---|---|
| `cam_id` | which camera | must match System 2's `cameras.yaml` exactly, else **silently dropped** |
| `timestamp` | UTC time of the frame | NTP-synced UTC; drift > 0.5 s breaks triangulation's time window |
| `bearing_vector` | unit `[E, N, U]` direction to the drone | must be ENU and unit-length |
| `score` | confidence | YOLO confidence (rtsp) or 1.0 (sim) |

`detections: []` is valid (camera sees nothing this frame).

---

## 4. The bridge: pixel → ENU bearing (the math that matters)

A wrong bearing produces a wrong GPS position **with no error anywhere** — System 2
will happily triangulate garbage. This is the highest-risk code in the system, so
it is pinned by `tests/test_geometry.py`.

**ENU frame:** `+E` east, `+N` north, `+U` up (right-handed).

### sim path — `sim_bearing(center_px, K, rot_q)`
```
d_cam   = normalize([(u-cx)/fx, -(v-cy)/fy, 1])   # back-project; negate y (image down vs Unity up)
d_world = quaternion_rotate(rot_q, d_cam)          # camera-local → Unity world
[E,N,U] = [d_world.x, d_world.z, d_world.y]         # Unity (X,Y,Z) → ENU (E,N,U)
```

### rtsp path — `bbox_center_to_bearing(u, v, w, h, az, el, hfov, vfov)`
```
α = az + ((u - w/2)/w) · hfov        # azimuth, clockwise from North
φ = el - ((v - h/2)/h) · vfov        # elevation, up positive
[E,N,U] = [cosφ·sinα, cosφ·cosα, sinφ]
```

**Invariants the tests lock down** (so any change/format mismatch fails loudly):
- identity rotation + center pixel → pure North `[0,1,0]`
- pixel right of center → leans East; pixel above center → leans Up
- +90° yaw → camera faces East (matches Unity `Quaternion.Euler(0,90,0)*forward`)
- output is always a unit vector

---

## 5. End-to-end data flow (sim demo)

```
1. Unity renders cam0 & cam1, sees a drone, broadcasts one UDP datagram
   { t_unix_ms, cameras:[ {name, K, rot_q, detections:[{center_px, visible}]} ] }
2. System 1 receives it, and per camera calls sim_bearing(...) → [E,N,U]
3. System 1 POSTs one /events per camera to System 2
4. System 2 caches bearings; when ≥2 cameras report the same drone within the
   time window, it triangulates the ray intersection → lat/lon/alt → positions table
5. System 3 polls positions, draws them on the HK map, checks forbidden zones, alerts
```

A position should appear in System 2's DB within ~1 s of the drone being visible
to two cameras.

---

## 6. Boundaries an agent must respect

| Boundary | Contract | Where defined |
|---|---|---|
| ML → System 1 | `DroneDetector.detect(frame) -> list[Detection]` | `src/drone_detector/detector.py` |
| pixel → world | `*_bearing(...) -> (E, N, U)` unit vector | `src/drone_detector/geometry.py` |
| System 1 → System 2 | `POST /events` JSON above | this doc + System 2 repo |
| Unity → System 1 | UDP datagram fields (`name`, `K`, `rot_q`, `center_px`, `visible`) | Marius + System 1 `udp_listener.py` |
| System 2 → System 3 | `positions` table (lat, lon, alt_m, cam_pair, …) | System 2 repo |

### Silent-failure traps (no error is raised — verify these manually)
1. `cam_id` mismatch with System 2's `cameras.yaml` → events dropped.
2. Non-ENU or non-unit bearing → wrong GPS, no error.
3. Clock drift between cameras → triangulation time window misses.
4. Unity datagram field/format mismatch → bearings silently wrong; check with
   `python -m system1.main --dump`.

---

## 7. Where to change what

- **Better detection / new weights** → swap `models/yolov8n-drone.onnx`; point
  System 1's `MODEL_PATH` at it. No code change.
- **Long-range / small drones** → add SAHI in `detector.py` (see MODEL_CARD).
- **Bearing math** → `geometry.py` here is the canonical reference; keep
  `system1-detection-agent/system1/geometry.py` identical, and run the tests in
  both repos.
- **New camera** → add it to System 1 and System 2 `cameras.yaml` with matching
  `cam_id`.
