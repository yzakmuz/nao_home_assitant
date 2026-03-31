# Webots Integration — Detailed Implementation Plan

**Date:** 2026-03-17
**Webots Version:** R2023b (confirmed fully compatible with NAO V5)
**Python:** 3.9+ (natively supported by R2023b)
**Status:** Planning complete, implementing step by step
**Implementation folder:** `webots-sim/`
**Goal:** Run the ElderGuard system in Webots with a virtual NAO, virtual room,
and virtual person — demonstrating all features in a 3D physics-based environment.

### Version Compatibility Confirmed
- NAO V5 PROTO: included, actively improved in R2023b (PR #5622)
- All 15 motion files: present (Forwards50, TurnLeft60, HandWave, etc.)
- 25 DOF joints: same names as real NAO (HeadYaw, RHand, etc.)
- CameraTop/CameraBottom: functional, configurable resolution
- Python 3.9+: natively supported
- Connector node: available for object grasping
- Speaker TTS: available (SVOX Pico engine)
- 8 demo controllers + 8 sample worlds included

---

## 1. Research Findings — Critical Facts

Before planning, these are the hard facts from Webots documentation and source code:

### What Works Perfectly
- **Joint names are IDENTICAL** to NAOqi: `HeadYaw`, `HeadPitch`, `RShoulderPitch`,
  `RHand`, etc. Our `motion_library.py` constants work directly.
- **Camera returns NumPy-compatible data** — BGRA bytes convertible to BGR for OpenCV.
- **TCP server inside a Webots controller works** — `threading.Thread` is supported.
- **Webots Speaker has TTS** — `speaker.speak("Hello")` works cross-platform.
- **Connector node for grasping** — `lock()`/`unlock()` to attach objects to the hand.
- **Resolution is configurable** — can match our 320×240 Pi Camera setting.

### Critical Gaps (Must Solve)
| Gap | Problem | Solution |
|-----|---------|----------|
| **No walk engine** | Webots NAO has no `moveTo()`. Only pre-recorded `.motion` files for walking. | Use motion files (`Forwards50.motion`, `TurnLeft60.motion`, etc.) with loop/stop control. |
| **No face on pedestrian** | Default Webots `Pedestrian` has a featureless geometric head — **MediaPipe will NOT detect it**. | **Two options:** (A) Replace head with photo-textured sphere, or (B) skip MediaPipe in sim and feed simulated face coordinates based on pedestrian's known 3D position. |
| **No `goToPosture()`** | No posture manager. Must set all 26 joint angles manually. | Store joint angle presets (Stand, Sit, Crouch) in the controller. Our `motion_library.py` already has these values. |
| **`robot.step()` is blocking** | Simulation freezes if TCP `recv()` blocks in the main thread. | Run TCP server in a background thread, queue commands, process in the `robot.step()` loop. |

---

## 2. Architecture — Two Processes

```
┌─────────────────────────────────────────────────────────────────┐
│  Process 1: Brain (existing code, unchanged)                    │
│                                                                 │
│  PC Mic → Vosk STT → main.py FSM → TCP Client (port 5555)     │
│  Camera* → MediaPipe Face → PID Servo → TCP commands            │
│  Camera* → YOLO → Object detection                              │
│  Camera* → MediaPipe Pose → Fall detection                      │
│                                                                 │
│  * Camera source is either:                                     │
│    - Webots camera (streamed via TCP from Process 2)            │
│    - PC webcam (hybrid mode, same as current simulation)        │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                    TCP JSON (port 5555)
┌────────────────────────┴────────────────────────────────────────┐
│  Process 2: Webots Controller (new code)                        │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐     │
│  │ webots_nao_controller.py                               │     │
│  │                                                        │     │
│  │ TCP Server Thread:                                     │     │
│  │   - Listens on port 5555                               │     │
│  │   - Receives JSON commands from brain                  │     │
│  │   - Queues them for the main loop                      │     │
│  │   - Sends responses (ack, done, state)                 │     │
│  │                                                        │     │
│  │ Camera Server Thread:                                  │     │
│  │   - Streams camera images to brain (port 5556)         │     │
│  │   - BGRA → BGR conversion                             │     │
│  │   - Configurable resolution (320×240 default)          │     │
│  │                                                        │     │
│  │ Main Loop (robot.step):                                │     │
│  │   - Process queued TCP commands                        │     │
│  │   - Set motor positions                                │     │
│  │   - Play/stop motion files (walk, turn)                │     │
│  │   - Manage Connector (grasp/release)                   │     │
│  │   - Update state (posture, channels)                   │     │
│  │   - Read camera image for streaming                    │     │
│  └────────────────────────────────────────────────────────┘     │
│                                                                 │
│  Webots World:                                                  │
│   - Virtual room (4m × 4m, furniture, lighting)                │
│   - NAO V5 robot (with CameraTop, Speaker, Connectors)        │
│   - Virtual person (with face texture for MediaPipe)           │
│   - Objects on floor (phone, bottle, cup with Connectors)      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. The Face Detection Problem — Detailed Solution

### Option A: Photo-Textured Head on Pedestrian (Recommended)

Replace the pedestrian's geometric head with a sphere that has a **photo of a
real human face** mapped as a texture:

```
Pedestrian PROTO (modified):
  head Shape {
    appearance PBRAppearance {
      baseColorMap ImageTexture { url "textures/human_face.jpg" }
    }
    geometry Sphere { radius 0.10 }
  }
```

**Why this works:** MediaPipe BlazeFace is trained on real face images. A
photo-textured sphere will trigger detection as long as:
- The texture is a front-facing human face photo
- The resolution is sufficient (the sphere needs to be ~50+ pixels in the camera)
- The lighting is adequate (Webots' PBR renderer handles this)

**Testing needed:** Run MediaPipe on a screenshot of the Webots view with the
textured head to verify detection before committing to this approach.

### Option B: Simulated Face Coordinates (Fallback)

If Option A doesn't produce reliable MediaPipe detections, bypass face detection
entirely in the Webots controller:

1. The controller knows the pedestrian's exact 3D position (from Webots API)
2. Project the pedestrian's head position into the NAO camera's image plane
3. Send these simulated `(cx, cy, width, height)` coordinates to the brain
   as if MediaPipe had detected them
4. The brain's PID servo uses these coordinates normally — it doesn't know
   they're simulated

```python
# In webots_nao_controller.py:
ped_pos = pedestrian_node.getPosition()  # [x, y, z] in world coords
cam_pos = nao_camera_node.getPosition()
cam_rot = nao_camera_node.getOrientation()

# Project ped_pos into camera image coordinates
cx, cy = project_to_image(ped_pos, cam_pos, cam_rot, fov, width, height)
face_width = estimate_face_size(distance_to_ped)

# Send to brain via a separate channel or embed in state response
simulated_face = {"cx": cx, "cy": cy, "width": face_width, "height": face_width * 1.3}
```

**This approach is 100% reliable** because it uses exact 3D geometry, not
computer vision. The trade-off: it doesn't demonstrate MediaPipe working.
For a demo, you could show both: MediaPipe running on the PC webcam (your real
face) while the simulated coordinates drive the Webots NAO.

### Recommended Strategy
1. **Try Option A first** (photo-textured head). Test with one screenshot.
2. **If MediaPipe detects it →** use Option A for all demos.
3. **If MediaPipe fails →** use Option B (simulated coords) for the Webots 3D view,
   and show real MediaPipe working on the PC webcam in the dashboard side-by-side.

---

## 4. Walking Without a Walk Engine

Webots' NAO has no `ALMotion.moveTo()`. It uses pre-recorded motion files.

### Available Motion Files (Built Into Webots)

| File | Action | Duration |
|------|--------|----------|
| `Forwards50.motion` | Walk forward ~50cm | ~4s |
| `Backwards.motion` | Walk backward | ~3s |
| `SideStepLeft.motion` | Side step left | ~2s |
| `SideStepRight.motion` | Side step right | ~2s |
| `TurnLeft60.motion` | Turn left ~60° | ~3s |
| `TurnRight60.motion` | Turn right ~60° | ~3s |
| `TurnLeft180.motion` | Turn 180° left | ~6s |
| `StandUpFromFront.motion` | Get up from face-down | ~5s |
| `StandUpFromBack.motion` | Get up from back | ~5s |

### Walk Command Translation

When the brain sends `walk_toward(x=0.5, y=0, theta=0.3)`:

```python
# In webots_nao_controller.py:
def handle_walk_toward(self, x, y, theta):
    # Decide which motion to play based on parameters
    if abs(theta) > 0.5:
        # Turn first, then walk
        if theta > 0:
            self.turn_left_motion.play()
        else:
            self.turn_right_motion.play()
    elif x > 0.1:
        # Walk forward
        self.forward_motion.setLoop(True)
        self.forward_motion.play()
        # Schedule stop after x/0.12 seconds (NAO walks ~0.12 m/s)
        self._walk_stop_time = self.robot.getTime() + x / 0.12
    elif x < -0.1:
        self.backward_motion.play()

def handle_stop_walk(self):
    for motion in self._all_motions:
        motion.stop()

# In main loop:
if self._walk_stop_time and self.robot.getTime() > self._walk_stop_time:
    self.handle_stop_walk()
    self._walk_stop_time = None
```

### `set_walk_velocity` (Continuous Walking)

For continuous velocity-mode walking (used by "come here" follow mode):

```python
def handle_set_walk_velocity(self, x, y, theta):
    # Simple approach: loop forward motion + adjust heading
    if x > 0.05:
        if not self.forward_motion.isOver():
            pass  # Already playing
        else:
            self.forward_motion.setLoop(True)
            self.forward_motion.play()
    else:
        self.forward_motion.stop()

    # Heading adjustment: rotate body by setting hip/ankle joints
    # (Small corrections, not full turns)
    if abs(theta) > 0.1:
        yaw_adj = theta * 0.05  # Small incremental turn
        current = self.head_yaw_motor.getTargetPosition()
        # Apply to hip to create slight turn during walk
```

**Limitation:** The motion-file approach produces a fixed walking gait. The robot
can't smoothly adjust speed or direction mid-stride. For the demo, this is
acceptable — the audience sees the robot walking, which is the goal.

---

## 5. Camera Image Streaming to Brain

The brain needs camera frames from Webots NAO's camera. Two approaches:

### Approach A: Adapter Camera Class (Recommended)

Create a `WebotsCamera` adapter that implements the same `Camera.read()` API
as our existing `camera.py`, but fetches images from the Webots controller
via a TCP image stream:

```python
# webots_camera_adapter.py (runs in brain process)
class WebotsCamera:
    """Drop-in replacement for vision.camera.Camera.
    Receives images from Webots controller via TCP."""

    def __init__(self, host="127.0.0.1", port=5556):
        self._sock = socket.socket()
        self._sock.connect((host, port))
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._receive_loop, daemon=True).start()

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _receive_loop(self):
        while self._running:
            # Receive frame: 4-byte length prefix + raw BGR bytes
            length = struct.unpack(">I", self._sock.recv(4))[0]
            data = b""
            while len(data) < length:
                data += self._sock.recv(min(length - len(data), 65536))
            frame = np.frombuffer(data, dtype=np.uint8).reshape((240, 320, 3))
            with self._lock:
                self._frame = frame
```

**On the Webots side:**
```python
# In webots_nao_controller.py camera thread:
def camera_stream_thread(self):
    server = socket.socket()
    server.bind(("0.0.0.0", 5556))
    server.listen(1)
    conn, _ = server.accept()

    while self._running:
        image = self.camera.getImage()
        bgr = np.frombuffer(image, dtype=np.uint8).reshape(
            (self.cam_height, self.cam_width, 4))[:, :, :3]
        data = bgr.tobytes()
        conn.sendall(struct.pack(">I", len(data)) + data)
        time.sleep(0.066)  # ~15 FPS
```

### Approach B: PC Webcam (Hybrid, Simpler)

Keep using the real PC webcam for the brain's vision pipeline. The Webots 3D
view shows the robot reacting to commands, but face/YOLO detection runs on
your real camera. This is the **fastest to implement** and already works with
the current simulation.

**For the demo:** Show Webots on one screen (robot moving) and the dashboard
on the other (your face being tracked via PC webcam). Both react to the same
voice commands simultaneously.

---

## 6. Posture Management

Store NAOqi posture joint angles in the controller:

```python
POSTURES = {
    "StandInit": {
        "HeadPitch": -0.17, "HeadYaw": 0.0,
        "LShoulderPitch": 1.41, "LShoulderRoll": 0.35,
        "LElbowYaw": -1.39, "LElbowRoll": -1.04,
        "LWristYaw": -0.01, "LHand": 0.28,
        "RShoulderPitch": 1.41, "RShoulderRoll": -0.35,
        "RElbowYaw": 1.39, "RElbowRoll": 1.04,
        "RWristYaw": 0.01, "RHand": 0.28,
        "LHipYawPitch": -0.17, "LHipRoll": 0.06,
        "LHipPitch": -0.44, "LKneePitch": 0.69,
        "LAnklePitch": -0.35, "LAnkleRoll": -0.06,
        "RHipYawPitch": -0.17, "RHipRoll": -0.06,
        "RHipPitch": -0.44, "RKneePitch": 0.69,
        "RAnklePitch": -0.35, "RAnkleRoll": 0.06,
    },
    "Sit": {
        "LHipPitch": -1.53, "LKneePitch": 2.11,
        "LAnklePitch": -1.18,
        "RHipPitch": -1.53, "RKneePitch": 2.11,
        "RAnklePitch": -1.18,
        # ... other joints at relaxed angles
    },
    "Crouch": {
        "LHipPitch": -1.04, "LKneePitch": 2.11,
        "LAnklePitch": -1.18,
        "RHipPitch": -1.04, "RKneePitch": 2.11,
        "RAnklePitch": -1.18,
        # ... arms at sides
    },
}

def go_to_posture(self, name, speed=0.5):
    """Interpolate to a target posture over time."""
    target = POSTURES.get(name)
    if not target:
        return
    for joint_name, angle in target.items():
        motor = self.robot.getDevice(joint_name)
        if motor:
            motor.setPosition(angle)
            motor.setVelocity(speed * motor.getMaxVelocity())
```

---

## 7. Object Grasping with Connectors

### Setup in World File

```
# On NAO's right hand: active connector
Nao {
  ...
  rightHand [
    Connector {
      name "hand_connector"
      type "active"
      model "grasp"
      autoLock FALSE
      distanceRotation 0.1  # 10cm snap range
    }
  ]
}

# On the phone object: passive connector
Solid {
  name "phone"
  children [
    Shape {
      appearance PBRAppearance { baseColor 0.1 0.1 0.1 }
      geometry Box { size 0.07 0.15 0.01 }  # phone-sized
    }
    Connector {
      name "phone_connector"
      type "passive"
      model "grasp"      # matches hand_connector model
      autoLock FALSE
    }
  ]
  physics Physics { mass 0.2 }  # 200g phone
}
```

### Pickup Sequence in Controller

```python
def handle_pickup_object(self):
    """Full pickup sequence using Connector."""
    hand_connector = self.robot.getDevice("hand_connector")
    hand_connector.enablePresence(self.timestep)

    # 1. Open hand
    self.set_hand("right", 1.0)
    self.wait_steps(10)  # ~0.3s

    # 2. Crouch
    self.go_to_posture("Crouch")
    self.wait_steps(50)  # ~1.5s

    # 3. Reach arm down
    self.set_arm_reach_down()
    self.wait_steps(30)  # ~1.0s

    # 4. Check if object is in range
    if hand_connector.getPresence():
        hand_connector.lock()  # Attach object to hand
        self.wait_steps(10)

    # 5. Close hand (visual only — connector does the holding)
    self.set_hand("right", 0.0)
    self.wait_steps(10)

    # 6. Carry position
    self.set_arm_carry()
    self.wait_steps(15)

    # 7. Stand
    self.go_to_posture("StandInit")
    self.wait_steps(60)  # ~2.0s

def handle_offer_object(self):
    """Offer and release."""
    hand_connector = self.robot.getDevice("hand_connector")

    self.set_arm_offer()
    self.wait_steps(100)  # ~3.0s

    # Release
    self.set_hand("right", 1.0)
    hand_connector.unlock()  # Detach object
    self.wait_steps(30)

    self.set_arm_rest()
    self.wait_steps(15)
```

---

## 8. World File — Virtual Room

```
# elderguard_room.wbt (Webots world file)

#VRML_SIM R2023b utf8

WorldInfo {
  basicTimeStep 16        # 16ms = 62.5 Hz physics
  coordinateSystem "NUE"  # Y-up
}

Viewpoint {
  position 3.0 2.0 3.0
  orientation -0.5 0.8 0.2 0.8
}

# Floor
RectangleArena {
  floorSize 4 4
  floorTileSize 0.5 0.5
  floorAppearance PBRAppearance {
    baseColor 0.8 0.8 0.8
    roughness 0.5
  }
  wallHeight 2.5
  wallAppearance PBRAppearance {
    baseColor 0.95 0.93 0.9
  }
}

# Lighting
CeilingLight {
  position 2.0 2.4 2.0
  pointLightIntensity 5
}

# NAO Robot
Nao {
  name "nao"
  translation 2.0 0.334 2.0
  rotation 0 1 0 0
  controller "webots_nao_controller"
  cameraWidth 320
  cameraHeight 240
}

# Virtual Person (with face texture)
Pedestrian {
  name "elderly_person"
  translation 2.0 1.27 3.0   # 1m in front of NAO
  controller "person_controller"
}

# Objects
Solid {
  name "phone"
  translation 1.5 0.01 1.5   # on the floor, to the left
  children [
    Shape {
      appearance PBRAppearance {
        baseColor 0.1 0.1 0.1
        metalness 0.8
      }
      geometry Box { size 0.07 0.01 0.15 }
    }
    Connector { name "phone_conn" type "passive" model "grasp" }
  ]
  physics Physics { mass 0.2 }
}

# Furniture
SolidBox {
  name "table"
  translation 1.0 0.35 1.0
  size 0.8 0.7 0.6
  appearance PBRAppearance { baseColor 0.55 0.35 0.15 }
}

SolidBox {
  name "chair"
  translation 1.0 0.25 1.6
  size 0.4 0.5 0.4
  appearance PBRAppearance { baseColor 0.3 0.2 0.1 }
}
```

---

## 9. TCP Protocol Mapping — Command Translation

The Webots controller receives the **exact same JSON commands** as the real
NAO server. Here's the full mapping:

| Brain Command | Webots Controller Action |
|---------------|-------------------------|
| `move_head(yaw, pitch, speed)` | `HeadYaw.setPosition(yaw)`, `HeadPitch.setPosition(pitch)` |
| `move_head_relative(d_yaw, d_pitch)` | Read current + add delta, set position |
| `walk_toward(x, y, theta)` | Select motion file, play, schedule stop |
| `set_walk_velocity(x, y, theta)` | Loop forward motion + heading adjustment |
| `stop_walk` | Stop all motion files |
| `stop_all` | Stop all motions + stop speaker |
| `say(text)` | `speaker.speak(text)` |
| `animated_say(text)` | `speaker.speak(text)` + gesture motion |
| `pose(name)` | Set all joints to posture preset |
| `rest` | Crouch posture + disable motors |
| `wake_up` | StandInit posture + enable motors |
| `animate(wave)` | Play wave motion file or set arm joints |
| `animate(dance)` | Play dance motion sequence |
| `open_hand(hand)` | Set `RHand`/`LHand` to 1.0 |
| `close_hand(hand)` | Set `RHand`/`LHand` to 0.0 |
| `arm_carry` | Set right arm joint angles to carry position |
| `arm_reach_down` | Set right arm to reach-down angles |
| `arm_offer` | Set right arm to offer position |
| `arm_rest` | Set right arm to neutral |
| `pickup_object` | Full pickup sequence with Connector |
| `offer_object` | Offer + unlock Connector |
| `query_state` | Return joint angles, posture, channel states |
| `heartbeat` | No-op, return state |
| `get_posture` | Return current posture name |

### Two-Phase ACK Protocol

The Webots controller implements the same two-phase ACK protocol:
1. Command received → send `{"status": "accepted", "id": "..."}` immediately
2. Command completes → send `{"status": "ok", "type": "done", "id": "..."}`

For instant commands (move_head, stop_walk): send single `{"status": "ok"}`.
For threaded commands (walk, pose, pickup): send ack then done.

---

## 10. Files to Create

| File | Location | Purpose |
|------|----------|---------|
| `webots/elderguard_room.wbt` | World file | Virtual room, NAO, person, objects |
| `webots/controllers/webots_nao_controller/webots_nao_controller.py` | Controller | TCP server, command handler, motor control |
| `webots/controllers/webots_nao_controller/tcp_handler.py` | Controller | TCP protocol (JSON parse, ack/done) |
| `webots/controllers/webots_nao_controller/postures.py` | Controller | Joint angle presets for Stand/Sit/Crouch |
| `webots/controllers/webots_nao_controller/walk_engine.py` | Controller | Motion file management for walking |
| `webots/controllers/person_controller/person_controller.py` | Controller | Virtual person behavior (walk paths, sit, fall) |
| `webots/textures/human_face.jpg` | Texture | Face image for pedestrian head |
| `webots/README.md` | Docs | Setup and usage instructions |
| `Simulation/adapters/webots_camera.py` | Adapter | Receives camera images from Webots via TCP |
| `Simulation/adapters/bootstrap_webots.py` | Bootstrap | Patches Camera → WebotsCamera |

---

## 11. Implementation Order

| Step | What | Effort | Depends On |
|------|------|--------|-----------|
| **1** | Install Webots, create empty world with NAO | 0.5 day | Nothing |
| **2** | Basic controller: read camera, TCP server, move_head | 1 day | Step 1 |
| **3** | Posture presets (Stand/Sit/Crouch) + pose command | 0.5 day | Step 2 |
| **4** | Camera streaming to brain via TCP (port 5556) | 1 day | Step 2 |
| **5** | Walking via motion files (walk_toward, stop_walk) | 1 day | Step 3 |
| **6** | Speech (say command → Speaker device) | 0.5 day | Step 2 |
| **7** | Virtual person + face detection test | 1 day | Step 4 |
| **8** | Object grasping with Connectors | 1 day | Step 5 |
| **9** | Full command set (all 22 actions) | 1 day | Steps 3-8 |
| **10** | Person behavior controller (walk paths, sit, fall) | 1 day | Step 7 |
| **11** | Integration testing (all 18 voice commands) | 1 day | Steps 1-10 |
| **12** | Documentation + demo scripts | 0.5 day | Step 11 |

**Total: ~10 days**

Steps 3, 4, 5, 6 can be partially parallelized after Step 2 is done.

---

## 12. Edge Cases

### Simulation Timing
| Edge Case | How We Handle |
|-----------|---------------|
| Brain sends commands faster than Webots can process | Command queue in TCP handler, process one per `robot.step()` |
| Motion file still playing when new walk command arrives | Stop current motion, start new one (same as real NAO's walk interruption) |
| `robot.step()` returns -1 (simulation ended) | Controller exits cleanly, brain's TCP reconnect detects disconnect |
| Slow physics step causes TCP timeout | Use `asyncio` or non-blocking sockets, brain's 5s TCP timeout is generous |

### Face Detection
| Edge Case | How We Handle |
|-----------|---------------|
| MediaPipe doesn't detect virtual face | Fall back to Option B (simulated coordinates from known 3D position) |
| Virtual person partially occluded by furniture | Same as real life — face lost, servo handles it (existing face-lost logic) |
| Virtual person leaves camera FOV | Person search kicks in (head scan + body rotation) — already implemented |
| Multiple virtual people | FaceTracker picks largest face (closest) — same as real system |

### Walking
| Edge Case | How We Handle |
|-----------|---------------|
| `walk_toward` with very small distance | Skip motion, immediately report done |
| `walk_toward` with diagonal path (x + theta) | Turn first, then walk forward (two sequential motions) |
| Robot walks into wall/furniture | Webots physics handles collision — robot stops. Walk timeout prevents infinite waiting. |
| `set_walk_velocity` continuous mode | Loop `Forwards50.motion`, periodically adjust heading between loops |
| Walk interrupted by `stop_walk` | `motion.stop()` freezes at current keyframe. Not perfectly smooth but acceptable. |

### Object Grasping
| Edge Case | How We Handle |
|-----------|---------------|
| Hand connector not close enough to object | `getPresence()` returns 0, grip fails. Robot says "Got it" but holds nothing. Same behavior as real system with no force sensors. |
| Object too heavy (physics) | Webots physics may prevent lifting. Set object mass to realistic values (phone=0.2kg). |
| Object falls during carry (physics) | Connector `lock()` creates rigid link — object won't fall unless `unlock()` is called. |
| Multiple objects near hand | Connector snaps to closest matching connector. One object at a time. |

### Communication
| Edge Case | How We Handle |
|-----------|---------------|
| Brain starts before Webots | Brain's auto-reconnect (exponential backoff, 20 retries) handles this |
| Webots restarts during session | Brain detects TCP disconnect, auto-reconnects, sends `query_state` to resync |
| Camera stream drops frames | `WebotsCamera.read()` returns latest frame (same as real camera). Missing frames are fine — servo doesn't need every frame. |
| Network port already in use | Controller checks port availability, logs clear error. Configurable via environment variable. |

### Person Behavior
| Edge Case | How We Handle |
|-----------|---------------|
| Virtual person walks through walls | Webots physics prevents this (collision detection enabled) |
| Virtual person falls (ragdoll) | Disable physics on pedestrian, use controlled animation instead. Fall is scripted, not physics-based. |
| Person controller crashes | Person stops moving. NAO's servo handles "face not moving" gracefully. |

---

## 13. Demo Scenarios in Webots

### Demo 1: Face Tracking
1. Launch Webots world + brain
2. Virtual person stands in front of NAO
3. Say "hey nao, follow me"
4. Virtual person walks left → NAO head follows
5. Audience sees: 3D NAO head turning smoothly in Webots

### Demo 2: Full Follow with Person Search
1. Say "hey nao, come here"
2. Virtual person walks to the side of NAO
3. NAO body turns to follow (visible walking in Webots)
4. Virtual person walks behind NAO
5. NAO loses face → head scan visible → body rotation visible
6. NAO finds person → resumes following

### Demo 3: Object Retrieval
1. Virtual phone is on the floor
2. Say "hey nao, find my phone"
3. NAO head scans the room in Webots (visible rotation)
4. YOLO detects the virtual phone in the camera feed
5. Say "hey nao, bring me my phone"
6. NAO walks to phone → crouches → picks it up (Connector locks)
7. NAO searches for person → walks to them
8. NAO extends arm → releases phone (Connector unlocks)

### Demo 4: Fall Detection
1. Virtual person standing, NAO monitoring
2. Person controller triggers "fall" (person goes to ground)
3. Fall monitor detects pose change → "Are you okay?"
4. Visible in both Webots (person on ground) and dashboard (fall alert)

### Demo 5: Hybrid — Real Voice + Virtual Robot
1. YOU speak into your mic (real voice, real Vosk, real ECAPA)
2. Virtual NAO in Webots responds to YOUR commands
3. Split screen: Webots 3D view + dashboard with your camera feed
4. Shows the full offline AI pipeline working end-to-end

---

## 14. Requirements

### Hardware
- Windows 10/11 64-bit
- CPU: Quad-core (for Webots physics + brain AI + camera processing)
- RAM: 8 GB minimum (Webots ~2 GB + brain ~500 MB + Python overhead)
- GPU: Dedicated NVIDIA/AMD recommended (Webots rendering + camera)
- Microphone: Any USB mic (for voice commands)

### Software
- Webots R2023b or later (free download from cyberbotics.com)
- Python 3.9+ (for brain + Webots controller)
- All existing pip dependencies (vosk, mediapipe, opencv, etc.)
- No additional pip packages needed for Webots (controller module is bundled)

### Installation Steps
1. Download and install Webots from https://cyberbotics.com
2. Set `WEBOTS_HOME` environment variable to Webots install directory
3. Copy `webots/` folder into the project
4. Launch Webots → Open `webots/elderguard_room.wbt`
5. In a separate terminal: run the brain
   ```bash
   python run_simulation.py --webots
   ```
6. Webots starts the NAO controller automatically
7. Brain connects via TCP to port 5555

---

## 15. Compatibility with Existing System

| Component | Change Needed |
|-----------|---------------|
| `rpi_brain/main.py` | **NONE** — unchanged |
| `rpi_brain/settings.py` | **NONE** — NAO_IP=127.0.0.1 already |
| `rpi_brain/servo/visual_servo.py` | **NONE** |
| `rpi_brain/vision/*` | **NONE** |
| `rpi_brain/audio/*` | **NONE** |
| `rpi_brain/comms/tcp_client.py` | **NONE** |
| `Simulation/run_simulation.py` | Add `--webots` flag to use WebotsCamera adapter |
| `Simulation/adapters/bootstrap.py` | Add Webots camera patch option |
| `nao_body/server.py` | **NOT USED** — replaced by Webots controller |
| `nao_body/motion_library.py` | **NOT USED** — reimplemented in Webots controller using Webots API |

**Key principle: The brain is unchanged. Only the "NAO side" is replaced.**
