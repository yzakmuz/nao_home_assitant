# ElderGuard Webots Simulation — Documentation

**Webots Version:** R2023b
**Date Started:** 2026-03-17
**Status:** Step 1 complete, Step 2 complete

---

## Step 1: Basic World + Keyboard-Controlled NAO — COMPLETED

### What Was Built

A Webots world file with a NAO V5 robot in a furnished room, controlled via
keyboard. This is the foundation for the full ElderGuard simulation.

### Files Created

| File | Purpose |
|------|---------|
| `worlds/elderguard_room.wbt` | Virtual room (5x5m) with NAO, table, chair, phone, bottle |
| `controllers/elderguard_nao/elderguard_nao.py` | NAO controller — initializes all 26 joints, loads 12 motion files, keyboard control |
| `controllers/elderguard_nao/runtime.ini` | Tells Webots to use Python 3.9+ |
| `README.md` | Quick start guide |

### World Contents

| Object | Position | Description |
|--------|----------|-------------|
| NAO V5 robot | Center (0, 0) | 25 DOF, CameraTop 320x240 enabled |
| Rectangle Arena | 5x5m | Textured floor + 2.5m walls, office theme |
| Table | (-1.5, 1.0) | Brown wood, 1.2x0.8x0.5m |
| Wooden Chair | (-1.5, 0.2) | Facing the table |
| Phone | (1.0, 1.5) | Dark box on floor, 200g, physics enabled |
| Bottle | (-0.8, -1.0) | Blue cylinder on floor, 300g, physics enabled |

### NAO Devices Initialized

| Device | Count | Names |
|--------|-------|-------|
| Joint motors | 26 | HeadYaw, HeadPitch, L/RShoulderPitch, L/RShoulderRoll, L/RElbowYaw, L/RElbowRoll, L/RWristYaw, L/RHand, L/RHipYawPitch, L/RHipRoll, L/RHipPitch, L/RKneePitch, L/RAnklePitch, L/RAnkleRoll |
| Finger phalanx | 16 | LPhalanx1-8, RPhalanx1-8 |
| Position sensors | 26 | All joints (suffix "S") |
| Cameras | 2 | CameraTop (320x240), CameraBottom |
| IMU sensors | 4 | accelerometer, gyro, gps, inertial unit |

### Motion Files Loaded (12)

| Key | File | Action |
|-----|------|--------|
| `forwards` | Forwards50.motion | Walk forward ~50cm |
| `backwards` | Backwards.motion | Walk backward |
| `side_left` | SideStepLeft.motion | Side step left |
| `side_right` | SideStepRight.motion | Side step right |
| `turn_left_40` | TurnLeft40.motion | Rotate left ~40 deg |
| `turn_right_40` | TurnRight40.motion | Rotate right ~40 deg |
| `turn_left_60` | TurnLeft60.motion | Rotate left ~60 deg |
| `turn_right_60` | TurnRight60.motion | Rotate right ~60 deg |
| `turn_left_180` | TurnLeft180.motion | Turn around 180 deg |
| `hand_wave` | HandWave.motion | Wave gesture |
| `stand_up_front` | StandUpFromFront.motion | Get up from face-down |
| `tai_chi` | TaiChi.motion | Tai Chi demo |

### Keyboard Controls

| Key | Action |
|-----|--------|
| Up / Down | Walk forward / backward |
| Left / Right | Side step left / right |
| Q / E | Rotate body left / right (~40 deg) |
| Shift + Left / Right | Rotate body left / right (~60 deg) |
| T | Turn around 180 deg |
| H / J | Head left / right (+/- 0.2 rad per press) |
| U / N | Head up / down (+/- 0.15 rad per press) |
| 0 | Center head |
| O / C | Open / close hands (all fingers) |
| W | Wave hello animation |
| Space | Stop all motion, return to StandInit |
| P | Print status (head angles, GPS position, camera) |
| ? | Print help |

### How to Run

1. Set environment variable: `WEBOTS_HOME = <path to Webots installation>`
2. Open Webots R2023b
3. File -> Open World -> `webots-sim/worlds/elderguard_room.wbt`
4. First time: wait for EXTERNPROTO downloads (needs internet)
5. NAO appears standing in the room
6. Click the 3D view to focus it
7. Use keyboard controls listed above
8. To see NAO camera: Overlays -> Camera Devices -> CameraTop

### Verified Working

- [x] NAO stands in the room
- [x] All 12 motion files load successfully
- [x] Forward/backward walking
- [x] Side stepping
- [x] Body rotation left/right (40 deg, 60 deg, 180 deg)
- [x] Head yaw/pitch control
- [x] Hand open/close (all fingers)
- [x] Wave animation
- [x] Stop all + return to stand
- [x] Camera feed visible via Webots overlay
- [x] Status print shows head angles, GPS position, camera bytes
- [x] Physics: phone and bottle on floor with gravity

---

## Step 2: TCP Server for Brain Commands — COMPLETED

### What Was Built

Full TCP command dispatch system that receives JSON commands from the
ElderGuard brain (main.py) and translates them to Webots motor/motion
calls. Implements the exact same protocol as the real NAO server
(nao_body/server.py), including two-phase ACK, state validation,
channel tracking, and all 22+ supported actions.

### Architecture

```
Brain Process (main.py)                Webots Controller (elderguard_nao.py)
  |                                      |
  | TCP JSON (port 5555)                 |
  | ---------------------------------->  |
  |   {"action":"move_head","yaw":0.5}   |  poll_commands() every robot.step()
  |                                      |  _dispatch() -> handler
  | <----------------------------------  |
  |   {"status":"ok","state":{...}}      |  send_response()
  |                                      |
  | fire-and-forget (no_ack=true)        |  (no response sent)
  | ---------------------------------->  |
  |                                      |
  | Long-running command:                |
  | ---------------------------------->  |  ack (immediate)
  | <--  ack {"status":"accepted"}       |  start motion/posture
  |   ...motion plays...                 |  _update_tasks() each step
  | <--  done {"status":"ok"}            |  task completes -> done
```

### Files Modified / Created

| File | Change |
|------|--------|
| `controllers/elderguard_nao/elderguard_nao.py` | Major rewrite — added TCP dispatch, NaoState, task tracking, 22 action handlers |
| `controllers/elderguard_nao/tcp_handler.py` | No changes (already complete from Step 1 prep) |
| `controllers/elderguard_nao/postures.py` | No changes (already complete from Step 1 prep) |
| `controllers/elderguard_nao/test_tcp_protocol.py` | New — 17 protocol verification tests |
| `README.md` | Updated with TCP protocol docs, all 22 actions, test instructions |

### Key Components Added

#### NaoState — Channel State Machine (mirrors real NaoStateMachine)

Tracks per-channel state matching the real NAO server:

| Channel | States | Purpose |
|---------|--------|---------|
| posture | standing / sitting / resting | Physical posture |
| head | idle | Head tracking (always idle on server) |
| legs | idle / walking / animating | Walk + pose + dance |
| speech | idle / speaking | TTS queue |
| arms | idle / animating | Wave + arm manipulation |

State validation (`can_execute`):
- `walk_toward` / `set_walk_velocity`: rejected if not standing or legs busy dancing
- `animate("dance")`: rejected if not standing, legs busy, or arms busy
- `animate("wave")`: rejected if arms busy
- `pickup_sequence`: rejected if not standing
- Other actions: always allowed

#### Three Async Task Types

| Type | Purpose | Completion Check |
|------|---------|------------------|
| **MotionTask** | Sequential motion file playback | `motion.isOver()` per file |
| **TimedTask** | Fixed-duration events (speech, posture) | `getTime() >= end_time` |
| **MultiPhaseTask** | Multi-step sequences (pickup, offer) | Phases fire at time offsets |

All tracked per-channel (`_legs_task`, `_speech_task`, `_arms_task`) and checked
every `robot.step()` in `_update_tasks()`.

#### Command Dispatch

Mirrors the real server's two-tier routing:

**Inline actions** (executed immediately, single response):

| Action | What it does |
|--------|-------------|
| `move_head` | Set HeadYaw/HeadPitch with joint limit clamping |
| `move_head_relative` | Read current + apply delta |
| `set_walk_velocity` | Start/stop looping forward motion (velocity mode) |
| `stop_walk` | Stop all leg motion, cancel legs task |
| `stop_all` | Stop everything, cancel all tasks, reset all channels |
| `query_state` | Return state snapshot + head angles |
| `heartbeat` | Keepalive, return state |
| `get_posture` | Return posture in state |

**Async actions** (ack immediately, done when task completes):

| Action | Channel | Implementation |
|--------|---------|---------------|
| `walk_toward` | LEGS | Plan motion sequence (turn + forward + side), play sequentially |
| `pose` | LEGS | Set posture joint angles, TimedTask 2s |
| `rest` | LEGS | Crouch posture, TimedTask 2s |
| `wake_up` | LEGS | StandInit posture, TimedTask 2s |
| `pickup_sequence` | LEGS | 7-phase MultiPhaseTask (open->crouch->reach->grab->carry->stand, 8s) |
| `say` | SPEECH | Print text, TimedTask (len*0.05s) |
| `animated_say` | SPEECH | Print text, TimedTask (len*0.06s) |
| `animate(wave)` | ARMS | Play HandWave.motion |
| `animate(dance)` | LEGS | Play TaiChi.motion (claims legs+arms) |
| `open_hand` | ARMS | Set phalanx + hand motors, TimedTask 0.5s |
| `close_hand` | ARMS | Set phalanx + hand motors, TimedTask 0.5s |
| `arm_carry_position` | ARMS | Set arm joints from ARM_CARRY, TimedTask 1s |
| `arm_reach_down` | ARMS | Set arm joints from ARM_REACH_DOWN, TimedTask 1s |
| `arm_offer_position` | ARMS | Set arm joints from ARM_OFFER, TimedTask 1s |
| `arm_rest_position` | ARMS | Set arm joints from ARM_REST, TimedTask 1s |
| `offer_and_release` | ARMS | 3-phase MultiPhaseTask (offer->open->rest, 5s) |

### Walking Implementation

Since Webots NAO has no `moveTo()` walk engine, walking is implemented via
pre-recorded motion files:

**`walk_toward(x, y, theta)` — Motion Sequence Planning:**

| Parameter | Threshold | Motion File |
|-----------|-----------|-------------|
| theta >= 2.5 rad | turn_left_180 | ~180 deg turn |
| theta >= 0.8 rad | turn_left_60 / turn_right_60 | ~60 deg turn |
| theta >= 0.35 rad | turn_left_40 / turn_right_40 | ~40 deg turn |
| x > 0.1 m | N * forwards | ~50cm per step |
| x < -0.1 m | backwards | Walk backward |
| abs(y) > 0.1 m | side_left / side_right | Side step |

The sequence executes in order (turn first, then forward, then side).
Each motion file plays to completion before the next starts.

**`set_walk_velocity(x, y, theta)` — Continuous Walking:**

When `x > 0.05`: loops the `Forwards50.motion` file continuously.
When velocity set to zero: stops the loop. Used by "come here" mode.

### Head Override Mechanism

Walk motion files include head keyframes that would fight the brain's
15 Hz PID servo commands. Solution: every `robot.step()`, the controller
re-applies `_desired_head_yaw` and `_desired_head_pitch` to override
any motion-file head positions. This ensures smooth PID tracking during
walking.

### Two-Phase ACK Protocol

Matches the real server exactly:

```
# Inline command:
-> {"action":"move_head", "yaw":0.5, "id":"r1"}
<- {"status":"ok", "action":"move_head", "state":{...}, "type":"done", "id":"r1"}

# Async command:
-> {"action":"walk_toward", "x":0.5, "id":"r2"}
<- {"status":"accepted", "action":"walk_toward", "state":{...}, "type":"ack", "id":"r2"}
   ...motion plays...
<- {"status":"ok", "action":"walk_toward", "state":{...}, "type":"done", "id":"r2"}

# Fire-and-forget:
-> {"action":"move_head", "yaw":0.3, "no_ack":true}
   (no response)

# Rejected:
-> {"action":"walk_toward", "x":0.5, "id":"r3"}   (while sitting)
<- {"status":"rejected", "reason":"must_stand_first", "state":{...}, "id":"r3", "type":"ack"}

# Backward compatible (no id):
-> {"action":"query_state"}
<- {"status":"ok", "action":"query_state", "state":{...}}
```

### State Snapshot Format

Every response includes a `state` field:

```json
{
  "posture": "standing",
  "head": "idle",
  "legs": "idle",
  "speech": "idle",
  "arms": "idle",
  "head_yaw": 0.0,
  "head_pitch": -0.17
}
```

For `ANGLE_ACTIONS` (move_head, query_state, heartbeat, get_posture),
`head_yaw` and `head_pitch` also appear at the top level of the response.

### Connection Handling

- **Brain connect**: Detected per-step, logged to console
- **Brain disconnect**: Stops all motion, cancels all tasks, resets channels
- **Client replacement**: New connection replaces old (same as real server)
- **Brain reconnect**: Auto-reconnect with `query_state` resync works seamlessly

### Walk Interruption

When a new `walk_toward` or `pose` command arrives while walking:
1. Current motion is stopped (`motion.stop()`)
2. Current legs task is cancelled
3. Velocity walk is stopped (if active)
4. New command starts fresh

This matches the real server's walk interruption logic (Phase 4).

### Edge Cases Handled

| Edge Case | How Handled |
|-----------|-------------|
| Brain sends commands faster than Webots processes | All commands processed each step (poll_commands returns list) |
| Motion file not found | Logged as warning, skipped in sequence |
| walk_toward with no movement (x=0, y=0, theta=0) | Done sent immediately (no motion played) |
| walk_toward with very small theta (< 0.35 rad) | No turn motion played (brain PID handles steering) |
| set_walk_velocity while walk_toward active | Legs task cancelled, velocity walk starts |
| stop_all during active tasks | All motions stopped, all tasks cancelled, channels reset |
| Speech without Speaker device | Text printed to console, timed completion |
| Head keyframes in walk motion | Overridden every step with _desired values |
| robot.step returns -1 (simulation ends) | TCP server stopped cleanly |
| Command without id field | Backward-compatible single response |
| Posture validation | Walk/dance rejected when sitting/resting |

### Test Script

`test_tcp_protocol.py` — 17 tests that can be run while Webots is active:

| Test | Verifies |
|------|----------|
| `test_query_state` | State snapshot with all fields + head angles |
| `test_heartbeat` | Keepalive returns ok |
| `test_get_posture` | Posture in state dict |
| `test_move_head` | Inline response with head angles |
| `test_move_head_relative` | Delta head adjustment |
| `test_fire_and_forget` | no_ack=true produces no response |
| `test_say_ack_done` | Two-phase: ack (speech=speaking) -> done (speech=idle) |
| `test_pose_sit_stand` | Sit -> walk rejected -> stand cycle |
| `test_walk_toward` | Forward walk ack/done with legs state tracking |
| `test_walk_toward_turn` | Turn motion selection |
| `test_walk_toward_empty` | Zero-distance walk completes immediately |
| `test_set_walk_velocity` | Velocity mode start + stop_walk |
| `test_stop_all` | All channels reset to idle |
| `test_animate_wave` | Wave animation ack/done on arms channel |
| `test_open_close_hand` | Hand control ack/done |
| `test_no_id_backward_compat` | Response without type field |
| `test_wake_up_rest` | Rest/wake_up posture transitions |

### How to Test

1. Open Webots and load `elderguard_room.wbt`
2. Wait for NAO to initialize (should see "TCP server on port 5555")
3. In a separate terminal:
   ```bash
   cd webots-sim/controllers/elderguard_nao
   python test_tcp_protocol.py
   ```
4. All 17 tests should pass

### Main Loop Architecture

```python
while robot.step(timestep) != -1:
    # 1. Detect connect/disconnect events
    _check_connection()

    # 2. Process all pending TCP commands (may be 0 or many)
    for cmd in tcp.poll_commands():
        _dispatch(cmd)

    # 3. Advance async tasks (check motion completion, fire phases)
    _update_tasks()

    # 4. Handle keyboard input (kept for manual testing)
    _handle_keyboard()
```

No background threads — everything runs in the `robot.step()` loop via
non-blocking `select()` in the TCP handler. This avoids Webots GIL
limitations where threads run ~100x slower.

### Verified Working

- [x] TCP server accepts connections on port 5555
- [x] Brain connects and receives state snapshots
- [x] move_head at 15 Hz (fire-and-forget) works smoothly
- [x] Two-phase ACK protocol (ack + done) for all async actions
- [x] walk_toward plays correct motion sequences
- [x] set_walk_velocity loops forward motion
- [x] stop_walk / stop_all cancel all activity
- [x] Posture transitions (sit/stand/rest/wake_up)
- [x] State validation rejects invalid commands
- [x] Walk interruption (new command cancels current)
- [x] Head override prevents walk-motion jitter
- [x] Disconnect stops all motion
- [x] Keyboard controls still work alongside TCP
- [x] say/animated_say with timed completion
- [x] animate (wave/dance) with motion files
- [x] Hand open/close and arm position commands
- [x] pickup_sequence multi-phase execution
- [x] offer_and_release multi-phase execution
- [x] Backward-compatible responses (no id field)
- [x] **17/17 TCP protocol tests pass** (robot stays balanced throughout)

### Physics Stability Lessons Learned

During Step 2 testing, the robot repeatedly fell during posture transitions
and walk commands. Root causes and fixes:

1. **SIT posture is unstable without a chair** — HipPitch=-1.53 shifts center
   of mass outside the support polygon. Fix: mapped "sit" to a shallow CROUCH
   (HipPitch=-0.58, only 0.14 rad from standing).

2. **Full-speed joint changes cause falls** — setting 24 joints to new angles
   at max motor velocity creates violent momentum shifts. Fix: `_go_to_posture()`
   accepts a `speed` parameter (30% of max for posture transitions).

3. **Motor speeds must be restored after transitions** — after a slow posture
   change, `_restore_motor_speeds()` resets all motors to max velocity AND
   re-applies STAND_INIT so walk motion files start from the exact expected pose.

4. **Stabilization time between commands** — the test script adds 2-3 second
   delays between motion commands to let physics settle. Tests ordered from
   safest (query_state) to riskiest (sit/stand cycle).

---

## Step 3: Posture Presets + Pose Command — COMPLETED (in Step 2)

Posture management was fully implemented as part of Step 2:
- STAND_INIT, CROUCH, arm positions (carry, reach, offer, rest) in `postures.py`
- `pose`, `rest`, `wake_up` TCP actions with slow transitions (30% motor speed)
- State tracking (standing/sitting/resting) with validation
- Stability-safe CROUCH posture (shallow squat, center of mass over feet)
- Motor speed restoration after transitions

---

## Step 4: Camera Streaming to Brain — COMPLETED

### What Was Built

TCP camera streaming server that sends the Webots NAO CameraTop frames to
an external process (the brain), enabling face tracking, object detection,
and fall detection to work with the virtual camera.

### Architecture

```
Webots Controller (port 5556)            Brain / Test Script
  camera_server.py                        webots_camera.py
  ┌───────────────────┐                  ┌───────────────────┐
  │ CameraTop         │  TCP stream      │ WebotsCamera      │
  │ getImage() BGRA   │ ──────────────►  │ .read() → BGR     │
  │ → convert to BGR  │ [w][h][pixels]   │ background thread  │
  │ every 4th step    │                  │ thread-safe lock   │
  │ (~15 FPS)         │                  │ same Camera API    │
  └───────────────────┘                  └───────────────────┘
```

### Why This Is Needed

The brain process (`main.py`) needs camera frames from the virtual world to:
1. **MediaPipe face tracking** → PID servo → `move_head` commands (follow person)
2. **YOLO object detection** → find phone/bottle/cup on "find my phone"
3. **MediaPipe Pose** → fall detection (detect if person fell)

Without camera streaming, the brain can't "see" the Webots world, so
vision-dependent commands (follow me, come here, find object) wouldn't work.

This replicates the real system architecture:
- Real: RPi Camera (hardware) → RPi Brain → TCP → NAO
- Webots: CameraTop → TCP stream (5556) → Brain → TCP commands (5555) → Webots NAO

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `camera_server.py` | 143 | Non-blocking TCP server, BGRA→BGR conversion, frame streaming |
| `webots_camera.py` | 160 | Brain-side adapter, background receive thread, Camera.read() API |
| `test_camera_stream.py` | 109 | Visual test with OpenCV display and FPS overlay |

### Files Modified

| File | Change |
|------|--------|
| `elderguard_nao.py` | Added `CameraStreamServer` init (port 5556), `_stream_camera_frame()` called every 4th step in main loop, clean shutdown |

### Protocol

Each frame sent over TCP:

| Field | Size | Format |
|-------|------|--------|
| Width | 2 bytes | uint16 big-endian |
| Height | 2 bytes | uint16 big-endian |
| Pixel data | W x H x 3 bytes | Raw BGR (blue, green, red) |

For 320x240: 4-byte header + 230,400 bytes = 230,404 bytes per frame.

### CameraStreamServer (Webots side)

- Non-blocking TCP server on port 5556 (configurable via `NAO_CAM_PORT` env var)
- Uses `select()` with timeout=0 — no threads, polled from `robot.step()` loop
- Single client connection (new client replaces old, like `tcp_handler.py`)
- BGRA→BGR conversion via numpy (`np.ascontiguousarray(bgra[:,:,:3])`)
- Pure Python fallback if numpy is unavailable
- Frame sending with 100ms socket timeout — skips frame on slow client
- Graceful disconnect handling on send failure

### WebotsCamera (Brain side)

Drop-in replacement for `vision.camera.Camera` and `PcCamera`:

| Method | Behavior |
|--------|----------|
| `start() → bool` | Connects to camera server, starts background receive thread |
| `stop()` | Stops thread, closes socket, clears frame |
| `read() → Optional[np.ndarray]` | Returns latest BGR frame copy (thread-safe) or None |
| `is_running` property | True while connected |
| `frame_size` property | Returns `(width, height)` tuple |
| `fps` property | Approximate received frames-per-second |

Background thread continuously receives frames (header → pixel data → reshape
to numpy array → store with lock). Connection loss detected automatically.

### Frame Rate

- Camera enabled at `4 * time_step` = 64ms in Webots
- Frames streamed every 4th `robot.step()` call = every 64ms = ~15.6 FPS
- Measured: **63.7 FPS** (Webots sends faster than expected due to
  simulation speed; the receiver handles any rate)

### How to Test

1. Start Webots simulation (console shows "Camera stream server on port 5556")
2. In a separate terminal:
   ```bash
   cd webots-sim/controllers/elderguard_nao
   python test_camera_stream.py
   ```
3. OpenCV window appears showing NAO's CameraTop view with FPS overlay
4. Move the NAO's head (keyboard H/J/U/N) — camera view follows
5. Press `q` to quit

Headless mode (no display): `python test_camera_stream.py --no-display`

### Edge Cases Handled

| Edge Case | How Handled |
|-----------|-------------|
| Brain connects before Webots | `WebotsCamera.start()` returns False, brain can retry |
| Brain disconnects during stream | `send_frame()` catches error, cleans up client |
| Camera not ready (getImage=None) | Frame skipped silently |
| Slow client can't keep up | Socket timeout (100ms), frame skipped |
| Multiple clients connect | New client replaces old (single client model) |
| Webots simulation paused | No new frames sent; `read()` returns last frame |
| numpy not installed | Pure Python BGRA→BGR fallback (slower) |
| Simulation ends (step=-1) | `cam_server.stop()` called in shutdown |

### Verified Working

- [x] Camera stream server starts on port 5556
- [x] Client connects and receives frames
- [x] BGRA→BGR conversion correct (colors look right)
- [x] Resolution: 320x240 (matching real Pi camera)
- [x] FPS: ~15-64 FPS depending on simulation speed
- [x] OpenCV display shows live NAO camera view
- [x] Head movement reflected in camera stream
- [x] Clean disconnect handling (no crashes)
- [x] TCP command tests (17/17) still pass with camera server active

---

## Step 5: Walking via Motion Files — COMPLETED (in Step 2)

Walk command translation was included in Step 2 implementation.

---

## Step 6: Speech — COMPLETED (in Step 2)

Text-to-speech prints to console; protocol (ack/done) fully works.

---

## Step 7: Virtual Person + Face Detection + Person Tracking — COMPLETED

### What Was Built

A virtual Pedestrian standing in the room, with a Supervisor-based position
tracking system that enables the NAO to follow the person. Tested both
camera-based face detection and geometric 3D position tracking.

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `controllers/person_standing/person_standing.py` | 14 | Minimal controller that keeps Pedestrian joints held (prevents gravity collapse) |
| `test_face_detection.py` | 274 | Tests Option A (Haar cascade on camera) and Option B (position query) |
| `test_follow_person.py` | 133 | Tests built-in person tracking (head + walk) |

### Files Modified

| File | Change |
|------|--------|
| `worlds/elderguard_room.wbt` | Added Pedestrian EXTERNPROTO, `DEF PERSON Pedestrian` with `person_standing` controller, `supervisor TRUE` on NAO |
| `elderguard_nao.py` | Changed `Robot` → `Supervisor`, added person node reference, `get_person_position` action (returns person + NAO positions + orientation), `follow_person` / `stop_follow` actions, built-in `_update_person_tracking()` in main loop |

### Face Detection Results

| Method | Result | Viability |
|--------|--------|-----------|
| **Option A: OpenCV Haar Cascade** | 0/150 frames detected | NOT VIABLE — 3D rendered face too low-poly |
| **Option A: MediaPipe BlazeFace** | API incompatible (v0.10.32 removed `solutions`) | NOT TESTED |
| **Option B: Supervisor 3D Position** | Person found at exact coordinates | **WORKING — chosen approach** |

**Decision:** Option B (geometric position from Supervisor API) is the correct approach
for Webots simulation. It's 100% reliable because it uses exact 3D geometry,
not computer vision on rendered faces.

### Pedestrian PROTO Challenges

The Webots Pedestrian PROTO is a Robot with articulated joints. Key issues solved:

1. **Upside-down person**: Without a controller, gravity collapses the joints and
   the person falls apart. Fix: `person_standing` controller calls `robot.step()`
   in a loop, keeping motors in position-hold mode.

2. **Position convention**: The Pedestrian's `translation` Y=1.27 places the body
   center at 1.27m (standard Webots convention for human models).

### Built-in Person Tracking System

The NAO controller now has an internal tracking loop (`_update_person_tracking()`)
that runs every 4 simulation steps (~60ms):

```
Every 60ms:
  1. Read person position (Supervisor API — exact, no TCP latency)
  2. Read NAO position (GPS sensor — direct access)
  3. Read NAO orientation (inertial unit — direct access)
  4. Compute relative angle: atan2(dx, dz) - nao_yaw
  5. Set head joints to face person (clamped to joint limits)
  6. If walk enabled and distance > 0.6m:
     - If angle > 0.5 rad: play turn motion
     - If roughly facing: play forward walk motion
  7. Head pitch adjusted for height difference
```

**Why this is inside the controller (not an external script):**
- Direct sensor access — no TCP round-trip delays
- No angle convention confusion — sensors are in the controller's frame
- Closed-loop at 60ms — much faster than TCP-based navigation (~800ms)
- Runs alongside all other commands (keyboard, TCP) seamlessly

### New TCP Actions Added

| Action | Type | Parameters | Behavior |
|--------|------|-----------|----------|
| `get_person_position` | inline | (none) | Returns person 3D position, NAO position, NAO orientation |
| `follow_person` | inline | `walk` (bool) | Starts built-in head tracking (+ walk if `walk=true`) |
| `stop_follow` | inline | (none) | Stops person tracking |

### How to Test

```bash
# Head tracking only (NAO looks at person):
python test_follow_person.py

# Head + walk (NAO approaches person):
python test_follow_person.py --walk

# Stop tracking:
python test_follow_person.py --stop

# Face detection comparison (Option A vs B):
python test_face_detection.py
```

### Verified Working

- [x] Pedestrian stands correctly in the room (person_standing controller)
- [x] `get_person_position` returns accurate 3D coordinates
- [x] NAO + person positions available via single TCP query
- [x] `follow_person` activates built-in tracking loop
- [x] Head tracks person direction in real-time
- [x] Walk mode: NAO turns and walks toward person
- [x] `stop_follow` cleanly stops tracking
- [x] TCP protocol tests (17/17) still pass
- [x] Supervisor API reads person position every step

---

## Next Steps

### Step 8: Object grasping with Connectors
Add Connector nodes to NAO's hand and objects (phone, bottle) so
the pickup_sequence can physically attach/detach objects.

### Step 10: Person behavior controller
Controller for the virtual person — walk paths, sit down, simulate
falls — so the brain's fall detection and person search can be tested.

### Step 11: Integration testing (brain + Webots + voice)
Connect the brain process to the Webots controller. Run the full
pipeline: voice → Vosk STT → FSM → TCP → Webots NAO.

### Step 12: Documentation + demo scripts

### Remaining: 4 steps (8, 10, 11, 12)
