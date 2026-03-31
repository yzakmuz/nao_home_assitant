# ElderGuard Webots Simulation — User Manual

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setting Up Webots](#2-setting-up-webots)
3. [Starting the Simulation](#3-starting-the-simulation)
4. [Understanding the TCP Protocol](#4-understanding-the-tcp-protocol)
5. [Running Demo 1: Follow Person](#5-running-demo-1-follow-person)
6. [Running Demo 2: Phone Delivery](#6-running-demo-2-phone-delivery)
7. [Manual Keyboard Controls](#7-manual-keyboard-controls)
8. [Sending TCP Commands Manually](#8-sending-tcp-commands-manually)
9. [Running Protocol Tests](#9-running-protocol-tests)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| **Webots** | R2023b | Download from https://cyberbotics.com |
| **Python** | 3.9+ | Must be in PATH |
| **numpy** | any | `pip install numpy` |
| **Operating System** | Windows 10/11 | Tested on Windows 10 Pro |

### Setting the WEBOTS_HOME Environment Variable

Webots must know where its installation is:

```bash
# Windows (PowerShell)
$env:WEBOTS_HOME = "C:\Program Files\Webots"

# Or set permanently via System Properties > Environment Variables
# Variable name: WEBOTS_HOME
# Variable value: C:\Program Files\Webots
```

---

## 2. Setting Up Webots

### First-Time Setup

1. Install Webots R2023b from https://cyberbotics.com/download
2. Open Webots
3. File > Open World > navigate to:
   ```
   webots-sim/worlds/elderguard_room.wbt
   ```
4. **First launch only:** Webots will download EXTERNPROTO files (NAO model, pedestrian, furniture). This requires internet and takes 1-2 minutes.
5. Wait for the console to show:
   ```
   [ElderGuard NAO] Controller initialized (TCP + Camera + Supervisor)
   [ElderGuard NAO] TCP command server on port 5555
   [ElderGuard NAO] Camera stream server on port 5556
   ```

### What You Should See

A virtual room (5x5 meters) containing:
- **NAO V5 robot** — white humanoid robot standing in the center
- **Table + Chair** — wooden furniture in the corner
- **Phone** — small dark object on the floor
- **Bottle** — blue cylinder on the floor
- **Person** — virtual pedestrian standing in the room

---

## 3. Starting the Simulation

### Step 1: Open the World

1. Launch Webots R2023b
2. File > Open World > `webots-sim/worlds/elderguard_room.wbt`
3. The simulation starts automatically

### Step 2: Verify the Controller

The Webots console (bottom panel) should show:

```
[ElderGuard NAO] Starting controller...
[ElderGuard NAO] Virtual person found (DEF PERSON)
[ElderGuard NAO] Object found: DEF PHONE
[ElderGuard NAO] Object found: DEF BOTTLE
[ElderGuard NAO] Controller initialized (TCP + Camera + Supervisor)
[ElderGuard NAO] TCP command server on port 5555
[ElderGuard NAO] Camera stream server on port 5556
```

### Step 3: Verify TCP Server

The TCP server is ready when you see `port 5555`. From a terminal:

```bash
# Quick test (Python one-liner)
python -c "
import socket, json
s = socket.socket()
s.connect(('127.0.0.1', 5555))
s.send((json.dumps({'action':'query_state','id':'t1'})+'\n').encode())
print(s.recv(4096).decode())
s.close()
"
```

You should see a JSON response with `"status": "ok"` and the current state.

### Step 4: Reset the Simulation (if needed)

If the robot has fallen or objects are out of place:
- Press **Ctrl+Shift+T** in Webots to reset the simulation
- Or: Simulation > Reset from the menu

---

## 4. Understanding the TCP Protocol

### Connection

- **Host:** `127.0.0.1` (localhost)
- **Port:** `5555` (commands), `5556` (camera stream)
- **Format:** Newline-delimited JSON (`{"action":"...", ...}\n`)

### Command Types

**Inline commands** — single immediate response:
```json
// Request:
{"action": "query_state", "id": "q1"}
// Response:
{"status": "ok", "action": "query_state", "type": "done", "id": "q1", "state": {...}}
```

**Async commands** — ack first, done when complete:
```json
// Request:
{"action": "say", "text": "Hello!", "id": "s1"}
// Response 1 (immediate):
{"status": "accepted", "action": "say", "type": "ack", "id": "s1", "state": {...}}
// Response 2 (when speech finishes):
{"status": "ok", "action": "say", "type": "done", "id": "s1", "state": {...}}
```

**Fire-and-forget** — no response:
```json
{"action": "move_head", "yaw": 0.5, "pitch": 0.0, "no_ack": true}
```

### Available Actions

| Action | Type | Description |
|--------|------|-------------|
| `query_state` | inline | Get full state + head angles |
| `heartbeat` | inline | Keepalive |
| `get_posture` | inline | Current posture |
| `get_person_position` | inline | Person + NAO positions |
| `get_object_position` | inline | Named object position |
| `move_head` | inline | Set head yaw/pitch |
| `move_head_relative` | inline | Delta head adjustment |
| `set_walk_velocity` | inline | Continuous velocity walk |
| `stop_walk` | inline | Stop legs only |
| `stop_all` | inline | Emergency stop all |
| `follow_person` | inline | Start person tracking |
| `stop_follow` | inline | Stop person tracking |
| `stop_navigate` | inline | Cancel navigation |
| `navigate_to` | async | Walk to [x,y,z] coordinates |
| `say` | async | Speak text |
| `animated_say` | async | Animated speech |
| `walk_toward` | async | Walk relative distance |
| `pose` | async | Change posture (stand/sit) |
| `rest` | async | Crouch and rest |
| `wake_up` | async | Stand up from rest |
| `animate` | async | Wave or dance |
| `pickup_sequence` | async | Full pickup animation |
| `offer_and_release` | async | Offer + release object |
| `open_hand` / `close_hand` | async | Hand control |
| `arm_carry/reach/offer/rest` | async | Arm positions |
| `start_carrying` | async | Attach object to hand |
| `stop_carrying` | async | Release carried object |

### State Fields in Every Response

```json
"state": {
    "posture": "standing",    // standing / sitting / resting
    "head": "idle",
    "legs": "idle",           // idle / walking / animating
    "speech": "idle",         // idle / speaking
    "arms": "idle"            // idle / animating
}
```

---

## 5. Running Demo 1: Follow Person

### What It Does

NAO detects the virtual person's position, announces its intent, then walks toward the person while tracking with its head. Upon arrival, NAO announces it has arrived and waves.

### How to Run

1. Make sure Webots is running with the world loaded (see Section 3)
2. Open a terminal and navigate to:
   ```bash
   cd webots-sim/controllers/elderguard_nao
   ```
3. Run the demo:
   ```bash
   python demo_follow_person.py
   ```

### What You'll See

```
========================================
  Demo 1: Follow Person
========================================
[1/9] Querying state...          OK (standing)
[2/9] Getting person position... Person at (0.18, 1.70, 1.30), NAO at (-0.07, -1.15, 0.32)
      Distance: 2.86m
[3/9] Saying: "I see you..."     OK
[4/9] Starting follow (HEAD+WALK)... OK
[5/9] Walking toward person...
      Distance: 2.45m
      Distance: 2.01m
      Distance: 1.53m
      ...
      Distance: 0.65m (close enough!)
[6/9] Stopping follow...         OK
[7/9] Saying: "I am here..."     OK
[8/9] Waving hello...            OK
[9/9] Done!
========================================
```

### Duration

~15-30 seconds depending on initial distance.

---

## 6. Running Demo 2: Phone Delivery

### What It Does

NAO locates the phone on the ground, walks to it, performs a crouching pickup sequence, then carries the phone to the person and offers it with an outstretched arm.

### How to Run

1. Make sure Webots is running with the world loaded (see Section 3)
2. Open a terminal:
   ```bash
   cd webots-sim/controllers/elderguard_nao
   python demo_phone_delivery.py
   ```

### What You'll See

```
========================================
  Demo 2: Phone Delivery
========================================
Phase A: Initialization
  [1] State: standing
  [2] Person at (0.18, 1.70, 1.30)
  [3] Phone at (1.00, 1.50, 0.07)

Phase B: Navigate to Phone
  [4] "I see your phone on the floor..."
  [5] Navigating to phone (stop at 0.35m)...
      Walking... distance: 1.82m
      Walking... distance: 1.35m
      ...
      Arrived! (0.33m)

Phase C: Pick Up Phone
  [6] "Let me pick this up."
  [7] Running pickup sequence (10s)...   OK
  [8] Attaching phone to hand...          OK

Phase D: Navigate to Person
  [9]  "Got it! Bringing it to you."
  [10] Navigating to person (stop at 0.6m)...
       Walking... distance: 2.15m
       ...
       Arrived! (0.58m)

Phase E: Deliver Phone
  [11] "Here is your phone."
  [12] Releasing phone...                 OK
  [13] Offering and releasing (5s)...     OK

Phase F: Complete
  [14] "There you go!"
  [15] Waving goodbye...                  OK
========================================
```

### Duration

~45-90 seconds (two navigation walks + pickup + delivery).

---

## 7. Manual Keyboard Controls

Click the 3D view in Webots to focus it, then use these keys:

### Movement

| Key | Action |
|-----|--------|
| **Up** / **Down** | Walk forward / backward |
| **Left** / **Right** | Side step left / right |
| **Q** / **E** | Rotate body left / right (40 deg) |
| **Shift+Left** / **Shift+Right** | Rotate body 60 deg |
| **T** | Turn around 180 deg |

### Head

| Key | Action |
|-----|--------|
| **H** / **J** | Head left / right |
| **U** / **N** | Head up / down |
| **0** | Center head |

### Actions

| Key | Action |
|-----|--------|
| **W** | Wave hello |
| **O** / **C** | Open / close hands |
| **Space** | Stop all motion, return to standing |
| **P** | Print status (head angles, GPS) |
| **?** | Print help |

---

## 8. Sending TCP Commands Manually

### Using Python

```python
import socket, json, time

def send(sock, cmd):
    """Send a command and print the response."""
    if "id" not in cmd:
        cmd["id"] = "manual_%d" % int(time.time())
    msg = json.dumps(cmd) + "\n"
    sock.send(msg.encode())
    resp = sock.recv(4096).decode()
    for line in resp.strip().split("\n"):
        print(json.dumps(json.loads(line), indent=2))
    return json.loads(resp.strip().split("\n")[0])

# Connect
sock = socket.socket()
sock.connect(("127.0.0.1", 5555))

# Examples:
send(sock, {"action": "query_state"})
send(sock, {"action": "say", "text": "Hello!"})
send(sock, {"action": "get_person_position"})
send(sock, {"action": "get_object_position", "name": "PHONE"})
send(sock, {"action": "move_head", "yaw": 0.5, "pitch": 0.0})
send(sock, {"action": "walk_toward", "x": 0.5, "y": 0, "theta": 0})
send(sock, {"action": "follow_person", "walk": True})
send(sock, {"action": "navigate_to", "x": 1.0, "y": 0.075, "z": 1.5, "stop_distance": 0.5})
send(sock, {"action": "stop_all"})

sock.close()
```

### Common Command Recipes

**Make NAO wave:**
```json
{"action": "animate", "name": "wave", "id": "w1"}
```

**Make NAO say something:**
```json
{"action": "say", "text": "Hello! I am NAO.", "id": "s1"}
```

**Make NAO sit down then stand up:**
```json
{"action": "pose", "name": "sit", "id": "p1"}
// wait 3 seconds...
{"action": "pose", "name": "stand", "id": "p2"}
```

**Make NAO dance:**
```json
{"action": "animate", "name": "dance", "id": "d1"}
```

**Check where things are:**
```json
{"action": "get_person_position", "id": "gp1"}
{"action": "get_object_position", "name": "PHONE", "id": "go1"}
```

---

## 9. Running Protocol Tests

The test suite verifies all 17+ TCP actions work correctly:

```bash
cd webots-sim/controllers/elderguard_nao

# Run all protocol tests (Webots must be running)
python test_tcp_protocol.py

# Run camera stream test (opens OpenCV window)
python test_camera_stream.py

# Run person tracking test
python test_follow_person.py           # head only
python test_follow_person.py --walk    # head + walk

# Run pickup demo test
python test_pickup_demo.py             # manual step-by-step
python test_pickup_demo.py --builtin   # built-in sequence
```

Expected output for protocol tests:
```
Test  1: query_state ................ PASS
Test  2: heartbeat .................. PASS
...
Test 17: wake_up + rest cycle ....... PASS
==========================================
  RESULTS: 17/17 passed, 0 failed
==========================================
```

---

## 10. Troubleshooting

### "Connection refused" when running demo scripts

The Webots controller hasn't started yet. Make sure:
1. Webots is open with `elderguard_room.wbt` loaded
2. The simulation is running (not paused)
3. Console shows "TCP command server on port 5555"

### NAO falls over during posture change

This is a known physics issue. The controller uses slow posture transitions (30% motor speed) to prevent falls. If it happens:
1. Press **Ctrl+Shift+T** in Webots to reset
2. Re-run the demo

### "Object found: DEF PHONE" not printed

The world file is missing the `DEF PHONE` keyword on the phone Solid. Verify:
```
DEF PHONE Solid {
  ...
  name "phone"
```

### Demo hangs waiting for "done"

The robot may be stuck (e.g., bumped into furniture). Solutions:
1. Press Ctrl+C to kill the demo script
2. Reset the simulation (Ctrl+Shift+T)
3. Increase timeout values in the demo script

### Phone doesn't appear picked up (Demo 2)

The `start_carrying` command uses Supervisor-based teleportation. If the phone doesn't visually attach:
1. Verify `DEF PHONE` exists in the world file
2. Check the carry offset values in the controller
3. The phone may need positional tuning

### NAO walks past the target

Walk overshoot of ~0.1m is normal (the check runs every 400ms). If it's excessive:
1. Reduce the walk speed or increase the stop distance
2. The demo compensates with generous stop distances (0.35m for phone, 0.6m for person)

### Camera stream test shows no image

The camera server runs on port 5556. Make sure:
1. No other program is using port 5556
2. OpenCV is installed: `pip install opencv-python`
3. The camera is enabled in the world file (CameraTop, 320x240)

---

## Quick Reference Card

```
Start Simulation:
  1. Open Webots R2023b
  2. File > Open World > webots-sim/worlds/elderguard_room.wbt
  3. Wait for "TCP command server on port 5555"

Run Demos:
  cd webots-sim/controllers/elderguard_nao
  python demo_follow_person.py      # Demo 1: Follow Person
  python demo_phone_delivery.py     # Demo 2: Phone Delivery

Reset:
  Ctrl+Shift+T in Webots

Keyboard Controls:
  Arrows=Walk  Q/E=Turn  H/J=Head  W=Wave  Space=Stop  ?=Help

TCP Connection:
  Host: 127.0.0.1  Port: 5555 (commands)  Port: 5556 (camera)
  Format: {"action":"...", "id":"..."}\n
```
