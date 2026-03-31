# Webots Simulation — Concept & Architecture Overview

**Date:** 2026-03-17
**Purpose:** Document how Webots can simulate the entire ElderGuard system

---

## 1. The Big Picture

Webots gives you a **virtual room** with a **virtual NAO robot** that has **virtual
cameras and microphones**. Your existing Python brain code connects to the virtual
NAO instead of the real one — and everything works the same way.

```
CURRENT (Real Hardware):
  PC Mic → Vosk STT → Brain (main.py) → TCP → Real NAO Robot
  Pi Camera → Face/YOLO → Brain                  ↓
                                          Real person walks around

WEBOTS (Virtual):
  PC Mic → Vosk STT → Brain (main.py) → TCP → Webots NAO Controller
  Webots Camera → Face/YOLO → Brain              ↓
                                          Virtual person walks around
                                          in a virtual room
```

**Your production code (`main.py`, `settings.py`, all of `rpi_brain/`) stays 100%
unchanged.** Only the NAO server side is replaced by a Webots controller.

---

## 2. What Webots Provides Out of the Box

**NAO V5 robot model (built-in):**
- Full articulated skeleton (25 degrees of freedom)
- Accurate joint limits matching real NAO
- Simulated cameras (top and bottom, same resolution as real NAO)
- Simulated microphones
- Walking engine (uses NAO's actual gait parameters)
- Physics — the robot can fall, pick up objects, sit on chairs
- Appearance — realistic white/gray NAO body

**Virtual environment:**
- 3D rooms with walls, floor, ceiling
- Furniture (tables, chairs, shelves — from Webots object library)
- Objects on the floor (phone, bottle, cup — you create or import these)
- Lighting (realistic shadows)
- Virtual humans (animated 3D pedestrian models)

---

## 3. How the Camera Works

### Option A: Webots Simulated Camera (Full Virtual)

Webots renders what the NAO's camera "sees" — a virtual image of the virtual room:

```
Virtual room with virtual person standing in front of NAO
    ↓
Webots renders the camera view (RGB image, 320×240 or 640×480)
    ↓
Your code receives this image as a NumPy array
    ↓
MediaPipe BlazeFace detects the virtual person's face
    ↓
PID servo tracks the virtual face → sends head commands → Webots moves NAO head
    ↓
Virtual camera angle changes → new image → closed loop
```

**Does MediaPipe detect faces on virtual humans?** Yes — if the virtual human
model has a realistic face texture. Webots' built-in pedestrian models have
photorealistic face textures that MediaPipe detects reliably. YOLO also detects
"person", "bottle", "cup" etc. on virtual objects with realistic textures.

### Option B: PC Camera Feed Injected (Hybrid)

You can also inject your **real PC webcam** feed into the simulation:

```
Real PC webcam captures YOU (real person)
    ↓
Image sent to your brain code as if it came from NAO's camera
    ↓
MediaPipe detects YOUR real face
    ↓
PID servo sends head commands → Webots NAO head moves
    ↓
The virtual NAO in Webots turns its head to track your real face
```

This is the **most impressive demo** — you stand in front of your PC camera,
say "follow me," and the virtual NAO in Webots turns its head to follow your
real face on screen.

### Option C: Both at the Same Time (Split Screen)

Show the PC camera feed on one side and the Webots 3D view on the other. The
audience sees your real face AND the virtual robot reacting to it simultaneously.

---

## 4. How Voice Commands Work

**Voice commands use your real PC microphone** — no simulation needed:

```
YOU speak into your PC mic: "hey nao, find my phone"
    ↓
Vosk STT recognizes the phrase (runs on your PC, same as now)
    ↓
ECAPA-TDNN verifies your voice (runs on your PC, same as now)
    ↓
Brain sends TCP command to Webots NAO controller
    ↓
Webots NAO says "Looking for your phone" (virtual TTS or PC TTS)
    ↓
Webots NAO scans head through angles (visible in 3D view)
    ↓
Webots camera captures the virtual room → YOLO detects virtual phone object
    ↓
Webots NAO says "I found your phone!"
```

**Voice is always real.** The simulation doesn't simulate your voice — it uses
your actual voice through the actual Vosk/ECAPA pipeline. This means the demo
shows real speaker verification working.

---

## 5. Virtual Person — "Follow Me" Demo

Webots has animated **pedestrian models** that can walk predefined paths:

```python
# In Webots world file: place a virtual person
Pedestrian {
    name "elderly_person"
    position 2.0 0 1.5
    controllerArgs "--speed 0.5 --trajectory circle"
}
```

**Scenario: "follow me" demo**

1. Virtual person stands in front of virtual NAO
2. You say "hey nao, follow me" into your real mic
3. Brain starts head-only tracking servo
4. Virtual NAO's camera sees the virtual person's face
5. MediaPipe detects the face → PID tracks it
6. Virtual person starts walking in a circle around the room
7. Virtual NAO head rotates to follow the virtual person
8. Console shows real-time: STT, verification, servo state, head angles

**Scenario: "come here" demo**

1. Same setup, but you say "hey nao, come here"
2. Brain starts full-follow mode (head + body walking)
3. Virtual NAO walks toward the virtual person
4. Virtual person moves — NAO follows
5. If person goes behind NAO → person search kicks in (head scan → body rotation)
6. NAO finds the person → resumes following
7. All visible in the 3D Webots view with realistic walking animation

---

## 6. "Bring Me My Phone" — Full Demo

**Setup in Webots:**
- Virtual room with table, chair, NAO robot standing
- Virtual phone object on the floor (3D model with realistic texture)
- Virtual person standing nearby

**Demo flow:**

1. You say: "hey nao, find my phone"
2. Virtual NAO scans the room (head rotation visible in 3D)
3. Webots camera captures the virtual phone → YOLO detects "cell phone"
4. NAO says: "I found your phone!"
5. You say: "hey nao, bring me my phone"
6. Virtual NAO walks toward the phone (visible walking in 3D)
7. NAO crouches (visible posture change in 3D)
8. NAO extends arm, closes hand (visible arm movement)
9. NAO stands up holding the phone (phone attached to hand)
10. NAO scans for person (head scan → body rotation visible)
11. NAO finds the virtual person → walks toward them
12. NAO extends arm → offers the phone
13. Phone drops from hand to the person's position

**All visible in a beautiful 3D environment** with realistic lighting, shadows,
and physics.

---

## 7. Architecture

```
┌──────────────────────────────────────────────────────────┐
│  YOUR PC                                                  │
│                                                           │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │ PC Microphone   │    │ Webots Simulation            │  │
│  │ (real voice)    │    │                              │  │
│  └────────┬────────┘    │  ┌────────┐  ┌───────────┐  │  │
│           │             │  │ Virtual │  │ Virtual   │  │  │
│  ┌────────▼────────┐    │  │  NAO   │  │  Person   │  │  │
│  │ Brain (main.py) │    │  │  Robot │  │  (walks)  │  │  │
│  │ Vosk STT        │    │  └────┬───┘  └───────────┘  │  │
│  │ ECAPA Verify    │    │       │                      │  │
│  │ Face Tracker    │◄───│───────┤ Camera image         │  │
│  │ YOLO Detector   │    │       │                      │  │
│  │ Fall Monitor    │    │  ┌────▼───────────────────┐  │  │
│  │ Visual Servo    │    │  │ NAO Controller         │  │  │
│  └────────┬────────┘    │  │ (Python, replaces      │  │  │
│           │             │  │  mock_server.py)        │  │  │
│     TCP JSON            │  │                         │  │  │
│           │             │  │ Receives: move_head,    │  │  │
│           └─────────────│──│ walk_toward, say,       │  │  │
│                         │  │ pickup_object, etc.     │  │  │
│                         │  │                         │  │  │
│                         │  │ Translates to Webots    │  │  │
│                         │  │ motor commands          │  │  │
│                         │  └─────────────────────────┘  │  │
│                         └───────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**The key component is the NAO Controller** — a Python script that:
1. Runs inside Webots as the NAO's controller
2. Listens on TCP port 5555 (same as the real/mock server)
3. Receives the exact same JSON commands your brain sends
4. Translates them to Webots motor API calls

---

## 8. NAO Controller Code Concept

```python
# webots_nao_controller.py (runs inside Webots)
from controller import Robot, Motor, Camera

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get NAO motors (same joint names as real NAO)
head_yaw_motor = robot.getDevice("HeadYaw")
head_pitch_motor = robot.getDevice("HeadPitch")
r_hand_motor = robot.getDevice("RHand")
# ... all 25 joints

# Get NAO camera
camera = robot.getDevice("CameraTop")
camera.enable(timestep)

# TCP server (same protocol as server.py)
server = TcpServer(port=5555)

while robot.step(timestep) != -1:
    # Read camera image → send to brain if requested
    image = camera.getImage()

    # Process incoming TCP commands
    cmd = server.receive()
    if cmd:
        action = cmd["action"]
        if action == "move_head":
            head_yaw_motor.setPosition(cmd["yaw"])
            head_pitch_motor.setPosition(cmd["pitch"])
        elif action == "walk_toward":
            # Use Webots locomotion
            robot.move(cmd["x"], cmd["y"], cmd["theta"])
        elif action == "pickup_object":
            # Crouch, reach, grab sequence using joint motors
            ...
        elif action == "say":
            robot.speak(cmd["text"])  # Webots TTS
```

---

## 9. What the Final Demo Would Look Like

**Screen layout during presentation:**

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌─────────────────────┐  ┌──────────────────────────────┐  │
│   │  Webots 3D View     │  │  Your Dashboard              │  │
│   │                     │  │  (OpenCV or Dear PyGui)       │  │
│   │  [Virtual room      │  │                              │  │
│   │   with NAO robot    │  │  Camera feed | State panel   │  │
│   │   walking toward    │  │  Robot view  | Console       │  │
│   │   virtual person    │  │  Audio bar   | Charts        │  │
│   │   carrying phone]   │  │                              │  │
│   │                     │  │                              │  │
│   └─────────────────────┘  └──────────────────────────────┘  │
│                                                              │
│   You speak into your mic: "hey nao, bring me my phone"      │
│   → Everything happens live in both windows simultaneously   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**The audience sees:**
- Left: Beautiful 3D Webots view of NAO walking in a room, picking up a phone
- Right: Your dashboard showing the AI pipeline (STT, verification, servo, YOLO)
- They hear: Your voice commands and NAO's responses
- They understand: The complete end-to-end offline AI system working in real-time

---

## 10. Webots vs Current Simulation — Comparison

| Feature | Current Simulation | Webots Simulation |
|---------|-------------------|-------------------|
| Robot visual | 3D wireframe (custom OpenCV) | Photorealistic NAO model |
| Environment | Black background | Furnished 3D room |
| Walking | State text + animation | Actual robot walking with physics |
| Object pickup | State text + arm angles | Robot physically picks up objects |
| Person | Face in camera only | Full 3D animated human |
| Person fall | Pose keypoints | Virtual person falls (ragdoll) |
| Camera | Real PC webcam | Simulated NAO camera (or real webcam hybrid) |
| Voice | Real mic (works) | Real mic (works, same) |
| Brain code | Real (unchanged) | Real (unchanged) |
| Installation | pip install | Webots installer + pip |
