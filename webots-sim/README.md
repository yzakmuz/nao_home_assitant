# ElderGuard — Webots 3D Simulation

3D physics-based simulation of the ElderGuard system using Webots R2023b
with a virtual NAO V5 robot in a furnished room.

## Requirements

- **Webots R2023b** (download from https://cyberbotics.com)
- **Python 3.9+** (same as the brain)
- Set `WEBOTS_HOME` environment variable to your Webots installation

## Quick Start

1. Open Webots R2023b
2. File → Open World → `webots-sim/worlds/elderguard_room.wbt`
3. The controller starts automatically (TCP server on port 5555)
4. Click the 3D view to focus it
5. Use keyboard controls OR connect the brain via TCP

### Connecting the Brain

The controller listens on TCP port 5555 and accepts the same JSON
commands as the real NAO server (`nao_body/server.py`). To connect:

```bash
# From the Simulation/ directory, run the brain:
python run_simulation.py
# Or connect any TCP client to localhost:5555
```

### Running Protocol Tests

While Webots is running, in a separate terminal:

```bash
cd webots-sim/controllers/elderguard_nao
python test_tcp_protocol.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| Up / Down | Walk forward / backward |
| Left / Right | Side step |
| Q / E | Turn left / right (40 deg) |
| Shift + Left/Right | Turn 60 degrees |
| T | Turn around (180 deg) |
| H / J | Head left / right |
| U / N | Head up / down |
| 0 | Center head |
| O / C | Open / close hands |
| W | Wave hello |
| Space | Stop all motion + stand |
| P | Print status |
| ? | Print help |

## TCP Protocol

The controller implements the full two-phase ACK protocol:

- **Inline commands** (`move_head`, `stop_walk`, `query_state`, etc.) return a single `{"status":"ok"}` response
- **Async commands** (`walk_toward`, `say`, `pose`, `animate`, etc.) return `{"status":"accepted", "type":"ack"}` immediately, then `{"status":"ok", "type":"done"}` when complete
- **Fire-and-forget** (`no_ack: true`) — no response (used for 15 Hz head tracking)
- Every response includes a `state` field with channel states and head angles

### Supported Actions (22)

| Action | Channel | Type |
|--------|---------|------|
| `move_head` | HEAD | inline |
| `move_head_relative` | HEAD | inline |
| `set_walk_velocity` | SYSTEM | inline |
| `stop_walk` | SYSTEM | inline |
| `stop_all` | SYSTEM | inline |
| `query_state` | SYSTEM | inline |
| `heartbeat` | SYSTEM | inline |
| `get_posture` | SYSTEM | inline |
| `walk_toward` | LEGS | async |
| `pose` | LEGS | async |
| `rest` | LEGS | async |
| `wake_up` | LEGS | async |
| `pickup_sequence` | LEGS | async |
| `say` | SPEECH | async |
| `animated_say` | SPEECH | async |
| `animate` | ARMS/LEGS | async |
| `open_hand` | ARMS | async |
| `close_hand` | ARMS | async |
| `arm_carry_position` | ARMS | async |
| `arm_reach_down` | ARMS | async |
| `arm_offer_position` | ARMS | async |
| `arm_rest_position` | ARMS | async |
| `offer_and_release` | ARMS | async |
| `get_person_position` | SYSTEM | inline |
| `follow_person` | SYSTEM | inline |
| `stop_follow` | SYSTEM | inline |

## Directory Structure

```
webots-sim/
├── worlds/
│   └── elderguard_room.wbt              # 5×5m room: NAO, table, chair, phone, bottle, person
├── controllers/
│   ├── elderguard_nao/
│   │   ├── elderguard_nao.py            # NAO controller (TCP + camera + tracking, Supervisor)
│   │   ├── tcp_handler.py               # Non-blocking TCP server (port 5555)
│   │   ├── camera_server.py             # Camera frame streaming (port 5556)
│   │   ├── webots_camera.py             # Brain-side camera adapter
│   │   ├── postures.py                  # Joint angle presets
│   │   ├── test_tcp_protocol.py         # 17 TCP protocol tests
│   │   ├── test_camera_stream.py        # Camera stream viewer
│   │   ├── test_face_detection.py       # Face detection test (Option A + B)
│   │   ├── test_follow_person.py        # Person following test
│   │   ├── test_pickup_demo.py          # Pickup sequence demo
│   │   └── runtime.ini                  # Python version config
│   └── person_standing/
│       └── person_standing.py           # Keeps Pedestrian standing (joint hold)
├── documentation.md                     # Full implementation documentation
└── README.md
```
