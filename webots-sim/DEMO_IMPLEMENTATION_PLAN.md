# Demo Implementation Plan: Two Webots Simulations

**Date:** 2026-03-31
**Status:** Implementing
**Goal:** Create two standalone demo scripts that demonstrate NAO robot capabilities in Webots

---

## Overview

Two demos running against the existing Webots controller via TCP on port 5555:

1. **Demo 1 — Follow Person**: NAO tracks a person with its head and walks toward them
2. **Demo 2 — Phone Delivery**: NAO finds a phone on the ground, picks it up, walks to the person, delivers it

Both demos use **known 3D coordinates** from the Supervisor API (not camera-based detection).

---

## Architecture

```
Demo Script (external Python)          Webots Controller (elderguard_nao.py)
  TCP client (port 5555)                 TCP server (port 5555)
  Sends JSON commands ───────────────►   Dispatches actions
  Receives ack/done   ◄───────────────   Returns status + state
                                         Supervisor API: person + object positions
                                         GPS + IMU: NAO position + orientation
                                         Motion files: walk, turn, wave, dance
```

---

## What Already Works (No Changes Needed)

- `follow_person` action with `walk=true` — built-in head + walk tracking
- `get_person_position` — returns person AND NAO positions via Supervisor
- `pickup_sequence` — 7-phase: open hand, crouch, reach, close, carry, stand (10s)
- `offer_and_release` — 3-phase: offer arm, open hand, rest arm (5s)
- `walk_toward(x, y, theta)` — plans motion sequences
- All 22+ TCP actions, 17/17 protocol tests passing
- Head override prevents motion-file keyframes from fighting brain commands

---

## What Needs to Be Added

### Part 1: World File Changes (`elderguard_room.wbt`)

| Line | Current | Change To | Why |
|------|---------|-----------|-----|
| 236 | `Solid {` | `DEF PHONE Solid {` | Enable `getFromDef("PHONE")` for Supervisor queries |
| 261 | `Solid {` | `DEF BOTTLE Solid {` | Enable `getFromDef("BOTTLE")` for Supervisor queries |

Person already has `DEF PERSON` on line 289.

### Part 2: Controller Changes (`elderguard_nao.py`)

#### 2A. New State Variables (after existing tracking vars at line 306)

```python
# -- navigate_to tracking --
self._nav_active = False
self._nav_target = None          # [x, y, z] world coordinates
self._nav_stop_dist = 0.6        # stop when this close (meters)
self._nav_request_id = None      # for async "done" response
self._nav_step = 0               # update counter

# -- object carrying (Supervisor-based fake grasping) --
self._carrying_object = None     # Webots Node reference
self._carry_active = False
```

#### 2B. Object Node Lookup (after person lookup at line 316)

```python
self._object_nodes = {}
for def_name in ("PHONE", "BOTTLE"):
    node = self.getFromDef(def_name)
    if node:
        self._object_nodes[def_name] = node
```

#### 2C. New TCP Actions (5 total)

| Action | Channel | Type | Purpose |
|--------|---------|------|---------|
| `get_object_position` | SYSTEM | inline | Query any DEF'd object's world position |
| `navigate_to` | LEGS | async (ack+done) | Walk to arbitrary [x,y,z] coordinates |
| `stop_navigate` | SYSTEM | inline | Cancel current navigation |
| `start_carrying` | ARMS | async | Attach object to hand (Supervisor teleport) |
| `stop_carrying` | ARMS | async | Release carried object |

#### 2D. `navigate_to` — Core New Feature

**Modeled on existing `_update_person_tracking()`** (lines 1329-1436):
- Same coordinate convention: `horiz_dist = sqrt(dx^2 + dz^2)`, `world_angle = atan2(dx, dz)`
- Head update every 4 steps (~80ms), walk update every 20 steps (~400ms)
- Walk logic: turn body if `|head_yaw| > 0.5`, walk forward if `|head_yaw| < 0.4`
- On arrival (`horiz_dist < stop_dist`): stop legs, send async "done" with `arrived: true`
- Uses standard ack/done protocol (async action)
- Interrupts current legs activity before starting
- Mutually exclusive with `follow_person`

**Parameters:**
```json
{"action": "navigate_to", "x": 1.0, "y": 0.075, "z": 1.5, "stop_distance": 0.35, "id": "nav1"}
```

**Responses:**
```json
{"type": "ack", "status": "accepted", "id": "nav1", ...}
{"type": "done", "status": "ok", "id": "nav1", "arrived": true, "final_distance": 0.32}
```

#### 2E. Object Carrying — Supervisor-Based Fake Grasping

Since there are no Connector nodes on NAO's hands or on the objects:

1. `start_carrying(name)`: stores Supervisor Node reference
2. `_update_carrying()` (every step): reads NAO GPS + yaw, computes hand world position with fixed offset, teleports object via `setSFVec3f()` + `resetPhysics()`
3. `stop_carrying()`: releases object (stops teleporting, physics resumes)

**Hand offset from NAO torso** (tunable):
- Forward: ~0.0m, Up: ~0.05m, Right: ~-0.15m
- Rotated by NAO's yaw to get world coordinates

#### 2F. Main Loop Updates

Add after `_update_person_tracking()` (line 1474):
```python
self._update_navigation()
self._update_carrying()
```

#### 2G. State Validation

```python
# In can_execute():
if action == "navigate_to":
    if self.posture != "standing":
        return False, "must_stand_first"
```

#### 2H. Cleanup

`_stop_all_activity()` now also calls `_stop_navigation()` and `_stop_carrying()`.

---

### Part 3: Demo Script 1 — Follow Person

**File:** `demo_follow_person.py` (~120 lines)

#### Command Sequence

```
1. Connect TCP port 5555
2. query_state                    → verify standing
3. get_person_position            → print initial distance
4. say "I see you..."             → announce intent
5. follow_person walk=true        → start head+walk tracking
6. LOOP every 2s:
     get_person_position          → check distance
     if dist < 0.7m: break
7. stop_follow + stop_walk        → stop tracking
8. say "I am here..."             → announce arrival
9. animate wave                   → friendly wave
```

**Expected runtime:** ~15-30 seconds (NAO at ~2.8m from person, walks ~0.25m/s)

---

### Part 4: Demo Script 2 — Phone Delivery

**File:** `demo_phone_delivery.py` (~200 lines)

#### Command Sequence

```
Phase A: Init
1. Connect TCP port 5555
2. query_state                          → verify standing
3. get_person_position                  → store person position
4. get_object_position name=PHONE       → store phone position

Phase B: Navigate to Phone
5. say "I see your phone..."
6. navigate_to <phone pos> stop_distance=0.35
7. Wait for done (NAO walks to phone)

Phase C: Pick Up
8. say "Let me pick this up." (fire-and-forget)
9. pickup_sequence hand=right           → 10s pickup animation
10. start_carrying name=PHONE           → phone attaches to hand

Phase D: Navigate to Person
11. say "Got it! Bringing it to you."
12. navigate_to <person pos> stop_distance=0.6
13. Wait for done (NAO walks to person)

Phase E: Deliver
14. say "Here is your phone."
15. stop_carrying                       → release from hand
16. offer_and_release hand=right        → offer, open hand, rest (5s)

Phase F: Wrap Up
17. say "There you go!"
18. animate wave
```

**Expected runtime:** ~45-90 seconds

---

## Edge Cases Handled

| Edge Case | Solution |
|-----------|----------|
| NAO not standing | `can_execute()` rejects; demo checks `query_state` first |
| Walk overshoot (~0.1m) | Phone stop_dist=0.35m, person stop_dist=0.6m |
| Phone kicked by feet | Stop at 0.35m keeps feet ~0.25m from phone |
| Navigation cancelled mid-walk | `_stop_navigation()` sends done with `arrived=false` |
| Missing DEF name | `get_object_position` returns `object_found: false` |
| `navigate_to` + `follow_person` conflict | Mutual exclusion: each stops the other |
| `stop_all` during nav | Cleans up navigation + carrying |
| Phone jitter during carry | `resetPhysics()` every step zeros velocity |
| Already at target | First nav check (80ms) detects arrival immediately |

---

## File Summary

| File | Status | Lines Changed |
|------|--------|--------------|
| `worlds/elderguard_room.wbt` | Modify | 2 lines (add DEF) |
| `controllers/elderguard_nao/elderguard_nao.py` | Modify | ~200 lines added |
| `controllers/elderguard_nao/demo_follow_person.py` | New | ~120 lines |
| `controllers/elderguard_nao/demo_phone_delivery.py` | New | ~200 lines |

---

## Verification Checklist

- [ ] Webots prints "Object found: DEF PHONE" on startup
- [ ] `get_object_position` returns phone coordinates
- [ ] `navigate_to` walks NAO to target and sends done
- [ ] `start_carrying` teleports phone to hand
- [ ] Demo 1: NAO walks to person, waves
- [ ] Demo 2: NAO picks up phone, delivers to person
- [ ] `test_tcp_protocol.py` still passes 17/17
