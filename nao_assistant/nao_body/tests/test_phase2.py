#!/usr/bin/env python3
"""
test_phase2.py -- Standalone integration test for Phase 2 changes.

Mocks NAOqi and motions to test protocol logic, state machine,
two-phase ACK, and BUG 4 fix without needing real hardware.
"""

import json
import os
import sys
import threading
import time
import types

# Ensure nao_body/ is on sys.path (tests now live in nao_body/tests/)
_NAO_BODY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _NAO_BODY_DIR)

# ============================================================
# Mock NAOqi + motions so server.py can load standalone
# ============================================================
class MockMotion(object):
    def __init__(self):
        self._angles = {"HeadYaw": 0.0, "HeadPitch": 0.0}
        self._move_active = False

    def setAngles(self, names, angles, speed):
        for n, a in zip(names, angles):
            self._angles[n] = a

    def getAngles(self, names, use_sensors):
        return [self._angles.get(n, 0.0) for n in names]

    def moveTo(self, x, y, theta):
        self._move_active = True
        time.sleep(0.1)
        self._move_active = False

    def moveToward(self, x, y, theta):
        self._move_active = (x != 0 or y != 0 or theta != 0)

    def moveIsActive(self):
        return self._move_active

    def stopMove(self):
        self._move_active = False

    def wakeUp(self):
        pass

    def rest(self):
        pass


class MockTTS(object):
    def say(self, text):
        pass


class MockPosture(object):
    def goToPosture(self, name, speed):
        time.sleep(0.05)

    def getPosture(self):
        return "Stand"


class MockProxies(object):
    def __init__(self):
        self.motion = MockMotion()
        self.tts = MockTTS()
        self.animated_tts = MockTTS()
        self.posture = MockPosture()
        self.leds = None


class MockClientConn(object):
    """Collects async responses sent by background threads."""

    def __init__(self):
        self.responses = []
        self._lock = threading.Lock()

    def send_response(self, resp):
        with self._lock:
            self.responses.append(resp)
        return True

    def get_responses(self):
        with self._lock:
            return list(self.responses)


# ============================================================
# Patch imports so server.py loads without real naoqi
# ============================================================
naoqi_mock = types.ModuleType("naoqi")
naoqi_mock.ALProxy = lambda *a, **k: None
sys.modules["naoqi"] = naoqi_mock

# Load motion_library into a mock module
motions_mod = types.ModuleType("motion_library")
_ml_path = os.path.join(_NAO_BODY_DIR, "motion_library.py")
with open(_ml_path, encoding="utf-8") as f:
    exec(compile(f.read(), _ml_path, "exec"), motions_mod.__dict__)
sys.modules["motion_library"] = motions_mod

# Load server classes
server_ns = {}
_srv_path = os.path.join(_NAO_BODY_DIR, "server.py")
with open(_srv_path, encoding="utf-8") as f:
    server_src = f.read()
# Patch out the actual imports (already in sys.modules)
server_src = server_src.replace("from naoqi import ALProxy", "# naoqi mocked")
server_src = server_src.replace(
    "import motion_library as motions",
    "import motion_library as motions  # from sys.modules",
)
exec(compile(server_src, _srv_path, "exec"), server_ns)

NaoStateMachine = server_ns["NaoStateMachine"]
CommandDispatcher = server_ns["CommandDispatcher"]


# ============================================================
# Tests
# ============================================================
def run_tests():
    px = MockProxies()
    state = NaoStateMachine(initial_posture="standing")
    dispatcher = CommandDispatcher(px, state)
    client = MockClientConn()
    passed = 0

    # --- Test 1: Blocking say without id ---
    resp = dispatcher.dispatch({"action": "say", "text": "hello"}, client)
    assert resp["status"] == "ok", "T1a: %s" % resp
    assert "state" in resp, "T1b: no state"
    assert resp["state"]["posture"] == "standing", "T1c"
    assert "id" not in resp, "T1d: unexpected id"
    assert "type" not in resp, "T1e: unexpected type"
    print("  [PASS] Test 1: blocking say without id -> ok")
    passed += 1

    # --- Test 2: Blocking say WITH id ---
    resp = dispatcher.dispatch({"action": "say", "text": "hi", "id": "r1"}, client)
    assert resp["status"] == "ok", "T2a"
    assert resp["id"] == "r1", "T2b"
    assert resp["type"] == "done", "T2c"
    print("  [PASS] Test 2: blocking say with id -> done")
    passed += 1

    # --- Test 3: Threaded walk without id (backward compatible) ---
    resp = dispatcher.dispatch({"action": "walk_toward", "x": 0.5}, client)
    assert resp["status"] == "ok", "T3a: %s" % resp
    assert "id" not in resp, "T3b"
    assert "type" not in resp, "T3c"
    time.sleep(0.2)
    print("  [PASS] Test 3: threaded walk without id -> ok (backward compat)")
    passed += 1

    # --- Test 4: Two-phase ACK (walk with id) ---
    client2 = MockClientConn()
    resp = dispatcher.dispatch(
        {"action": "walk_toward", "x": 0.3, "id": "r42"}, client2
    )
    assert resp["status"] == "accepted", "T4a: %s" % resp
    assert resp["id"] == "r42", "T4b"
    assert resp["type"] == "ack", "T4c"
    assert resp["state"]["legs"] == "walking", "T4d"
    time.sleep(0.3)
    done_resps = client2.get_responses()
    assert len(done_resps) == 1, "T4e: expected 1 done, got %d" % len(done_resps)
    done = done_resps[0]
    assert done["id"] == "r42", "T4f"
    assert done["type"] == "done", "T4g"
    assert done["status"] == "ok", "T4h"
    assert done["state"]["legs"] == "idle", "T4i"
    print("  [PASS] Test 4: two-phase ACK (ack + done)")
    passed += 1

    # --- Test 5: Rejection (walk while sitting) ---
    state.set_channel("posture", "sitting")
    resp = dispatcher.dispatch(
        {"action": "walk_toward", "x": 0.5, "id": "r99"}, client
    )
    assert resp["status"] == "rejected", "T5a: %s" % resp
    assert resp["reason"] == "must_stand_first", "T5b"
    assert resp["id"] == "r99", "T5c"
    assert resp["type"] == "ack", "T5d"
    print("  [PASS] Test 5: walk while sitting -> rejected")
    state.set_channel("posture", "standing")
    passed += 1

    # --- Test 6: query_state includes head angles ---
    resp = dispatcher.dispatch({"action": "query_state"}, client)
    assert resp["status"] == "ok", "T6a"
    assert "head_yaw" in resp["state"], "T6b"
    assert "head_pitch" in resp["state"], "T6c"
    print("  [PASS] Test 6: query_state includes head angles")
    passed += 1

    # --- Test 7: heartbeat ---
    resp = dispatcher.dispatch({"action": "heartbeat"}, client)
    assert resp["status"] == "ok", "T7a"
    assert "state" in resp, "T7b"
    print("  [PASS] Test 7: heartbeat returns state")
    passed += 1

    # --- Test 8: get_posture ---
    resp = dispatcher.dispatch({"action": "get_posture"}, client)
    assert resp["status"] == "ok", "T8a"
    print("  [PASS] Test 8: get_posture works")
    passed += 1

    # --- Test 9: move_head ---
    resp = dispatcher.dispatch(
        {"action": "move_head", "yaw": 0.5, "pitch": 0.0}, client
    )
    assert resp["status"] == "ok", "T9a"
    print("  [PASS] Test 9: move_head returns ok")
    passed += 1

    # --- Test 10: BUG 4 fix - say does NOT stop walk ---
    state.set_channel("legs", "walking")
    resp = dispatcher.dispatch({"action": "say", "text": "still walking"}, client)
    assert resp["status"] == "ok", "T10a"
    assert state.legs == "walking", "T10b: say killed walking!"
    print("  [PASS] Test 10: say does NOT affect legs (BUG 4 fixed)")
    state.set_channel("legs", "idle")
    passed += 1

    # --- Test 11: Pose auto-stops walking ---
    state.set_channel("legs", "walking")
    state.set_channel("posture", "standing")
    client3 = MockClientConn()
    resp = dispatcher.dispatch(
        {"action": "pose", "name": "sit", "id": "p1"}, client3
    )
    assert resp["status"] == "accepted", "T11a: %s" % resp
    assert state.legs == "idle", "T11b: pre_threaded should stop walk"
    time.sleep(0.2)
    assert state.posture == "sitting", "T11c: posture should be sitting"
    print("  [PASS] Test 11: pose auto-stops walking, sets posture")
    passed += 1

    # --- Test 12: Dance rejected while sitting ---
    resp = dispatcher.dispatch({"action": "animate", "name": "dance"}, client)
    assert resp["status"] == "rejected", "T12a: %s" % resp
    assert resp["reason"] == "must_stand_first", "T12b"
    print("  [PASS] Test 12: dance rejected while sitting")
    passed += 1

    # --- Test 13: Wake up changes posture ---
    state.set_channel("posture", "resting")
    client4 = MockClientConn()
    resp = dispatcher.dispatch({"action": "wake_up", "id": "w1"}, client4)
    assert resp["status"] == "accepted", "T13a"
    time.sleep(0.2)
    assert state.posture == "standing", "T13b"
    print("  [PASS] Test 13: wake_up -> posture standing")
    passed += 1

    # --- Test 14: stop_walk sets legs idle ---
    state.set_channel("legs", "walking")
    resp = dispatcher.dispatch({"action": "stop_walk"}, client)
    assert resp["status"] == "ok", "T14a"
    assert state.legs == "idle", "T14b"
    print("  [PASS] Test 14: stop_walk sets legs idle")
    passed += 1

    # --- Test 15: Unknown action ---
    resp = dispatcher.dispatch({"action": "fly"}, client)
    assert resp["status"] == "error", "T15a"
    print("  [PASS] Test 15: unknown action -> error")
    passed += 1

    # --- Test 16: Missing action ---
    resp = dispatcher.dispatch({}, client)
    assert resp["status"] == "error", "T16a"
    print("  [PASS] Test 16: missing action -> error")
    passed += 1

    # --- Test 17: State snapshot structure ---
    state.set_channel("posture", "standing")
    snap = state.snapshot()
    expected_keys = {"posture", "head", "legs", "speech", "arms"}
    assert set(snap.keys()) == expected_keys, "T17a: %s" % snap.keys()
    snap_with_angles = state.snapshot(px.motion)
    assert "head_yaw" in snap_with_angles, "T17b"
    assert "head_pitch" in snap_with_angles, "T17c"
    print("  [PASS] Test 17: snapshot structure correct")
    passed += 1

    # --- Test 18: Animated say without stopMove ---
    state.set_channel("legs", "walking")
    resp = dispatcher.dispatch(
        {"action": "animated_say", "text": "test"}, client
    )
    assert resp["status"] == "ok", "T18a"
    assert state.legs == "walking", "T18b: animated_say killed walking!"
    print("  [PASS] Test 18: animated_say does NOT affect legs")
    state.set_channel("legs", "idle")
    passed += 1

    # --- Test 19: Arms busy rejection ---
    state.set_channel("posture", "standing")
    state.set_channel("arms", "animating")
    resp = dispatcher.dispatch({"action": "animate", "name": "wave"}, client)
    assert resp["status"] == "rejected", "T19a: %s" % resp
    assert resp["reason"] == "arms_busy", "T19b"
    print("  [PASS] Test 19: wave rejected while arms busy")
    state.set_channel("arms", "idle")
    passed += 1

    # --- Test 20: Rest auto-stops walking ---
    state.set_channel("legs", "walking")
    state.set_channel("posture", "standing")
    client5 = MockClientConn()
    resp = dispatcher.dispatch({"action": "rest", "id": "rest1"}, client5)
    assert resp["status"] == "accepted", "T20a: %s" % resp
    assert state.legs == "idle", "T20b: rest pre_threaded should stop walk"
    time.sleep(0.2)
    assert state.posture == "resting", "T20c"
    print("  [PASS] Test 20: rest auto-stops walking, posture -> resting")
    passed += 1

    print()
    print("=" * 50)
    print("  ALL %d TESTS PASSED" % passed)
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
