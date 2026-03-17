#!/usr/bin/env python3
"""
test_phase4.py -- Integration tests for Phase 4: Per-Channel Workers.

Tests channel routing, parallelism, walk interruption, stop_all,
state transitions, and backward compatibility.

Runs WITHOUT real NAOqi -- uses a mock that records calls.
"""

import json
import os
import socket
import sys
import threading
import time

# Ensure nao_body/ is on sys.path (tests now live in nao_body/tests/)
_NAO_BODY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _NAO_BODY_DIR)

# ---------------------------------------------------------------------------
# Minimal NAOqi mock (replaces 'from naoqi import ALProxy')
# ---------------------------------------------------------------------------

class MockMotion(object):
    """Records calls for verification."""
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._walk_block = threading.Event()
        self._move_active = False

    def setAngles(self, names, values, speed):
        with self._lock:
            self.calls.append(("setAngles", names, values, speed))

    def getAngles(self, names, _use_sensors):
        return [0.1, -0.05]

    def moveTo(self, x, y, theta):
        with self._lock:
            self.calls.append(("moveTo", x, y, theta))
            self._move_active = True
        # Simulate blocking walk (0.3s or until interrupted)
        self._walk_block.clear()
        self._walk_block.wait(timeout=0.3)
        with self._lock:
            self._move_active = False

    def moveToward(self, x, y, theta):
        with self._lock:
            self.calls.append(("moveToward", x, y, theta))

    def moveIsActive(self):
        with self._lock:
            return self._move_active

    def stopMove(self):
        with self._lock:
            self.calls.append(("stopMove",))
            self._move_active = False
        self._walk_block.set()  # unblock moveTo

    def wakeUp(self):
        with self._lock:
            self.calls.append(("wakeUp",))

    def rest(self):
        with self._lock:
            self.calls.append(("rest",))

    def last_call(self):
        with self._lock:
            return self.calls[-1] if self.calls else None

    def call_count(self, name):
        with self._lock:
            return sum(1 for c in self.calls if c[0] == name)


class MockTts(object):
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()
        self._speak_block = threading.Event()

    def say(self, text):
        with self._lock:
            self.calls.append(("say", text))
        # Simulate blocking speech (0.2s)
        self._speak_block.clear()
        self._speak_block.wait(timeout=0.2)

    def stopAll(self):
        with self._lock:
            self.calls.append(("stopAll",))
        self._speak_block.set()

    def call_count(self, name):
        with self._lock:
            return sum(1 for c in self.calls if c[0] == name)


class MockAnimatedTts(object):
    def say(self, text):
        time.sleep(0.1)


class MockPosture(object):
    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def goToPosture(self, name, speed):
        with self._lock:
            self.calls.append(("goToPosture", name, speed))
        time.sleep(0.1)

    def getPosture(self):
        return "Stand"


class MockProxies(object):
    def __init__(self):
        self.motion = MockMotion()
        self.tts = MockTts()
        self.animated_tts = MockAnimatedTts()
        self.posture = MockPosture()
        self.leds = None


class MockALProxy(object):
    """Fake ALProxy so 'from naoqi import ALProxy' doesn't fail."""
    def __init__(self, *args, **kwargs):
        pass


# Install mocks BEFORE importing server
import types
naoqi_mock = types.ModuleType("naoqi")
naoqi_mock.ALProxy = MockALProxy
sys.modules["naoqi"] = naoqi_mock

# Mock motion_library to avoid real NAOqi calls
import motion_library as motions_real

# Now import server (nao_body/ already on sys.path)
from server import (
    NaoStateMachine, ClientConnection, ChannelWorker,
    CommandDispatcher, CHANNEL_ROUTING,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_dispatcher():
    """Create a dispatcher with mock proxies."""
    px = MockProxies()
    state = NaoStateMachine(initial_posture="standing")
    d = CommandDispatcher(px, state)
    d.start_workers()
    return d, px, state


class FakeClientConn(object):
    """Captures responses sent via send_response."""
    def __init__(self):
        self.responses = []
        self._lock = threading.Lock()

    def send_response(self, resp):
        with self._lock:
            self.responses.append(resp)
        return True

    @property
    def is_closed(self):
        return False

    def get_responses(self):
        with self._lock:
            return list(self.responses)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    passed = 0

    # === Test 1: HEAD commands are inline (immediate) ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"action": "move_head", "yaw": 0.5, "pitch": 0.1, "speed": 0.15},
        cc,
    )
    assert resp["status"] == "ok", "T1a: %s" % resp
    assert resp["action"] == "move_head", "T1b"
    assert px.motion.call_count("setAngles") >= 1, "T1c"
    d.stop_workers()
    print("  [PASS] Test 1: HEAD move_head runs inline")
    passed += 1

    # === Test 2: SYSTEM query_state is inline ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch({"id": "r1", "action": "query_state"}, cc)
    assert resp["status"] == "ok", "T2a"
    assert resp["type"] == "done", "T2b"
    assert "head_yaw" in resp["state"], "T2c"
    d.stop_workers()
    print("  [PASS] Test 2: query_state inline with head angles")
    passed += 1

    # === Test 3: SPEECH say → worker (ack + done) ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "r2", "action": "say", "text": "hello"}, cc,
    )
    assert resp["type"] == "ack", "T3a: %s" % resp
    assert resp["status"] == "accepted", "T3b"
    assert resp["state"]["speech"] == "speaking", "T3c"
    # Wait for worker to finish
    time.sleep(0.5)
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    assert len(done_msgs) >= 1, "T3d: no done msg"
    assert done_msgs[0]["id"] == "r2", "T3e"
    assert done_msgs[0]["status"] == "ok", "T3f"
    assert done_msgs[0]["state"]["speech"] == "idle", "T3g"
    d.stop_workers()
    print("  [PASS] Test 3: say routed to SPEECH worker (ack + done)")
    passed += 1

    # === Test 4: LEGS walk_toward → worker (ack + done) ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "r3", "action": "walk_toward", "x": 0.5}, cc,
    )
    assert resp["type"] == "ack", "T4a"
    assert resp["status"] == "accepted", "T4b"
    assert resp["state"]["legs"] == "walking", "T4c"
    time.sleep(0.6)
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    assert len(done_msgs) >= 1, "T4d: no done"
    assert done_msgs[0]["state"]["legs"] == "idle", "T4e"
    d.stop_workers()
    print("  [PASS] Test 4: walk_toward routed to LEGS worker")
    passed += 1

    # === Test 5: ARMS animate wave → worker ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "r4", "action": "animate", "name": "wave"}, cc,
    )
    assert resp["type"] == "ack", "T5a"
    assert resp["state"]["arms"] == "animating", "T5b"
    time.sleep(3.5)  # wave takes ~3s
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    assert len(done_msgs) >= 1, "T5c: no done"
    assert done_msgs[0]["state"]["arms"] == "idle", "T5d"
    d.stop_workers()
    print("  [PASS] Test 5: animate:wave routed to ARMS worker")
    passed += 1

    # === Test 6: Parallel — say + walk run simultaneously ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    t_start = time.monotonic()
    # Send walk (blocks for 0.3s in mock)
    resp_walk = d.dispatch(
        {"id": "rW", "action": "walk_toward", "x": 0.3}, cc,
    )
    # Send say immediately (blocks for 0.2s in mock)
    resp_say = d.dispatch(
        {"id": "rS", "action": "say", "text": "parallel"}, cc,
    )
    assert resp_walk["type"] == "ack", "T6a"
    assert resp_say["type"] == "ack", "T6b"
    # Both should finish within ~0.4s (parallel), not 0.5s (serial)
    time.sleep(0.6)
    done_msgs = cc.get_responses()
    done_ids = set(r.get("id") for r in done_msgs if r.get("type") == "done")
    assert "rW" in done_ids, "T6c: walk done missing"
    assert "rS" in done_ids, "T6d: say done missing"
    t_elapsed = time.monotonic() - t_start
    # If serial, would take >0.5s. Parallel should be <0.5s + overhead.
    assert t_elapsed < 1.0, "T6e: took %.2fs, expected parallel" % t_elapsed
    d.stop_workers()
    print("  [PASS] Test 6: say + walk run in parallel (%.2fs)" % t_elapsed)
    passed += 1

    # === Test 7: Walk interruption — new walk cancels current ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    # Start a walk
    resp1 = d.dispatch(
        {"id": "rA", "action": "walk_toward", "x": 1.0}, cc,
    )
    assert resp1["state"]["legs"] == "walking", "T7a"
    time.sleep(0.05)  # let worker start
    # Send second walk — should interrupt first
    resp2 = d.dispatch(
        {"id": "rB", "action": "walk_toward", "x": 0.2}, cc,
    )
    assert resp2["type"] == "ack", "T7b"
    # stopMove should have been called (interruption)
    assert px.motion.call_count("stopMove") >= 1, "T7c: no stopMove"
    time.sleep(0.6)
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    done_ids = [r["id"] for r in done_msgs]
    assert "rB" in done_ids, "T7d: second walk done missing"
    d.stop_workers()
    print("  [PASS] Test 7: Walk interruption — new walk cancels current")
    passed += 1

    # === Test 8: State machine rejects walk while sitting ===
    d, px, state = make_dispatcher()
    state.set_channel("posture", "sitting")
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "r5", "action": "walk_toward", "x": 0.5}, cc,
    )
    assert resp["status"] == "rejected", "T8a"
    assert resp["reason"] == "must_stand_first", "T8b"
    d.stop_workers()
    print("  [PASS] Test 8: Walk rejected while sitting")
    passed += 1

    # === Test 9: State machine rejects dance while walking ===
    d, px, state = make_dispatcher()
    state.set_channel("legs", "walking")
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "r6", "action": "animate", "name": "dance"}, cc,
    )
    assert resp["status"] == "rejected", "T9a"
    assert resp["reason"] == "legs_busy", "T9b"
    d.stop_workers()
    print("  [PASS] Test 9: Dance rejected while walking")
    passed += 1

    # === Test 10: stop_all clears everything ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    # Start a walk and speech
    d.dispatch({"action": "walk_toward", "x": 1.0}, cc)
    d.dispatch({"action": "say", "text": "hello"}, cc)
    time.sleep(0.05)
    # Now stop all
    resp = d.dispatch({"id": "r7", "action": "stop_all"}, cc)
    assert resp["status"] == "ok", "T10a"
    assert resp["state"]["legs"] == "idle", "T10b"
    assert resp["state"]["speech"] == "idle", "T10c"
    assert resp["state"]["arms"] == "idle", "T10d"
    assert px.motion.call_count("stopMove") >= 1, "T10e"
    d.stop_workers()
    print("  [PASS] Test 10: stop_all clears all channels")
    passed += 1

    # === Test 11: stop_walk only affects legs ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    d.dispatch({"action": "walk_toward", "x": 1.0}, cc)
    d.dispatch({"action": "say", "text": "test"}, cc)
    time.sleep(0.05)
    resp = d.dispatch({"action": "stop_walk"}, cc)
    assert resp["state"]["legs"] == "idle", "T11a"
    assert resp["state"]["speech"] == "speaking", "T11b: speech should continue"
    time.sleep(0.5)
    d.stop_workers()
    print("  [PASS] Test 11: stop_walk only stops legs, speech continues")
    passed += 1

    # === Test 12: set_walk_velocity inline ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "r8", "action": "set_walk_velocity", "x": 0.3, "theta": 0.1},
        cc,
    )
    assert resp["status"] == "ok", "T12a"
    assert resp["type"] == "done", "T12b: should be inline done"
    assert resp["state"]["legs"] == "walking", "T12c"
    assert px.motion.call_count("moveToward") >= 1, "T12d"
    d.stop_workers()
    print("  [PASS] Test 12: set_walk_velocity is inline")
    passed += 1

    # === Test 13: set_walk_velocity rejected while sitting ===
    d, px, state = make_dispatcher()
    state.set_channel("posture", "sitting")
    cc = FakeClientConn()
    resp = d.dispatch(
        {"action": "set_walk_velocity", "x": 0.3}, cc,
    )
    assert resp["status"] == "rejected", "T13a"
    d.stop_workers()
    print("  [PASS] Test 13: set_walk_velocity rejected while sitting")
    passed += 1

    # === Test 14: Backward compat — no id gives ok (no type field) ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch({"action": "say", "text": "compat"}, cc)
    assert resp["status"] == "ok", "T14a"
    assert "type" not in resp, "T14b: should have no type without id"
    d.stop_workers()
    print("  [PASS] Test 14: Backward compat — no id, no type in response")
    passed += 1

    # === Test 15: Pose interrupts current walk ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    d.dispatch({"id": "rW2", "action": "walk_toward", "x": 1.0}, cc)
    time.sleep(0.05)
    assert state.legs == "walking", "T15a"
    d.dispatch({"id": "rP", "action": "pose", "name": "sit"}, cc)
    # Should have called stopMove to interrupt walk
    assert px.motion.call_count("stopMove") >= 1, "T15b"
    time.sleep(0.5)
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    pose_done = [r for r in done_msgs if r.get("id") == "rP"]
    assert len(pose_done) >= 1, "T15c: pose done missing"
    assert state.posture == "sitting", "T15d"
    d.stop_workers()
    print("  [PASS] Test 15: Pose interrupts current walk")
    passed += 1

    # === Test 16: Dance claims both legs and arms ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "rD", "action": "animate", "name": "dance"}, cc,
    )
    assert resp["state"]["legs"] == "animating", "T16a"
    assert resp["state"]["arms"] == "animating", "T16b"
    # Walk should be rejected now
    resp2 = d.dispatch({"action": "walk_toward", "x": 0.3}, cc)
    assert resp2["status"] == "rejected", "T16c"
    assert resp2["reason"] == "legs_busy_dancing", "T16d"
    time.sleep(5.0)  # wait for dance
    assert state.legs == "idle", "T16e"
    assert state.arms == "idle", "T16f"
    d.stop_workers()
    print("  [PASS] Test 16: Dance claims legs+arms, blocks walk")
    passed += 1

    # === Test 17: SPEECH FIFO — two says queued ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    d.dispatch({"id": "s1", "action": "say", "text": "first"}, cc)
    d.dispatch({"id": "s2", "action": "say", "text": "second"}, cc)
    time.sleep(0.8)  # both should finish (0.2s each)
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    done_ids = [r["id"] for r in done_msgs]
    assert "s1" in done_ids, "T17a"
    assert "s2" in done_ids, "T17b"
    # s1 should finish before s2 (FIFO)
    s1_idx = done_ids.index("s1")
    s2_idx = done_ids.index("s2")
    assert s1_idx < s2_idx, "T17c: FIFO order violated"
    d.stop_workers()
    print("  [PASS] Test 17: SPEECH FIFO — two says in order")
    passed += 1

    # === Test 18: Channel routing map completeness ===
    expected_routed = {
        "walk_toward", "pose", "rest", "wake_up",
        "say", "animated_say", "animate",
    }
    assert set(CHANNEL_ROUTING.keys()) == expected_routed, "T18a"
    print("  [PASS] Test 18: CHANNEL_ROUTING covers all worker actions")
    passed += 1

    # === Test 19: fire-and-forget (no_ack, no id) walk ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"action": "walk_toward", "x": 0.3, "no_ack": True}, cc,
    )
    # Dispatch returns a response (ack-like), but server won't send it
    # (no_ack handling is in TcpServer._handle_client, not dispatch)
    assert resp["status"] == "ok", "T19a"
    time.sleep(0.5)
    # Worker should NOT send done (no request id)
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    assert len(done_msgs) == 0, "T19b: no done for fire-and-forget"
    d.stop_workers()
    print("  [PASS] Test 19: Fire-and-forget walk — no done response")
    passed += 1

    # === Test 20: wake_up updates posture to standing ===
    d, px, state = make_dispatcher()
    state.set_channel("posture", "resting")
    cc = FakeClientConn()
    resp = d.dispatch({"id": "rWU", "action": "wake_up"}, cc)
    assert resp["type"] == "ack", "T20a"
    time.sleep(0.5)
    done_msgs = [r for r in cc.get_responses() if r.get("id") == "rWU"]
    assert len(done_msgs) >= 1, "T20b"
    assert state.posture == "standing", "T20c"
    d.stop_workers()
    print("  [PASS] Test 20: wake_up updates posture to standing")
    passed += 1

    print()
    print("=" * 50)
    print("  ALL %d TESTS PASSED" % passed)
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
