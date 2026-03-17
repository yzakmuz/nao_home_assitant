#!/usr/bin/env python3
"""
test_phase5.py -- Integration tests for Phase 5: Connection Resilience.

Tests watchdog (safe-sit on timeout), fall detection, disconnect cleanup,
heartbeat, auto-reconnect, and event handling.

Runs WITHOUT real NAOqi -- uses mocks that record calls.
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
# Minimal NAOqi mock (same as Phase 4 + ALMemory)
# ---------------------------------------------------------------------------

class MockMotion(object):
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
        self._walk_block.set()

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
        self._speak_block.clear()
        self._speak_block.wait(timeout=0.2)

    def stopAll(self):
        with self._lock:
            self.calls.append(("stopAll",))
        self._speak_block.set()

    def call_count(self, name):
        with self._lock:
            return sum(1 for c in self.calls if c[0] == name)

    def last_say_text(self):
        with self._lock:
            says = [c for c in self.calls if c[0] == "say"]
            return says[-1][1] if says else None


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

    def call_count(self, name):
        with self._lock:
            return sum(1 for c in self.calls if c[0] == name)


class MockMemory(object):
    """Mock ALMemory with controllable fall state."""
    def __init__(self):
        self._fallen = False
        self._lock = threading.Lock()

    def getData(self, key):
        if key == "robotHasFallen":
            with self._lock:
                return self._fallen
        return None

    def set_fallen(self, value):
        with self._lock:
            self._fallen = value


class MockProxies(object):
    def __init__(self):
        self.motion = MockMotion()
        self.tts = MockTts()
        self.animated_tts = MockAnimatedTts()
        self.posture = MockPosture()
        self.leds = None
        self._memory = MockMemory()
        self.ip = "127.0.0.1"

    @property
    def memory(self):
        return self._memory


class MockALProxy(object):
    def __init__(self, *args, **kwargs):
        pass


# Install mocks BEFORE importing server
import types
naoqi_mock = types.ModuleType("naoqi")
naoqi_mock.ALProxy = MockALProxy
sys.modules["naoqi"] = naoqi_mock

import motion_library as motions_real

from server import (
    NaoStateMachine, ClientConnection, ChannelWorker,
    CommandDispatcher, Watchdog, FallDetector, TcpServer,
    CHANNEL_ROUTING, WATCHDOG_TIMEOUT_S,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_dispatcher():
    px = MockProxies()
    state = NaoStateMachine(initial_posture="standing")
    d = CommandDispatcher(px, state)
    d.start_workers()
    return d, px, state


class FakeClientConn(object):
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

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    passed = 0

    # === Test 1: Watchdog triggers after timeout ===
    d, px, state = make_dispatcher()
    wd = Watchdog(1.0, px, d)  # 1s timeout for fast test
    wd.set_client_connected(True)
    wd.start()
    time.sleep(2.5)  # Wait for watchdog to trigger
    wd.stop()
    assert px.motion.call_count("stopMove") >= 1, "T1a: stopMove not called"
    assert px.posture.call_count("goToPosture") >= 1, "T1b: no safe sit"
    assert state.posture == "sitting", "T1c: posture should be sitting, got %s" % state.posture
    d.stop_workers()
    print("  [PASS] Test 1: Watchdog triggers safe-sit after timeout")
    passed += 1

    # === Test 2: Watchdog resets on message received ===
    d, px, state = make_dispatcher()
    wd = Watchdog(3.0, px, d)  # 3s timeout (generous margin)
    wd.set_client_connected(True)
    wd.start()
    # Send messages every 0.5s for 3s — should NOT trigger (gap < 3s)
    for _ in range(6):
        wd.message_received()
        time.sleep(0.5)
    wd.stop()
    # Watchdog should NOT have triggered (messages kept it alive)
    assert state.posture == "standing", "T2a: watchdog should not have triggered"
    assert px.posture.call_count("goToPosture") == 0, "T2b: no goToPosture expected"
    d.stop_workers()
    print("  [PASS] Test 2: Watchdog resets on message received")
    passed += 1

    # === Test 3: Watchdog does not trigger when no client connected ===
    d, px, state = make_dispatcher()
    wd = Watchdog(0.5, px, d)  # Very short timeout
    # Do NOT call set_client_connected(True)
    wd.start()
    time.sleep(1.5)
    wd.stop()
    assert state.posture == "standing", "T3a: should not trigger without client"
    d.stop_workers()
    print("  [PASS] Test 3: Watchdog inactive when no client connected")
    passed += 1

    # === Test 4: Fall detector sends event to client ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    fd = FallDetector(px, state)
    fd.set_client(cc)
    fd._running = True
    # Start the fall detector loop manually
    fd_thread = threading.Thread(target=fd._loop, daemon=True)
    fd_thread.start()
    # Trigger fall
    px._memory.set_fallen(True)
    time.sleep(1.0)  # Wait for detection
    fd._running = False
    fd_thread.join(timeout=2.0)
    # Check results
    assert state.posture == "fallen", "T4a: posture should be fallen, got %s" % state.posture
    assert px.motion.call_count("stopMove") >= 1, "T4b: stopMove not called"
    events = [r for r in cc.get_responses() if r.get("type") == "event"]
    assert len(events) >= 1, "T4c: no fall event sent"
    assert events[0]["event"] == "fallen", "T4d: event should be 'fallen'"
    d.stop_workers()
    print("  [PASS] Test 4: Fall detector sends event to client")
    passed += 1

    # === Test 5: Fall detector recovers (fall handled once) ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    fd = FallDetector(px, state)
    fd.set_client(cc)
    fd._running = True
    fd_thread = threading.Thread(target=fd._loop, daemon=True)
    fd_thread.start()
    # Fall then recover
    px._memory.set_fallen(True)
    time.sleep(1.0)
    events_before = len([r for r in cc.get_responses() if r.get("type") == "event"])
    # Clear fall
    px._memory.set_fallen(False)
    time.sleep(1.0)
    # Fall again
    state.set_channel("posture", "standing")  # Reset for second fall
    px._memory.set_fallen(True)
    time.sleep(1.0)
    fd._running = False
    fd_thread.join(timeout=2.0)
    events_total = len([r for r in cc.get_responses() if r.get("type") == "event"])
    assert events_total >= 2, "T5a: should have 2 fall events, got %d" % events_total
    d.stop_workers()
    print("  [PASS] Test 5: Fall detector recovers and re-detects")
    passed += 1

    # === Test 6: Fall detector handles no client gracefully ===
    d, px, state = make_dispatcher()
    fd = FallDetector(px, state)
    # No client set (None)
    fd._running = True
    fd_thread = threading.Thread(target=fd._loop, daemon=True)
    fd_thread.start()
    px._memory.set_fallen(True)
    time.sleep(1.0)
    fd._running = False
    fd_thread.join(timeout=2.0)
    # Should set posture to fallen even without client
    assert state.posture == "fallen", "T6a: posture should be fallen"
    # Should not crash
    d.stop_workers()
    print("  [PASS] Test 6: Fall detector handles no client gracefully")
    passed += 1

    # === Test 7: Disconnect cleanup stops motions and clears state ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    # Start a walk and speech
    d.dispatch({"action": "walk_toward", "x": 1.0}, cc)
    d.dispatch({"action": "say", "text": "test"}, cc)
    time.sleep(0.05)
    assert state.legs == "walking", "T7a: should be walking"
    assert state.speech == "speaking", "T7b: should be speaking"

    # Simulate disconnect cleanup
    wd = Watchdog(10.0, px, d)
    fd = FallDetector(px, state)
    server = TcpServer.__new__(TcpServer)
    server.port = 0
    server.dispatcher = d
    server._proxies = px
    server._running = True
    server._watchdog = wd
    server._fall_detector = fd
    server._on_client_disconnect(cc)

    # Verify cleanup
    assert state.legs == "idle", "T7c: legs should be idle after cleanup"
    assert state.speech == "idle", "T7d: speech should be idle after cleanup"
    assert state.arms == "idle", "T7e: arms should be idle after cleanup"
    assert px.motion.call_count("stopMove") >= 1, "T7f: stopMove not called"
    assert px.tts.call_count("stopAll") >= 1, "T7g: tts.stopAll not called"
    d.stop_workers()
    print("  [PASS] Test 7: Disconnect cleanup stops motions and resets state")
    passed += 1

    # === Test 8: Watchdog safe-sit speaks announcement ===
    d, px, state = make_dispatcher()
    wd = Watchdog(0.5, px, d)
    wd.set_client_connected(True)
    wd.start()
    time.sleep(2.0)  # Wait for trigger
    wd.stop()
    assert px.tts.last_say_text() is not None, "T8a: should have said something"
    assert "connection" in px.tts.last_say_text().lower() or "safety" in px.tts.last_say_text().lower(), \
        "T8b: announcement should mention connection/safety"
    d.stop_workers()
    print("  [PASS] Test 8: Watchdog announces safe-sit via speech")
    passed += 1

    # === Test 9: Watchdog clears worker queues ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    # Queue up multiple commands
    for i in range(5):
        d.dispatch({"action": "say", "text": "queued %d" % i}, cc)
    time.sleep(0.05)

    wd = Watchdog(0.5, px, d)
    wd.set_client_connected(True)
    wd.start()
    time.sleep(2.0)
    wd.stop()

    # State should be reset
    assert state.speech == "idle", "T9a: speech should be idle"
    assert state.legs == "idle", "T9b: legs should be idle"
    d.stop_workers()
    print("  [PASS] Test 9: Watchdog clears all worker queues")
    passed += 1

    # === Test 10: Disconnect preserves posture ===
    d, px, state = make_dispatcher()
    state.set_channel("posture", "sitting")
    cc = FakeClientConn()

    server = TcpServer.__new__(TcpServer)
    server.port = 0
    server.dispatcher = d
    server._proxies = px
    server._running = True
    server._watchdog = Watchdog(10.0, px, d)
    server._fall_detector = FallDetector(px, state)
    server._on_client_disconnect(cc)

    # Posture should NOT be reset (it stays as-is)
    assert state.posture == "sitting", "T10a: posture should remain sitting"
    d.stop_workers()
    print("  [PASS] Test 10: Disconnect cleanup preserves posture state")
    passed += 1

    # === Test 11: Watchdog re-arms after client reconnects ===
    d, px, state = make_dispatcher()
    wd = Watchdog(1.0, px, d)
    wd.start()
    # First client connects and disconnects
    wd.set_client_connected(True)
    time.sleep(0.3)
    wd.set_client_connected(False)
    # No trigger (client disconnected)
    time.sleep(1.5)
    stopmove_before = px.motion.call_count("stopMove")

    # Second client connects — watchdog should re-arm
    wd.set_client_connected(True)
    wd.message_received()
    time.sleep(0.3)
    wd.message_received()  # Keep alive
    time.sleep(0.3)
    # Still alive
    stopmove_after = px.motion.call_count("stopMove")
    assert stopmove_after == stopmove_before, "T11a: should not trigger during keepalive"

    # Now let it timeout
    time.sleep(2.0)
    assert px.motion.call_count("stopMove") > stopmove_before, "T11b: should trigger after timeout"
    wd.stop()
    d.stop_workers()
    print("  [PASS] Test 11: Watchdog re-arms after reconnect")
    passed += 1

    # === Test 12: Phase 4 backward compat — HEAD inline still works ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"action": "move_head", "yaw": 0.5, "pitch": 0.1}, cc
    )
    assert resp["status"] == "ok", "T12a"
    assert px.motion.call_count("setAngles") >= 1, "T12b"
    d.stop_workers()
    print("  [PASS] Test 12: Phase 4 backward compat — HEAD inline works")
    passed += 1

    # === Test 13: Phase 4 backward compat — SPEECH worker still works ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    resp = d.dispatch(
        {"id": "r1", "action": "say", "text": "hello"}, cc
    )
    assert resp["type"] == "ack", "T13a"
    time.sleep(0.5)
    done_msgs = [r for r in cc.get_responses() if r.get("type") == "done"]
    assert len(done_msgs) >= 1, "T13b"
    d.stop_workers()
    print("  [PASS] Test 13: Phase 4 backward compat — SPEECH worker works")
    passed += 1

    # === Test 14: Phase 4 backward compat — walk interruption still works ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    d.dispatch({"id": "w1", "action": "walk_toward", "x": 1.0}, cc)
    time.sleep(0.05)
    d.dispatch({"id": "w2", "action": "walk_toward", "x": 0.2}, cc)
    assert px.motion.call_count("stopMove") >= 1, "T14a"
    time.sleep(0.6)
    d.stop_workers()
    print("  [PASS] Test 14: Phase 4 backward compat — walk interruption works")
    passed += 1

    # === Test 15: Phase 4 backward compat — stop_all still works ===
    d, px, state = make_dispatcher()
    cc = FakeClientConn()
    d.dispatch({"action": "walk_toward", "x": 1.0}, cc)
    d.dispatch({"action": "say", "text": "test"}, cc)
    time.sleep(0.05)
    resp = d.dispatch({"id": "sa", "action": "stop_all"}, cc)
    assert resp["status"] == "ok", "T15a"
    assert resp["state"]["legs"] == "idle", "T15b"
    assert resp["state"]["speech"] == "idle", "T15c"
    d.stop_workers()
    print("  [PASS] Test 15: Phase 4 backward compat — stop_all works")
    passed += 1

    # === Test 16: TcpServer accepts proxies parameter ===
    d, px, state = make_dispatcher()
    # Just verify the constructor accepts 3 args
    try:
        srv = TcpServer(12345, d, px)
        assert srv._watchdog is not None, "T16a: watchdog should exist"
        assert srv._fall_detector is not None, "T16b: fall_detector should exist"
    except Exception as exc:
        assert False, "T16c: TcpServer construction failed: %s" % exc
    d.stop_workers()
    print("  [PASS] Test 16: TcpServer accepts proxies and creates watchdog/fall_detector")
    passed += 1

    # === Test 17: Watchdog does not re-trigger after first trigger ===
    d, px, state = make_dispatcher()
    wd = Watchdog(0.5, px, d)
    wd.set_client_connected(True)
    wd.start()
    time.sleep(2.0)  # First trigger
    sit_count_1 = px.posture.call_count("goToPosture")
    time.sleep(2.0)  # Should NOT trigger again (already triggered)
    sit_count_2 = px.posture.call_count("goToPosture")
    wd.stop()
    assert sit_count_2 == sit_count_1, \
        "T17a: watchdog should not re-trigger (got %d vs %d)" % (sit_count_2, sit_count_1)
    d.stop_workers()
    print("  [PASS] Test 17: Watchdog triggers only once until reset")
    passed += 1

    # === Test 18: Full integration — heartbeat via mock TCP ===
    # This test creates a real TCP server+client pair to verify heartbeat
    d, px, state = make_dispatcher()
    d.start_workers()

    # Create a simple echo server
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind(("127.0.0.1", 0))
    port = srv_sock.getsockname()[1]
    srv_sock.listen(1)
    srv_sock.settimeout(5.0)

    received_messages = []
    recv_lock = threading.Lock()

    def mock_server():
        try:
            conn, _ = srv_sock.accept()
            conn.settimeout(0.5)
            buf = b""
            while len(received_messages) < 5:
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode("utf-8"))
                        with recv_lock:
                            received_messages.append(msg)
                        # Send a response for non-heartbeat commands
                        if msg.get("action") != "heartbeat" and not msg.get("no_ack"):
                            resp = json.dumps({
                                "id": msg.get("id"),
                                "type": "done",
                                "status": "ok",
                                "state": {"posture": "standing"},
                            }) + "\n"
                            conn.sendall(resp.encode("utf-8"))
                    except Exception:
                        pass
            conn.close()
        except Exception:
            pass

    srv_thread = threading.Thread(target=mock_server, daemon=True)
    srv_thread.start()

    # Import tcp_client here to avoid import issues with settings
    _rpi_brain_dir = os.path.join(_NAO_BODY_DIR, "..", "rpi_brain")
    sys.path.insert(0, _rpi_brain_dir)
    try:
        from comms.tcp_client import NaoTcpClient
        client = NaoTcpClient(host="127.0.0.1", port=port, timeout=2.0)
        connected = client.connect()
        if connected:
            # Wait for heartbeats to accumulate
            time.sleep(5.0)
            with recv_lock:
                hb_count = sum(
                    1 for m in received_messages
                    if m.get("action") == "heartbeat"
                )
            assert hb_count >= 1, "T18a: expected heartbeats, got %d" % hb_count
            client.disconnect()
            print("  [PASS] Test 18: Heartbeat sends to server (%d heartbeats)" % hb_count)
            passed += 1
        else:
            print("  [SKIP] Test 18: Could not connect to mock server")
    except ImportError:
        print("  [SKIP] Test 18: Could not import tcp_client (settings issue)")
    finally:
        srv_sock.close()

    d.stop_workers()

    print()
    print("=" * 50)
    print("  ALL %d TESTS PASSED" % passed)
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
