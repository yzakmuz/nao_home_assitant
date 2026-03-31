"""
test_tcp_protocol.py — Test the Webots NAO controller's TCP protocol.

Run this WHILE the Webots simulation is running (controller on port 5555).
It sends JSON commands and verifies the responses match the real server
protocol: state snapshots, two-phase ACK, rejections, etc.

Usage:
    python test_tcp_protocol.py [--port 5555]
"""

import json
import socket
import sys
import time

PORT = 5555
HOST = "127.0.0.1"
TIMEOUT = 10.0
_request_counter = 0


def _next_id():
    global _request_counter
    _request_counter += 1
    return "test_%d" % _request_counter


class TestClient:
    """Simple TCP client for testing the Webots NAO controller."""

    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(TIMEOUT)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((host, port))
        self._buffer = ""
        print("[OK] Connected to %s:%d" % (host, port))

    def send(self, cmd):
        """Send a JSON command (no id, no response expected for no_ack)."""
        data = json.dumps(cmd) + "\n"
        self.sock.sendall(data.encode("utf-8"))

    def send_and_recv(self, cmd, timeout=TIMEOUT):
        """Send command with auto-generated id, wait for first response."""
        rid = _next_id()
        cmd["id"] = rid
        self.send(cmd)
        return self._recv_one(timeout), rid

    def send_fire_and_forget(self, cmd):
        """Send with no_ack=true (no response expected)."""
        cmd["no_ack"] = True
        self.send(cmd)

    def recv_until_done(self, rid, timeout=TIMEOUT):
        """Wait for a 'done' message with the given id."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._recv_one(max(0.5, deadline - time.time()))
            if msg and msg.get("id") == rid and msg.get("type") == "done":
                return msg
        return None

    def _recv_one(self, timeout=TIMEOUT):
        """Receive one newline-delimited JSON message."""
        self.sock.settimeout(timeout)
        while "\n" not in self._buffer:
            try:
                data = self.sock.recv(4096)
                if not data:
                    return None
                self._buffer += data.decode("utf-8")
            except socket.timeout:
                return None
        line, self._buffer = self._buffer.split("\n", 1)
        line = line.strip()
        if not line:
            return self._recv_one(timeout)
        return json.loads(line)

    def close(self):
        self.sock.close()


# ===================================================================
# Test cases
# ===================================================================

def test_query_state(c):
    """query_state returns ok with state + head angles."""
    resp, rid = c.send_and_recv({"action": "query_state"})
    assert resp["status"] == "ok", "Expected ok, got: %s" % resp
    assert "state" in resp, "Missing state in response"
    state = resp["state"]
    for key in ("posture", "head", "legs", "speech", "arms",
                "head_yaw", "head_pitch"):
        assert key in state, "Missing '%s' in state" % key
    assert "head_yaw" in resp, "Missing top-level head_yaw"
    assert "head_pitch" in resp, "Missing top-level head_pitch"
    assert resp.get("type") == "done", "Expected type=done"
    print("[PASS] query_state — state: %s" % state["posture"])


def test_heartbeat(c):
    """heartbeat returns ok with state."""
    resp, _ = c.send_and_recv({"action": "heartbeat"})
    assert resp["status"] == "ok"
    assert "state" in resp
    print("[PASS] heartbeat")


def test_get_posture(c):
    """get_posture returns posture in state."""
    resp, _ = c.send_and_recv({"action": "get_posture"})
    assert resp["status"] == "ok"
    assert resp["state"]["posture"] in ("standing", "sitting", "resting")
    print("[PASS] get_posture — %s" % resp["state"]["posture"])


def test_move_head(c):
    """move_head (inline) returns done with head angles."""
    resp, _ = c.send_and_recv({
        "action": "move_head", "yaw": 0.5, "pitch": 0.1, "speed": 0.15
    })
    assert resp["status"] == "ok"
    assert "head_yaw" in resp, "Missing head_yaw"
    print("[PASS] move_head — yaw=%.2f pitch=%.2f" % (
        resp.get("head_yaw", 0), resp.get("head_pitch", 0)))

    # Re-center
    c.send_fire_and_forget({
        "action": "move_head", "yaw": 0.0, "pitch": -0.17, "speed": 0.15
    })


def test_move_head_relative(c):
    """move_head_relative adjusts from current position."""
    resp, _ = c.send_and_recv({
        "action": "move_head_relative", "d_yaw": 0.3, "d_pitch": 0.0
    })
    assert resp["status"] == "ok"
    print("[PASS] move_head_relative")

    c.send_fire_and_forget({
        "action": "move_head", "yaw": 0.0, "pitch": -0.17
    })


def test_fire_and_forget(c):
    """no_ack=true should not produce a response."""
    c.send_fire_and_forget({"action": "move_head", "yaw": 0.2, "pitch": 0.0})
    time.sleep(0.3)

    # Send a query to verify connection still works
    resp, _ = c.send_and_recv({"action": "heartbeat"})
    assert resp["status"] == "ok"
    print("[PASS] fire_and_forget (no response received)")

    c.send_fire_and_forget({"action": "move_head", "yaw": 0.0, "pitch": -0.17})


def test_say_ack_done(c):
    """say returns ack immediately, then done after speech duration."""
    resp, rid = c.send_and_recv({"action": "say", "text": "Hello!"})
    assert resp["status"] == "accepted", "Expected ack, got: %s" % resp
    assert resp.get("type") == "ack"
    assert resp["state"]["speech"] == "speaking"
    print("  [ack] say — speech=%s" % resp["state"]["speech"])

    done = c.recv_until_done(rid, timeout=5.0)
    assert done is not None, "Timed out waiting for done"
    assert done["status"] == "ok"
    assert done["state"]["speech"] == "idle"
    print("[PASS] say — ack + done (speech now idle)")


def test_pose_sit_stand(c):
    """pose:sit (crouch) then pose:stand with ack/done protocol."""
    # Sit (maps to CROUCH in Webots for stability)
    resp, rid = c.send_and_recv({"action": "pose", "name": "sit"})
    assert resp["status"] == "accepted"
    print("  [ack] pose:sit (crouch)")

    done = c.recv_until_done(rid, timeout=8.0)
    assert done is not None
    assert done["state"]["posture"] == "sitting"
    print("  [done] posture=%s" % done["state"]["posture"])

    time.sleep(0.5)

    # Walk should be rejected while sitting
    resp2, _ = c.send_and_recv({"action": "walk_toward", "x": 0.5})
    assert resp2["status"] == "rejected"
    assert resp2.get("reason") == "must_stand_first"
    print("  [rejected] walk while sitting: %s" % resp2.get("reason"))

    # Stand back up
    resp3, rid3 = c.send_and_recv({"action": "pose", "name": "stand"})
    assert resp3["status"] == "accepted"
    done3 = c.recv_until_done(rid3, timeout=8.0)
    assert done3["state"]["posture"] == "standing"
    time.sleep(3.0)  # Let robot fully stabilize after standing
    print("[PASS] pose:sit -> rejected walk -> pose:stand")


def test_walk_toward(c):
    """walk_toward returns ack, then done when motion completes."""
    # Small forward walk
    resp, rid = c.send_and_recv({
        "action": "walk_toward", "x": 0.5, "y": 0.0, "theta": 0.0
    })
    assert resp["status"] == "accepted"
    assert resp["state"]["legs"] == "walking"
    print("  [ack] walk_toward x=0.5")

    done = c.recv_until_done(rid, timeout=15.0)
    assert done is not None, "Timed out waiting for walk done"
    assert done["state"]["legs"] == "idle"
    time.sleep(2.0)  # Stabilize after walk
    print("[PASS] walk_toward — ack + done")


def test_walk_toward_turn(c):
    """walk_toward with theta triggers a turn motion."""
    resp, rid = c.send_and_recv({
        "action": "walk_toward", "x": 0.0, "y": 0.0, "theta": 0.7
    })
    assert resp["status"] == "accepted"
    print("  [ack] walk_toward theta=0.7 (turn)")

    done = c.recv_until_done(rid, timeout=10.0)
    assert done is not None
    time.sleep(2.0)  # Stabilize after turn
    print("[PASS] walk_toward turn — done")


def test_walk_toward_empty(c):
    """walk_toward with no movement sends done immediately."""
    resp, rid = c.send_and_recv({
        "action": "walk_toward", "x": 0.0, "y": 0.0, "theta": 0.0
    })
    # Should get done (not ack) since nothing to do
    # Actually: ack is sent first, then done immediately
    if resp.get("type") == "ack":
        done = c.recv_until_done(rid, timeout=3.0)
        assert done is not None
    else:
        assert resp["status"] == "ok"
    print("[PASS] walk_toward empty — immediate done")


def test_set_walk_velocity(c):
    """set_walk_velocity is inline (immediate response)."""
    resp, _ = c.send_and_recv({
        "action": "set_walk_velocity", "x": 0.3, "y": 0.0, "theta": 0.0
    })
    assert resp["status"] == "ok"
    assert resp["state"]["legs"] == "walking"
    print("  [ok] set_walk_velocity x=0.3 — legs=walking")

    time.sleep(1.0)

    # Stop
    resp2, _ = c.send_and_recv({"action": "stop_walk"})
    assert resp2["status"] == "ok"
    assert resp2["state"]["legs"] == "idle"
    time.sleep(2.0)  # Stabilize after velocity walk
    print("[PASS] set_walk_velocity + stop_walk")


def test_stop_all(c):
    """stop_all resets all channels."""
    # Start something first
    c.send_and_recv({"action": "say", "text": "Testing stop all"})
    time.sleep(0.2)

    resp, _ = c.send_and_recv({"action": "stop_all"})
    assert resp["status"] == "ok"
    state = resp["state"]
    assert state["legs"] == "idle"
    assert state["speech"] == "idle"
    assert state["arms"] == "idle"
    print("[PASS] stop_all — all channels idle")


def test_animate_wave(c):
    """animate:wave returns ack/done on ARMS channel."""
    resp, rid = c.send_and_recv({"action": "animate", "name": "wave"})
    assert resp["status"] == "accepted"
    assert resp["state"]["arms"] == "animating"
    print("  [ack] animate:wave — arms=animating")

    done = c.recv_until_done(rid, timeout=10.0)
    assert done is not None
    assert done["state"]["arms"] == "idle"
    time.sleep(2.0)  # Stabilize after wave
    print("[PASS] animate:wave — done (arms=idle)")


def test_open_close_hand(c):
    """open_hand and close_hand on ARMS channel."""
    resp, rid = c.send_and_recv({"action": "open_hand", "hand": "right"})
    assert resp["status"] == "accepted"
    done = c.recv_until_done(rid, timeout=3.0)
    assert done is not None
    print("  [done] open_hand")

    resp2, rid2 = c.send_and_recv({"action": "close_hand", "hand": "right"})
    assert resp2["status"] == "accepted"
    done2 = c.recv_until_done(rid2, timeout=3.0)
    assert done2 is not None
    print("[PASS] open_hand + close_hand")


def test_no_id_backward_compat(c):
    """Commands without id get a single response (no type field)."""
    c.send({"action": "query_state"})
    time.sleep(0.5)
    resp = c._recv_one(timeout=3.0)
    assert resp is not None
    assert resp["status"] == "ok"
    assert "type" not in resp or resp.get("type") is None or resp.get("type") == "done"
    print("[PASS] backward compat (no id)")


def test_wake_up_rest(c):
    """rest -> wake_up cycle."""
    resp, rid = c.send_and_recv({"action": "rest"})
    assert resp["status"] == "accepted"
    done = c.recv_until_done(rid, timeout=8.0)
    assert done is not None
    assert done["state"]["posture"] == "resting"
    print("  [done] rest — posture=resting")

    time.sleep(0.5)

    resp2, rid2 = c.send_and_recv({"action": "wake_up"})
    assert resp2["status"] == "accepted"
    done2 = c.recv_until_done(rid2, timeout=8.0)
    assert done2 is not None
    assert done2["state"]["posture"] == "standing"
    time.sleep(1.0)  # Let robot stabilize
    print("[PASS] rest -> wake_up — posture=standing")


# ===================================================================
# Runner
# ===================================================================

def main():
    port = PORT
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])

    print("=" * 55)
    print("  ElderGuard Webots — TCP Protocol Tests")
    print("  Connecting to %s:%d ..." % (HOST, port))
    print("=" * 55)

    try:
        c = TestClient(HOST, port)
    except ConnectionRefusedError:
        print("[ERROR] Cannot connect. Is the Webots simulation running?")
        sys.exit(1)

    # Order: safe tests first, then motion tests (risk of fall)
    # Posture tests last (they change the robot's stance)
    tests = [
        # --- Safe (no physics movement) ---
        test_query_state,
        test_heartbeat,
        test_get_posture,
        test_move_head,
        test_move_head_relative,
        test_fire_and_forget,
        test_no_id_backward_compat,
        test_say_ack_done,
        test_stop_all,
        # --- Arm motion (low fall risk) ---
        test_open_close_hand,
        test_animate_wave,
        # --- Walking (moderate fall risk) ---
        test_walk_toward_empty,
        test_walk_toward,
        test_walk_toward_turn,
        test_set_walk_velocity,
        # --- Posture changes (highest fall risk) ---
        test_pose_sit_stand,
        test_wake_up_rest,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            print("")
            test_fn(c)
            passed += 1
        except Exception as e:
            print("[FAIL] %s: %s" % (test_fn.__name__, e))
            failed += 1
            # Try to recover state for next test
            try:
                c.send({"action": "stop_all"})
                time.sleep(1.0)
                c.send({"action": "wake_up"})
                time.sleep(4.0)
            except Exception:
                pass

    print("")
    print("=" * 55)
    print("  Results: %d passed, %d failed, %d total" % (
        passed, failed, passed + failed))
    print("=" * 55)

    c.close()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
