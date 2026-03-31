"""
test_pickup_demo.py — Manual pickup sequence demonstration.

Sends individual TCP commands step by step to make the NAO:
  1. Announce intent
  2. Crouch down (slowly)
  3. Open hand
  4. Reach arm down toward the ground
  5. Close hand (grab object)
  6. Bring arm to carry position (hold close)
  7. Stand back up (slowly)
  8. Announce success

Each step waits for completion and prints what's happening.
Run this while Webots simulation is active.

Usage:
    python test_pickup_demo.py
"""

import json
import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 5555
_counter = 0


def _next_id():
    global _counter
    _counter += 1
    return "pickup_%d" % _counter


class PickupClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(15.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((HOST, PORT))
        self._buf = ""
        print("[Connected to %s:%d]" % (HOST, PORT))

    def send_and_wait(self, cmd, description, timeout=12.0):
        """Send command, wait for ack, then wait for done. Print progress."""
        rid = _next_id()
        cmd["id"] = rid
        data = json.dumps(cmd) + "\n"
        self.sock.sendall(data.encode("utf-8"))

        # Get first response (ack or done)
        resp = self._recv(timeout)
        if resp is None:
            print("  [TIMEOUT] No response for: %s" % description)
            return None

        if resp.get("status") == "rejected":
            print("  [REJECTED] %s — %s" % (description, resp.get("reason")))
            return resp

        # If it's an ack, wait for done
        if resp.get("type") == "ack":
            print("  [ack] %s ..." % description)
            done = self._wait_done(rid, timeout)
            if done:
                print("  [done] %s" % description)
                return done
            else:
                print("  [TIMEOUT] Waiting for done: %s" % description)
                return resp
        else:
            # Inline action — single response
            print("  [ok] %s" % description)
            return resp

    def send_fire_forget(self, cmd):
        cmd["no_ack"] = True
        data = json.dumps(cmd) + "\n"
        self.sock.sendall(data.encode("utf-8"))

    def _recv(self, timeout=12.0):
        self.sock.settimeout(timeout)
        while "\n" not in self._buf:
            try:
                data = self.sock.recv(4096)
                if not data:
                    return None
                self._buf += data.decode("utf-8")
            except socket.timeout:
                return None
        line, self._buf = self._buf.split("\n", 1)
        line = line.strip()
        if not line:
            return self._recv(timeout)
        return json.loads(line)

    def _wait_done(self, rid, timeout=12.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._recv(max(0.5, deadline - time.time()))
            if msg and msg.get("id") == rid and msg.get("type") == "done":
                return msg
        return None

    def close(self):
        self.sock.close()


def run_pickup():
    print("=" * 55)
    print("  ElderGuard NAO — Pickup Object Demo")
    print("=" * 55)

    try:
        c = PickupClient()
    except ConnectionRefusedError:
        print("[ERROR] Cannot connect. Is Webots running?")
        sys.exit(1)

    # Check starting state
    resp = c.send_and_wait({"action": "query_state"}, "Check state")
    if resp:
        state = resp.get("state", {})
        print("  Posture: %s  Legs: %s  Arms: %s" % (
            state.get("posture"), state.get("legs"), state.get("arms")))
        if state.get("posture") != "standing":
            print("\n  Robot not standing — sending wake_up first...")
            c.send_and_wait({"action": "wake_up"}, "Wake up")
            time.sleep(3.0)

    print("\n--- Phase 1: Announce ---")
    c.send_and_wait(
        {"action": "say", "text": "I will pick up the object for you."},
        "Say: picking up")
    time.sleep(1.0)

    print("\n--- Phase 2: Look down ---")
    c.send_and_wait(
        {"action": "move_head", "yaw": 0.0, "pitch": 0.4},
        "Look down at the ground")
    time.sleep(1.0)

    print("\n--- Phase 3: Open right hand ---")
    c.send_and_wait(
        {"action": "open_hand", "hand": "right"},
        "Open right hand")
    time.sleep(1.0)

    print("\n--- Phase 4: Crouch down ---")
    c.send_and_wait(
        {"action": "pose", "name": "crouch"},
        "Crouch down (slow)")
    time.sleep(2.0)  # Extra stabilization

    print("\n--- Phase 5: Reach arm down ---")
    c.send_and_wait(
        {"action": "arm_reach_down", "hand": "right"},
        "Extend right arm down")
    time.sleep(1.5)

    print("\n--- Phase 6: Close hand (grab) ---")
    c.send_and_wait(
        {"action": "close_hand", "hand": "right"},
        "Close right hand (grab)")
    time.sleep(1.0)

    print("\n--- Phase 7: Carry position ---")
    c.send_and_wait(
        {"action": "arm_carry_position", "hand": "right"},
        "Arm to carry position")
    time.sleep(1.0)

    print("\n--- Phase 8: Stand back up ---")
    c.send_and_wait(
        {"action": "pose", "name": "stand"},
        "Stand up (slow)")
    time.sleep(3.0)  # Extra stabilization

    print("\n--- Phase 9: Look forward ---")
    c.send_and_wait(
        {"action": "move_head", "yaw": 0.0, "pitch": -0.17},
        "Look forward")
    time.sleep(0.5)

    print("\n--- Phase 10: Announce success ---")
    c.send_and_wait(
        {"action": "say", "text": "I got it! Here you go."},
        "Say: got it")
    time.sleep(1.0)

    print("\n--- Phase 11: Offer object ---")
    c.send_and_wait(
        {"action": "arm_offer_position", "hand": "right"},
        "Extend arm to offer")
    time.sleep(3.0)  # Hold the offer pose

    print("\n--- Phase 12: Release ---")
    c.send_and_wait(
        {"action": "open_hand", "hand": "right"},
        "Open hand (release)")
    time.sleep(1.5)

    print("\n--- Phase 13: Return arm to rest ---")
    c.send_and_wait(
        {"action": "arm_rest_position", "hand": "right"},
        "Arm back to rest")
    time.sleep(1.0)

    print("\n--- Phase 14: Done ---")
    c.send_and_wait(
        {"action": "say", "text": "Object delivered!"},
        "Say: delivered")

    print("\n" + "=" * 55)
    print("  Pickup demo complete!")
    print("=" * 55)

    c.close()


def run_pickup_sequence_builtin():
    """Alternative: test the built-in pickup_sequence action."""
    print("=" * 55)
    print("  ElderGuard NAO — Built-in Pickup Sequence")
    print("=" * 55)

    try:
        c = PickupClient()
    except ConnectionRefusedError:
        print("[ERROR] Cannot connect. Is Webots running?")
        sys.exit(1)

    resp = c.send_and_wait({"action": "query_state"}, "Check state")
    if resp and resp.get("state", {}).get("posture") != "standing":
        c.send_and_wait({"action": "wake_up"}, "Wake up")
        time.sleep(3.0)

    print("\n--- Sending pickup_sequence ---")
    c.send_and_wait(
        {"action": "pickup_sequence", "hand": "right"},
        "Full pickup sequence (built-in)")
    time.sleep(2.0)

    print("\n--- Offering ---")
    c.send_and_wait(
        {"action": "offer_and_release", "hand": "right"},
        "Offer and release (built-in)")

    print("\n" + "=" * 55)
    print("  Built-in pickup sequence complete!")
    print("=" * 55)

    c.close()


if __name__ == "__main__":
    if "--builtin" in sys.argv:
        run_pickup_sequence_builtin()
    else:
        run_pickup()
