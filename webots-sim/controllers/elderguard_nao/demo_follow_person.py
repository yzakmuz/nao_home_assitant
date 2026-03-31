#!/usr/bin/env python3
"""
Demo 1: Follow Person
=====================
NAO detects the virtual person's position, announces intent,
walks toward the person while tracking with its head, and waves on arrival.

Usage:
    1. Start Webots with elderguard_room.wbt
    2. Wait for "TCP command server on port 5555"
    3. Run: python demo_follow_person.py

Requires: Webots controller running on localhost:5555
"""

import socket
import json
import time
import math
import sys

HOST = "127.0.0.1"
PORT = 5555
TIMEOUT = 12.0  # seconds per command
CLOSE_ENOUGH = 0.7  # meters — stop when this close


class DemoClient:
    """TCP client for sending JSON commands to the Webots NAO controller."""

    def __init__(self, host=HOST, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.settimeout(TIMEOUT)
        self._buf = ""
        self._req_counter = 0

    def _next_id(self):
        self._req_counter += 1
        return "demo_%d" % self._req_counter

    def _recv_line(self, timeout=None):
        """Read one newline-delimited JSON message."""
        old_timeout = self.sock.gettimeout()
        if timeout is not None:
            self.sock.settimeout(timeout)
        try:
            while "\n" not in self._buf:
                data = self.sock.recv(4096).decode("utf-8")
                if not data:
                    return None
                self._buf += data
            line, self._buf = self._buf.split("\n", 1)
            return json.loads(line)
        except socket.timeout:
            return None
        finally:
            self.sock.settimeout(old_timeout)

    def send_inline(self, cmd, description=""):
        """Send a command and get the single inline response."""
        cmd["id"] = self._next_id()
        msg = json.dumps(cmd) + "\n"
        self.sock.send(msg.encode("utf-8"))
        resp = self._recv_line(timeout=TIMEOUT)
        if resp and description:
            status = resp.get("status", "?")
            print("  [%s] %s" % (status.upper(), description))
        return resp

    def send_and_wait(self, cmd, description="", timeout=TIMEOUT):
        """Send async command, wait for ack then done."""
        cmd["id"] = self._next_id()
        req_id = cmd["id"]
        msg = json.dumps(cmd) + "\n"
        self.sock.send(msg.encode("utf-8"))

        # Wait for ack
        resp = self._recv_line(timeout=timeout)
        if not resp:
            print("  [TIMEOUT] %s — no ack" % description)
            return None

        # If it's already "done" (inline-style), return it
        if resp.get("type") == "done" or resp.get("id") != req_id:
            if description:
                print("  [%s] %s" % (resp.get("status", "?").upper(), description))
            return resp

        # Got ack — wait for done
        if resp.get("status") == "rejected":
            print("  [REJECTED] %s — %s" % (description, resp.get("reason", "")))
            return resp

        done = self._recv_line(timeout=timeout)
        if done and description:
            print("  [%s] %s" % (done.get("status", "?").upper(), description))
        return done

    def send_fire_and_forget(self, cmd):
        """Send with no_ack — no response expected."""
        cmd["no_ack"] = True
        msg = json.dumps(cmd) + "\n"
        self.sock.send(msg.encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def compute_distance(pos_a, pos_b):
    """Horizontal distance (XY plane, Z-up world) between two position dicts."""
    dx = pos_a["x"] - pos_b["x"]
    dy = pos_a["y"] - pos_b["y"]
    return math.sqrt(dx * dx + dy * dy)


def main():
    print("=" * 50)
    print("  Demo 1: Follow Person")
    print("=" * 50)
    print()

    # Connect
    try:
        client = DemoClient()
    except ConnectionRefusedError:
        print("[ERROR] Cannot connect to localhost:%d" % PORT)
        print("        Make sure Webots is running with elderguard_room.wbt")
        sys.exit(1)

    print("  Connected to NAO controller on port %d" % PORT)
    print()

    try:
        # Step 1: Query state
        resp = client.send_inline(
            {"action": "query_state"},
            "Query state")
        if resp:
            state = resp.get("state", {})
            posture = state.get("posture", "unknown")
            print("      Posture: %s" % posture)
            if posture != "standing":
                print("      NAO is not standing — sending wake_up...")
                client.send_and_wait(
                    {"action": "wake_up"},
                    "Wake up", timeout=5.0)
                time.sleep(2.0)

        # Step 2: Get person position
        resp = client.send_inline(
            {"action": "get_person_position"},
            "Get person position")
        if not resp or not resp.get("person_found"):
            print("  [ERROR] No person found in world!")
            client.close()
            sys.exit(1)

        person_pos = resp["person_position"]
        nao_pos = resp.get("nao_position", {"x": 0, "y": 0, "z": 0})
        dist = compute_distance(person_pos, nao_pos)
        print("      Person at (%.2f, %.2f, %.2f)" % (
            person_pos["x"], person_pos["y"], person_pos["z"]))
        print("      NAO at (%.2f, %.2f, %.2f)" % (
            nao_pos["x"], nao_pos["y"], nao_pos["z"]))
        print("      Distance: %.2fm" % dist)
        print()

        # Step 3: Announce intent
        client.send_and_wait(
            {"action": "say", "text": "I see you. Let me come to you."},
            "Say: announce intent")
        time.sleep(0.5)

        # Step 4: Start follow_person with walk
        client.send_inline(
            {"action": "follow_person", "walk": True},
            "Start follow (HEAD + WALK)")
        print()

        # Step 5: Monitor distance until close enough
        print("  Walking toward person...")
        max_time = 60  # safety timeout
        start = time.time()
        while time.time() - start < max_time:
            time.sleep(2.0)
            resp = client.send_inline(
                {"action": "get_person_position"}, "")
            if resp and resp.get("person_found"):
                p = resp["person_position"]
                n = resp.get("nao_position", nao_pos)
                d = compute_distance(p, n)
                print("      Distance: %.2fm" % d)
                if d < CLOSE_ENOUGH:
                    print("      Close enough!")
                    break
        print()

        # Step 6: Stop follow
        client.send_inline(
            {"action": "stop_follow"},
            "Stop follow")
        client.send_inline(
            {"action": "stop_walk"},
            "Stop walk")
        time.sleep(0.5)

        # Step 7: Announce arrival
        client.send_and_wait(
            {"action": "say", "text": "I am here. How can I help you?"},
            "Say: arrival")
        time.sleep(0.5)

        # Step 8: Wave hello
        client.send_and_wait(
            {"action": "animate", "name": "wave"},
            "Wave hello")

        print()
        print("  Demo complete!")

    except KeyboardInterrupt:
        print("\n  [Interrupted by user]")
        client.send_inline({"action": "stop_all"}, "Emergency stop")
    except Exception as e:
        print("\n  [ERROR] %s" % e)
        try:
            client.send_inline({"action": "stop_all"}, "Emergency stop")
        except Exception:
            pass
    finally:
        client.close()

    print("=" * 50)


if __name__ == "__main__":
    main()
