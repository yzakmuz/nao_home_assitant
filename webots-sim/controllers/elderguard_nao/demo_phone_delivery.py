#!/usr/bin/env python3
"""
Demo 2: Phone Delivery
======================
NAO locates a phone on the ground, walks to it, picks it up,
carries it to the virtual person, and delivers it.

Usage:
    1. Start Webots with elderguard_room.wbt
    2. Wait for "TCP command server on port 5555"
    3. Run: python demo_phone_delivery.py

Requires: Webots controller running on localhost:5555
"""

import socket
import json
import time
import math
import sys

HOST = "127.0.0.1"
PORT = 5555
TIMEOUT = 15.0        # seconds per regular command
NAV_TIMEOUT = 60.0    # seconds for navigation (walking takes time)
PHONE_STOP_DIST = 0.35  # meters — stop distance for phone pickup
PERSON_STOP_DIST = 0.6  # meters — stop distance for person delivery


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
        if resp.get("type") == "done":
            if description:
                print("  [%s] %s" % (resp.get("status", "?").upper(), description))
            return resp

        # Got ack — check for rejection
        if resp.get("status") == "rejected":
            print("  [REJECTED] %s — %s" % (description, resp.get("reason", "")))
            return resp

        # Wait for done
        done = self._recv_line(timeout=timeout)
        if done and description:
            status = done.get("status", "?").upper()
            extra = ""
            if done.get("arrived"):
                extra = " (dist=%.3fm)" % done.get("final_distance", 0)
            print("  [%s] %s%s" % (status, description, extra))
        elif not done:
            print("  [TIMEOUT] %s — no done" % description)
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


def main():
    print("=" * 55)
    print("  Demo 2: Phone Delivery")
    print("=" * 55)
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
        # ============================================================
        # Phase A: Initialization
        # ============================================================
        print("  Phase A: Initialization")
        print("  " + "-" * 40)

        # A1: Query state
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

        # A2: Get person position
        resp = client.send_inline(
            {"action": "get_person_position"},
            "Get person position")
        if not resp or not resp.get("person_found"):
            print("  [ERROR] No person found in world!")
            client.close()
            sys.exit(1)
        person_pos = resp["person_position"]
        print("      Person at (%.2f, %.2f, %.2f)" % (
            person_pos["x"], person_pos["y"], person_pos["z"]))

        # A3: Get phone position
        resp = client.send_inline(
            {"action": "get_object_position", "name": "PHONE"},
            "Get phone position")
        if not resp or not resp.get("object_found"):
            print("  [ERROR] Phone not found! Make sure world has DEF PHONE.")
            client.close()
            sys.exit(1)
        phone_pos = resp["object_position"]
        nao_pos = resp.get("nao_position", {"x": 0, "y": 0, "z": 0})
        print("      Phone at (%.2f, %.2f, %.2f)" % (
            phone_pos["x"], phone_pos["y"], phone_pos["z"]))
        print("      NAO at (%.2f, %.2f, %.2f)" % (
            nao_pos["x"], nao_pos["y"], nao_pos["z"]))
        print()

        # ============================================================
        # Phase B: Navigate to Phone
        # ============================================================
        print("  Phase B: Navigate to Phone")
        print("  " + "-" * 40)

        # B1: Announce
        client.send_and_wait(
            {"action": "say",
             "text": "I see your phone on the floor. Let me get it for you."},
            "Say: announce")
        time.sleep(0.5)

        # B2: Navigate to phone
        print("  Navigating to phone (stop at %.2fm)..." % PHONE_STOP_DIST)
        resp = client.send_and_wait(
            {"action": "navigate_to",
             "x": phone_pos["x"],
             "y": phone_pos["y"],
             "z": phone_pos["z"],
             "stop_distance": PHONE_STOP_DIST},
            "Navigate to phone",
            timeout=NAV_TIMEOUT)

        if not resp or not resp.get("arrived"):
            print("  [WARN] Navigation may not have completed")
        time.sleep(1.0)
        print()

        # ============================================================
        # Phase C: Pick Up Phone
        # ============================================================
        print("  Phase C: Pick Up Phone")
        print("  " + "-" * 40)

        # C1: Announce pickup
        client.send_fire_and_forget(
            {"action": "say", "text": "Let me pick this up."})
        time.sleep(1.0)

        # C2: Run pickup sequence (10 seconds)
        print("  Running pickup sequence (10s)...")
        resp = client.send_and_wait(
            {"action": "pickup_sequence", "hand": "right"},
            "Pickup sequence",
            timeout=15.0)
        time.sleep(0.5)

        # C3: Attach phone to hand
        client.send_and_wait(
            {"action": "start_carrying", "name": "PHONE"},
            "Attach phone to hand")
        time.sleep(0.5)
        print()

        # ============================================================
        # Phase D: Navigate to Person
        # ============================================================
        print("  Phase D: Navigate to Person")
        print("  " + "-" * 40)

        # D1: Announce
        client.send_and_wait(
            {"action": "say", "text": "Got it! Bringing it to you now."},
            "Say: bringing to you")
        time.sleep(0.5)

        # D2: Re-query person position (in case it moved)
        resp = client.send_inline(
            {"action": "get_person_position"}, "")
        if resp and resp.get("person_found"):
            person_pos = resp["person_position"]

        # D3: Navigate to person
        print("  Navigating to person (stop at %.2fm)..." % PERSON_STOP_DIST)
        resp = client.send_and_wait(
            {"action": "navigate_to",
             "x": person_pos["x"],
             "y": person_pos["y"],
             "z": person_pos["z"],
             "stop_distance": PERSON_STOP_DIST},
            "Navigate to person",
            timeout=NAV_TIMEOUT)

        if not resp or not resp.get("arrived"):
            print("  [WARN] Navigation may not have completed")
        time.sleep(1.0)
        print()

        # ============================================================
        # Phase E: Deliver Phone
        # ============================================================
        print("  Phase E: Deliver Phone")
        print("  " + "-" * 40)

        # E1: Announce
        client.send_and_wait(
            {"action": "say", "text": "Here is your phone."},
            "Say: here is your phone")
        time.sleep(0.5)

        # E2: Release phone from carrying
        client.send_and_wait(
            {"action": "stop_carrying", "name": "PHONE"},
            "Release phone from hand")
        time.sleep(0.3)

        # E3: Offer and release (5 seconds)
        print("  Offering and releasing (5s)...")
        resp = client.send_and_wait(
            {"action": "offer_and_release", "hand": "right"},
            "Offer and release",
            timeout=10.0)
        time.sleep(0.5)
        print()

        # ============================================================
        # Phase F: Wrap Up
        # ============================================================
        print("  Phase F: Complete")
        print("  " + "-" * 40)

        client.send_and_wait(
            {"action": "say",
             "text": "There you go! Is there anything else I can help with?"},
            "Say: done")
        time.sleep(0.5)

        client.send_and_wait(
            {"action": "animate", "name": "wave"},
            "Wave goodbye")

        print()
        print("  Demo complete!")

    except KeyboardInterrupt:
        print("\n  [Interrupted by user]")
        try:
            client.send_inline({"action": "stop_all"}, "Emergency stop")
            client.send_and_wait(
                {"action": "stop_carrying", "name": "PHONE"}, "Release phone")
        except Exception:
            pass
    except Exception as e:
        print("\n  [ERROR] %s" % e)
        import traceback
        traceback.print_exc()
        try:
            client.send_inline({"action": "stop_all"}, "Emergency stop")
        except Exception:
            pass
    finally:
        client.close()

    print("=" * 55)


if __name__ == "__main__":
    main()
