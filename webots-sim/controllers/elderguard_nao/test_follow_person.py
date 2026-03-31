"""
test_follow_person.py — Test built-in person tracking.

Sends a single 'follow_person' command to the Webots controller.
The controller handles ALL navigation internally using Supervisor API
(direct access to GPS, inertial unit, person position — no angle
convention issues).

Usage:
    python test_follow_person.py              (head tracking only)
    python test_follow_person.py --walk       (head + walk toward person)
    python test_follow_person.py --stop       (stop tracking)
"""

import json
import socket
import sys
import time

HOST = "127.0.0.1"
PORT = 5555


def send(sock, cmd):
    data = json.dumps(cmd) + "\n"
    sock.sendall(data.encode("utf-8"))


def recv(sock):
    buf = ""
    sock.settimeout(5.0)
    while "\n" not in buf:
        data = sock.recv(4096)
        if not data:
            return None
        buf += data.decode("utf-8")
    line = buf.split("\n")[0].strip()
    return json.loads(line) if line else None


def main():
    walk = "--walk" in sys.argv
    stop = "--stop" in sys.argv

    print("=" * 50)
    print("  ElderGuard NAO — Person Tracking Test")
    print("=" * 50)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((HOST, PORT))
    except ConnectionRefusedError:
        print("[ERROR] Cannot connect. Is Webots running?")
        sys.exit(1)
    print("[OK] Connected")

    if stop:
        # Stop tracking
        cmd = {"action": "stop_follow", "id": "stop1"}
        send(sock, cmd)
        resp = recv(sock)
        print("[OK] Tracking stopped")
        # Also stop walking
        send(sock, {"action": "stop_all", "id": "stop2"})
        recv(sock)
        sock.close()
        return

    # Check person exists
    cmd = {"action": "get_person_position", "id": "check1"}
    send(sock, cmd)
    resp = recv(sock)
    if not resp or not resp.get("person_found"):
        print("[ERROR] No person in world")
        sock.close()
        sys.exit(1)

    pp = resp["person_position"]
    np_ = resp.get("nao_position", {})
    print("Person at: (%.2f, %.2f, %.2f)" % (pp["x"], pp["y"], pp["z"]))
    print("NAO at:    (%.2f, %.2f, %.2f)" % (
        np_.get("x", 0), np_.get("y", 0), np_.get("z", 0)))

    # Start tracking
    mode = "HEAD + WALK" if walk else "HEAD ONLY"
    print("\nStarting: %s tracking" % mode)
    print("The controller handles all navigation internally.")
    print("Watch the Webots 3D view — NAO head should follow the person.")
    if walk:
        print("NAO will also walk toward the person.")
    print("\nPress Ctrl+C to stop.\n")

    cmd = {"action": "follow_person", "walk": walk, "id": "follow1"}
    send(sock, cmd)
    resp = recv(sock)
    print("[OK] follow_person command sent (status: %s)" % (
        resp.get("status", "?") if resp else "no response"))

    # Keep connection alive and periodically print position
    try:
        while True:
            time.sleep(2.0)
            send(sock, {"action": "get_person_position", "id": "pos"})
            resp = recv(sock)
            if resp and resp.get("nao_position"):
                np_ = resp["nao_position"]
                pp = resp.get("person_position", {})
                dx = pp.get("x", 0) - np_.get("x", 0)
                dz = pp.get("z", 0) - np_.get("z", 0)
                import math
                dist = math.sqrt(dx * dx + dz * dz)
                print("  dist=%.2fm  NAO=(%.2f,%.2f)  Person=(%.2f,%.2f)" % (
                    dist, np_["x"], np_["z"], pp.get("x", 0), pp.get("z", 0)))
    except KeyboardInterrupt:
        print("\n[STOP] Stopping tracking...")
        send(sock, {"action": "stop_follow", "no_ack": True})
        send(sock, {"action": "stop_all", "no_ack": True})
        time.sleep(0.5)

    sock.close()
    print("[OK] Done")


if __name__ == "__main__":
    main()
