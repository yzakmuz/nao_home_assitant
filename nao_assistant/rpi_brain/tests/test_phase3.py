#!/usr/bin/env python3
"""
test_phase3.py -- Integration test for Phase 3 changes.

Tests NaoStateCache, background reader thread, two-phase pending
request resolution, send_command_and_wait_done, and connection
failure handling -- all without real hardware.
"""

import json
import os
import socket
import sys
import threading
import time

# Ensure rpi_brain/ is on sys.path (tests now live in rpi_brain/tests/)
_RPI_BRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _RPI_BRAIN_DIR)


def run_tests():
    passed = 0

    # ============================================================
    # Test NaoStateCache independently
    # ============================================================
    from comms.tcp_client import NaoStateCache, _PendingRequest

    # --- Test 1: NaoStateCache init ---
    cache = NaoStateCache()
    assert cache.posture == "unknown", "T1a"
    assert cache.head_yaw == 0.0, "T1b"
    print("  [PASS] Test 1: NaoStateCache initializes correctly")
    passed += 1

    # --- Test 2: NaoStateCache update ---
    cache.update({
        "posture": "standing",
        "legs": "walking",
        "head_yaw": 0.123,
        "head_pitch": -0.05,
    })
    assert cache.posture == "standing", "T2a"
    assert cache.legs == "walking", "T2b"
    assert abs(cache.head_yaw - 0.123) < 0.001, "T2c"
    print("  [PASS] Test 2: NaoStateCache update works")
    passed += 1

    # --- Test 3: NaoStateCache snapshot ---
    snap = cache.snapshot()
    assert snap["posture"] == "standing", "T3a"
    assert snap["legs"] == "walking", "T3b"
    assert "head_yaw" in snap, "T3c"
    print("  [PASS] Test 3: NaoStateCache snapshot returns all fields")
    passed += 1

    # --- Test 4: NaoStateCache ignores unknown keys ---
    cache.update({"unknown_field": 42, "posture": "sitting"})
    assert cache.posture == "sitting", "T4a"
    assert not hasattr(cache, "unknown_field"), "T4b"
    print("  [PASS] Test 4: NaoStateCache ignores unknown keys")
    passed += 1

    # ============================================================
    # Test _PendingRequest
    # ============================================================

    # --- Test 5: PendingRequest event flow ---
    pr = _PendingRequest()
    assert not pr.ack_event.is_set(), "T5a"
    assert not pr.done_event.is_set(), "T5b"
    assert pr.ack_data is None, "T5c"
    pr.ack_data = {"type": "ack", "status": "accepted"}
    pr.ack_event.set()
    assert pr.ack_event.is_set(), "T5d"
    assert pr.ack_data["status"] == "accepted", "T5e"
    print("  [PASS] Test 5: PendingRequest ack event flow works")
    passed += 1

    # ============================================================
    # Test NaoTcpClient with a mock TCP server
    # ============================================================
    from comms.tcp_client import NaoTcpClient

    # Create a simple mock TCP server
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    test_port = server_sock.getsockname()[1]

    # Container for the accepted server-side connection
    server_conn = [None]
    server_ready = threading.Event()

    def accept_one():
        conn, _ = server_sock.accept()
        conn.settimeout(5.0)
        server_conn[0] = conn
        server_ready.set()

    accept_thread = threading.Thread(target=accept_one, daemon=True)
    accept_thread.start()

    # --- Test 6: Connect to mock server ---
    client = NaoTcpClient(host="127.0.0.1", port=test_port, timeout=3.0)
    ok = client.connect()
    assert ok, "T6a: connect failed"
    assert client.is_connected, "T6b"
    server_ready.wait(timeout=3.0)
    assert server_conn[0] is not None, "T6c: server did not accept"
    print("  [PASS] Test 6: NaoTcpClient connects to mock server")
    passed += 1

    # Helper: read a JSON command from server side (skips heartbeats)
    _server_buf = [b""]  # mutable buffer shared across calls

    def server_recv():
        conn = server_conn[0]
        while True:
            while b"\n" not in _server_buf[0]:
                chunk = conn.recv(4096)
                if not chunk:
                    return None
                _server_buf[0] += chunk
            line, _server_buf[0] = _server_buf[0].split(b"\n", 1)
            msg = json.loads(line.decode("utf-8"))
            # Skip heartbeat messages (Phase 5 heartbeat thread)
            if msg.get("action") == "heartbeat":
                continue
            return msg

    # Helper: send a JSON response from server side
    def server_send(response):
        conn = server_conn[0]
        data = (json.dumps(response) + "\n").encode("utf-8")
        conn.sendall(data)

    # --- Test 7: send_command (blocking command) ---
    def respond_blocking():
        cmd = server_recv()
        req_id = cmd["id"]
        server_send({
            "id": req_id,
            "type": "done",
            "status": "ok",
            "action": "say",
            "state": {"posture": "standing", "head": "idle", "legs": "idle",
                      "speech": "idle", "arms": "idle"},
        })

    resp_thread = threading.Thread(target=respond_blocking, daemon=True)
    resp_thread.start()

    resp = client.send_command({"action": "say", "text": "hello"})
    assert resp is not None, "T7a: no response"
    assert resp["status"] == "ok", "T7b: status=%s" % resp.get("status")
    assert resp["type"] == "done", "T7c: type=%s" % resp.get("type")
    assert client.nao_state.posture == "standing", "T7d: state not updated"
    resp_thread.join(timeout=2.0)
    print("  [PASS] Test 7: send_command blocking -> done response + state update")
    passed += 1

    # --- Test 8: send_command (threaded command — ack) ---
    def respond_two_phase():
        cmd = server_recv()
        req_id = cmd["id"]
        # Phase 1: ack
        server_send({
            "id": req_id,
            "type": "ack",
            "status": "accepted",
            "action": "walk_toward",
            "state": {"posture": "standing", "head": "idle", "legs": "walking",
                      "speech": "idle", "arms": "idle"},
        })
        # Phase 2: done (after short delay)
        time.sleep(0.2)
        server_send({
            "id": req_id,
            "type": "done",
            "status": "ok",
            "action": "walk_toward",
            "state": {"posture": "standing", "head": "idle", "legs": "idle",
                      "speech": "idle", "arms": "idle"},
        })

    resp_thread = threading.Thread(target=respond_two_phase, daemon=True)
    resp_thread.start()

    resp = client.send_command({"action": "walk_toward", "x": 0.5})
    assert resp is not None, "T8a"
    assert resp["status"] == "accepted", "T8b: %s" % resp.get("status")
    assert resp["type"] == "ack", "T8c"
    assert client.nao_state.legs == "walking", "T8d: legs should be walking"
    print("  [PASS] Test 8: send_command threaded -> ack response (non-blocking)")
    passed += 1

    # Wait for done to arrive and update state
    resp_thread.join(timeout=2.0)
    time.sleep(0.3)
    assert client.nao_state.legs == "idle", "T8e: legs should be idle after done"
    print("  [PASS] Test 8b: done response updates state cache asynchronously")
    passed += 1

    # --- Test 9: send_command_and_wait_done ---
    def respond_walk_full():
        cmd = server_recv()
        req_id = cmd["id"]
        server_send({
            "id": req_id,
            "type": "ack",
            "status": "accepted",
            "action": "pose",
            "state": {"posture": "standing", "legs": "idle"},
        })
        time.sleep(0.2)
        server_send({
            "id": req_id,
            "type": "done",
            "status": "ok",
            "action": "pose",
            "state": {"posture": "sitting", "legs": "idle"},
        })

    resp_thread = threading.Thread(target=respond_walk_full, daemon=True)
    resp_thread.start()

    resp = client.send_command_and_wait_done(
        {"action": "pose", "name": "sit"}, timeout=5.0
    )
    assert resp is not None, "T9a"
    assert resp["status"] == "ok", "T9b"
    assert resp["type"] == "done", "T9c"
    assert client.nao_state.posture == "sitting", "T9d"
    resp_thread.join(timeout=2.0)
    print("  [PASS] Test 9: send_command_and_wait_done waits for done")
    passed += 1

    # --- Test 10: send_command_and_wait_done with blocking command ---
    def respond_say():
        cmd = server_recv()
        req_id = cmd["id"]
        server_send({
            "id": req_id,
            "type": "done",
            "status": "ok",
            "action": "say",
            "state": {"posture": "sitting"},
        })

    resp_thread = threading.Thread(target=respond_say, daemon=True)
    resp_thread.start()

    resp = client.send_command_and_wait_done(
        {"action": "say", "text": "test"}, timeout=5.0
    )
    assert resp is not None, "T10a"
    assert resp["type"] == "done", "T10b"
    resp_thread.join(timeout=2.0)
    print("  [PASS] Test 10: send_command_and_wait_done returns immediately for blocking")
    passed += 1

    # --- Test 11: send_command with rejection ---
    def respond_reject():
        cmd = server_recv()
        req_id = cmd["id"]
        server_send({
            "id": req_id,
            "type": "ack",
            "status": "rejected",
            "reason": "must_stand_first",
            "action": "walk_toward",
            "state": {"posture": "sitting"},
        })

    resp_thread = threading.Thread(target=respond_reject, daemon=True)
    resp_thread.start()

    resp = client.send_command({"action": "walk_toward", "x": 0.5})
    assert resp is not None, "T11a"
    assert resp["status"] == "rejected", "T11b"
    assert resp.get("reason") == "must_stand_first", "T11c"
    resp_thread.join(timeout=2.0)
    print("  [PASS] Test 11: send_command handles rejection correctly")
    passed += 1

    # --- Test 12: send_command_and_wait_done with rejection ---
    def respond_reject2():
        cmd = server_recv()
        req_id = cmd["id"]
        server_send({
            "id": req_id,
            "type": "ack",
            "status": "rejected",
            "reason": "arms_busy",
            "action": "animate",
            "state": {"posture": "standing", "arms": "animating"},
        })

    resp_thread = threading.Thread(target=respond_reject2, daemon=True)
    resp_thread.start()

    resp = client.send_command_and_wait_done({"action": "animate", "name": "wave"})
    assert resp is not None, "T12a"
    assert resp["status"] == "rejected", "T12b"
    resp_thread.join(timeout=2.0)
    print("  [PASS] Test 12: send_command_and_wait_done returns rejected (no done wait)")
    passed += 1

    # --- Test 13: fire-and-forget ---
    ok = client.send_fire_and_forget(
        {"action": "move_head", "yaw": 0.5, "pitch": 0.0}
    )
    assert ok, "T13a"
    cmd = server_recv()
    assert cmd["action"] == "move_head", "T13b"
    assert cmd.get("no_ack") is True, "T13c: no_ack should be True"
    assert "id" not in cmd, "T13d: fire-and-forget should not have id"
    print("  [PASS] Test 13: fire-and-forget sends no_ack, no id")
    passed += 1

    # --- Test 14: send_command auto-generates unique ids ---
    ids = set()
    for _ in range(5):
        rid = client._next_id()
        assert rid not in ids, "T14: duplicate id!"
        ids.add(rid)
    print("  [PASS] Test 14: request IDs are unique")
    passed += 1

    # --- Test 15: query_state with head angles ---
    def respond_query():
        cmd = server_recv()
        req_id = cmd["id"]
        server_send({
            "id": req_id,
            "type": "done",
            "status": "ok",
            "action": "query_state",
            "state": {
                "posture": "standing", "head": "idle", "legs": "idle",
                "speech": "idle", "arms": "idle",
                "head_yaw": 0.456, "head_pitch": -0.123,
            },
        })

    resp_thread = threading.Thread(target=respond_query, daemon=True)
    resp_thread.start()

    resp = client.send_command({"action": "query_state"})
    assert resp is not None, "T15a"
    assert abs(client.nao_state.head_yaw - 0.456) < 0.001, "T15b"
    assert abs(client.nao_state.head_pitch - (-0.123)) < 0.001, "T15c"
    resp_thread.join(timeout=2.0)
    print("  [PASS] Test 15: query_state updates head angles in cache")
    passed += 1

    # --- Test 16: Timeout handling ---
    # Don't respond from server — send_command should timeout
    resp = client.send_command({"action": "say", "text": "timeout"}, timeout=0.5)
    # Server received the command but we timed out
    _ = server_recv()  # consume the command from server side
    assert resp is None, "T16: should return None on timeout"
    print("  [PASS] Test 16: send_command returns None on timeout")
    passed += 1

    # --- Test 17: Disconnect ---
    client.disconnect()
    assert not client.is_connected, "T17a"
    print("  [PASS] Test 17: disconnect works cleanly")
    passed += 1

    # --- Test 18: send_command after disconnect ---
    resp = client.send_command({"action": "say", "text": "offline"})
    assert resp is None, "T18: should return None when disconnected"
    print("  [PASS] Test 18: send_command returns None when disconnected")
    passed += 1

    # Cleanup
    try:
        server_conn[0].close()
    except Exception:
        pass
    server_sock.close()

    print()
    print("=" * 50)
    print("  ALL %d TESTS PASSED" % passed)
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
