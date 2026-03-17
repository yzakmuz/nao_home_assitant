"""
tcp_client.py — Thread-safe, auto-reconnecting JSON-over-TCP client.

Phase 3 additions:
    - NaoStateCache: updated from every NAO response's ``state`` field
    - Background reader thread: dispatches async responses by request id
    - Two-phase pending request tracking (ack + done events)
    - send_command_and_wait_done() for sequenced operations

Phase 5 additions:
    - Heartbeat thread: sends heartbeat every HEARTBEAT_INTERVAL_S
    - Auto-reconnect: exponential backoff on connection loss
    - State resync: query_state after reconnect to sync NaoStateCache
    - Event callbacks: on_event (fall detection), on_reconnect

Protocol:
    Each message is a UTF-8 JSON object terminated by a newline (\\n).
    If a request contains an ``id`` field, the server includes it in
    all responses for that request.  Threaded commands produce two
    responses: ``ack`` (immediate) and ``done`` (when finished).

    Phase 5 adds async server events (type="event") such as fall
    detection, which carry no request id.

Usage:
    client = NaoTcpClient()
    client.on_event = my_event_handler      # fall events, etc.
    client.on_reconnect = my_reconnect_fn   # called after auto-reconnect
    client.connect()
    resp = client.send_command({"action": "say", "text": "Hello!"})
    print(client.nao_state.posture)  # updated from response
    client.disconnect()
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

from settings import (
    NAO_IP,
    NAO_PORT,
    TCP_BUFFER_SIZE,
    TCP_RECONNECT_DELAY_S,
    TCP_TIMEOUT_S,
    HEARTBEAT_INTERVAL_S,
    RECONNECT_MAX_RETRIES,
    RECONNECT_BACKOFF_MAX_S,
)

log = logging.getLogger(__name__)


# ======================================================================
# NAO State Cache
# ======================================================================

class NaoStateCache:
    """Cached snapshot of the NAO's channel states.

    Updated automatically from the ``state`` field in every NAO
    server response.  Thread-safe for reads and writes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.posture: str = "unknown"
        self.head: str = "idle"
        self.legs: str = "idle"
        self.speech: str = "idle"
        self.arms: str = "idle"
        self.head_yaw: float = 0.0
        self.head_pitch: float = 0.0

    def update(self, state_dict: Dict[str, Any]) -> None:
        """Merge *state_dict* into the cache (thread-safe)."""
        with self._lock:
            for key, value in state_dict.items():
                if hasattr(self, key) and key != "_lock":
                    setattr(self, key, value)

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of all cached values."""
        with self._lock:
            return {
                "posture": self.posture,
                "head": self.head,
                "legs": self.legs,
                "speech": self.speech,
                "arms": self.arms,
                "head_yaw": self.head_yaw,
                "head_pitch": self.head_pitch,
            }


# ======================================================================
# Pending Request Tracker
# ======================================================================

class _PendingRequest:
    """Tracks a single in-flight request awaiting ack and/or done."""

    __slots__ = ("ack_event", "done_event", "ack_data", "done_data")

    def __init__(self) -> None:
        self.ack_event = threading.Event()
        self.done_event = threading.Event()
        self.ack_data: Optional[Dict[str, Any]] = None
        self.done_data: Optional[Dict[str, Any]] = None


# ======================================================================
# TCP Client
# ======================================================================

class NaoTcpClient:
    """Manages a persistent TCP connection to the NAO robot server.

    Phase 3 changes from Phase 1:
        * A background reader thread processes all incoming messages.
        * ``send_command`` auto-generates a request ``id`` and waits
          for the server's first response (ack or done).
        * ``send_command_and_wait_done`` additionally waits for the
          ``done`` notification (for threaded commands).
        * ``nao_state`` is updated from every response automatically.

    Phase 5 additions:
        * Heartbeat thread sends keepalive every HEARTBEAT_INTERVAL_S.
        * Auto-reconnect with exponential backoff on connection loss.
        * State resync via query_state after successful reconnect.
        * ``on_event`` callback for async server events (e.g., fall).
        * ``on_reconnect`` callback fires after successful reconnect.
    """

    def __init__(
        self,
        host: str = NAO_IP,
        port: int = NAO_PORT,
        timeout: float = TCP_TIMEOUT_S,
    ) -> None:
        self._host = host
        self._port = port
        self._timeout = timeout

        self._sock: Optional[socket.socket] = None
        self._connected = threading.Event()
        self._shutting_down = False

        # Thread-safe send serialization
        self._send_lock = threading.Lock()

        # Connection lifecycle lock (connect / disconnect)
        self._connect_lock = threading.Lock()

        # Background reader
        self._reader_thread: Optional[threading.Thread] = None

        # Pending request tracking
        self._pending: Dict[str, _PendingRequest] = {}
        self._pending_lock = threading.Lock()

        # Monotonic request id counter
        self._req_counter = 0
        self._req_counter_lock = threading.Lock()

        # State cache — updated from every response
        self.nao_state = NaoStateCache()

        # Phase 5: Heartbeat
        self._heartbeat_thread: Optional[threading.Thread] = None

        # Phase 5: Auto-reconnect
        self._reconnect_thread: Optional[threading.Thread] = None

        # Phase 5: Event callbacks
        self.on_event: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_reconnect: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------------
    # Request ID generation
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        with self._req_counter_lock:
            self._req_counter += 1
            return "r%d" % self._req_counter

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish TCP connection and start the reader thread."""
        with self._connect_lock:
            if self._connected.is_set():
                return True
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((self._host, self._port))
                self._sock = sock
                self._connected.set()
                self._shutting_down = False
                log.info("Connected to NAO at %s:%d", self._host, self._port)

                # Start background reader
                self._reader_thread = threading.Thread(
                    target=self._reader_loop, daemon=True
                )
                self._reader_thread.start()

                # Start heartbeat thread (once)
                if (
                    self._heartbeat_thread is None
                    or not self._heartbeat_thread.is_alive()
                ):
                    self._heartbeat_thread = threading.Thread(
                        target=self._heartbeat_loop, daemon=True
                    )
                    self._heartbeat_thread.start()

                return True
            except (OSError, socket.error) as exc:
                log.warning("Connection failed: %s", exc)
                self._close_socket()
                return False

    def disconnect(self) -> None:
        """Gracefully close the connection and stop the reader."""
        self._shutting_down = True
        self._connected.clear()
        self._close_socket()

        if self._reader_thread is not None:
            self._reader_thread.join(timeout=3.0)
            self._reader_thread = None

        # Heartbeat thread will exit on its own (_shutting_down check)
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=3.0)
            self._heartbeat_thread = None

        # Reconnect thread (if running) will exit on _shutting_down
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=5.0)
            self._reconnect_thread = None

        self._fail_all_pending()
        log.info("Disconnected from NAO.")

    def ensure_connected(self, max_retries: int = 10) -> bool:
        """Reconnect if the socket is down."""
        if self._connected.is_set():
            return True
        if self._shutting_down:
            return False
        log.info("Attempting reconnection to NAO ...")
        attempts = 0
        while not self.connect():
            attempts += 1
            if attempts >= max_retries:
                log.error("Max reconnection attempts (%d) exceeded", max_retries)
                return False
            log.info(
                "Reconnect attempt %d/%d failed, retrying in %.1fs ...",
                attempts, max_retries, TCP_RECONNECT_DELAY_S,
            )
            time.sleep(TCP_RECONNECT_DELAY_S)
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ------------------------------------------------------------------
    # Messaging — public API
    # ------------------------------------------------------------------

    def send_command(
        self,
        command: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send *command* with an auto-generated ``id`` and wait for
        the server's first response (ack or done).

        For **blocking** server actions (say, move_head, stop_walk) the
        response is ``type: "done"``.  For **threaded** actions (walk,
        pose, animate) the response is ``type: "ack"``.

        Returns the response dict, or ``None`` on failure / timeout.
        Thread-safe.
        """
        if not self._connected.is_set():
            if not self.ensure_connected():
                log.error("Cannot send command — no connection to NAO.")
                return None

        req_id = self._next_id()
        cmd = dict(command)
        cmd["id"] = req_id

        pending = _PendingRequest()
        with self._pending_lock:
            self._pending[req_id] = pending

        # --- Send (serialized) ---
        with self._send_lock:
            try:
                payload = (
                    json.dumps(cmd, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                self._sock.sendall(payload)
                log.debug("TX -> %s", cmd)
            except (OSError, socket.error) as exc:
                log.error("Send failed: %s", exc)
                self._on_connection_lost()
                with self._pending_lock:
                    self._pending.pop(req_id, None)
                return None

        # --- Wait for first response (ack or done) ---
        effective_timeout = timeout if timeout is not None else self._timeout
        if not pending.ack_event.wait(timeout=effective_timeout):
            log.warning(
                "Timeout (%.1fs) waiting for response to %s (id=%s)",
                effective_timeout, command.get("action"), req_id,
            )
            with self._pending_lock:
                self._pending.pop(req_id, None)
            return None

        response = pending.ack_data

        # If the first response was "done", clean up (blocking command)
        if response is not None and response.get("type") == "done":
            with self._pending_lock:
                self._pending.pop(req_id, None)

        return response

    def send_command_and_wait_done(
        self,
        command: Dict[str, Any],
        timeout: Optional[float] = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Send *command*, wait for ack, then wait for ``done``.

        For blocking commands this returns immediately (the first
        response IS the done).  For threaded commands this blocks until
        the background motion finishes on the NAO.

        Returns the ``done`` response dict, or ``None`` on failure.
        """
        ack = self.send_command(command, timeout=self._timeout)
        if ack is None:
            return None

        # Blocking command — ack is already the done response
        if ack.get("type") == "done":
            return ack

        # Rejected — no done will follow
        if ack.get("status") == "rejected":
            return ack

        # Threaded command — wait for the done response
        req_id = ack.get("id")
        if req_id is None:
            return ack

        with self._pending_lock:
            pending = self._pending.get(req_id)

        if pending is None:
            return ack

        if not pending.done_event.wait(timeout=timeout):
            log.warning(
                "Timeout (%.1fs) waiting for done on %s (id=%s)",
                timeout, command.get("action"), req_id,
            )
            with self._pending_lock:
                self._pending.pop(req_id, None)
            return ack  # return ack (caller knows it was accepted)

        done_response = pending.done_data
        with self._pending_lock:
            self._pending.pop(req_id, None)

        return done_response if done_response is not None else ack

    def send_fire_and_forget(self, command: Dict[str, Any]) -> bool:
        """Send without waiting for a reply.  No ``id`` is added.

        Returns True on success.
        """
        cmd = dict(command)
        cmd["no_ack"] = True

        if not self._connected.is_set():
            if not self.ensure_connected():
                log.error("Cannot send fire-and-forget — no connection.")
                return False

        with self._send_lock:
            try:
                payload = (
                    json.dumps(cmd, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                self._sock.sendall(payload)
                log.debug("TX (fire&forget) -> %s", cmd)
                return True
            except (OSError, socket.error) as exc:
                log.error("Fire-and-forget failed: %s", exc)
                self._on_connection_lost()
                return False

    # ------------------------------------------------------------------
    # Background reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Continuously read messages from the socket and dispatch them."""
        sock = self._sock  # local ref — survives _sock = None in cleanup
        if sock is None:
            return

        buf = b""
        while self._connected.is_set() and not self._shutting_down:
            try:
                chunk = sock.recv(TCP_BUFFER_SIZE)
            except socket.timeout:
                continue  # expected every TCP_TIMEOUT_S — just loop
            except (OSError, socket.error):
                break

            if not chunk:
                # Server closed connection
                break

            buf += chunk

            # Memory safety — discard if buffer grows unreasonably
            if len(buf) > 65536:
                log.error("Reader buffer overflow (%d bytes) — flushing.", len(buf))
                buf = b""
                continue

            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                try:
                    msg = json.loads(raw_line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                    log.warning("Malformed message from server — skipping.")
                    continue
                log.debug("RX <- %s", msg)
                self._dispatch_message(msg)

        # Reader exiting — connection lost
        if not self._shutting_down:
            log.warning("Reader thread exiting — connection lost.")
            self._on_connection_lost()

    def _dispatch_message(self, msg: Dict[str, Any]) -> None:
        """Route an incoming message to state cache + pending request."""
        # Always update state cache
        state = msg.get("state")
        if state and isinstance(state, dict):
            self.nao_state.update(state)

        # Phase 5: handle async server events (fall, etc.)
        msg_type = msg.get("type")
        if msg_type == "event":
            if self.on_event is not None:
                try:
                    self.on_event(msg)
                except Exception:
                    log.exception("Error in on_event callback")
            return

        # Resolve pending request by id
        req_id = msg.get("id")
        if req_id is None:
            return

        with self._pending_lock:
            pending = self._pending.get(req_id)

        if pending is None:
            return

        if msg_type == "ack":
            pending.ack_data = msg
            pending.ack_event.set()
        elif msg_type == "done" or msg_type is None:
            pending.done_data = msg
            pending.done_event.set()
            # Also unblock ack_event if it hasn't fired yet.
            # This handles blocking commands that skip ack and go
            # straight to done.
            if not pending.ack_event.is_set():
                pending.ack_data = msg
                pending.ack_event.set()

    # ------------------------------------------------------------------
    # Heartbeat thread (Phase 5)
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to NAO server."""
        log.info("Heartbeat thread started (interval=%.1fs).", HEARTBEAT_INTERVAL_S)
        while not self._shutting_down:
            if self._connected.is_set():
                try:
                    self.send_fire_and_forget({"action": "heartbeat"})
                except Exception:
                    pass  # send_fire_and_forget handles errors internally
            time.sleep(HEARTBEAT_INTERVAL_S)
        log.info("Heartbeat thread stopped.")

    # ------------------------------------------------------------------
    # Auto-reconnect (Phase 5)
    # ------------------------------------------------------------------

    def _start_auto_reconnect(self) -> None:
        """Launch the auto-reconnect thread if not already running."""
        if self._shutting_down:
            return
        if (
            self._reconnect_thread is not None
            and self._reconnect_thread.is_alive()
        ):
            return  # already reconnecting
        log.info("Starting auto-reconnect thread...")
        self._reconnect_thread = threading.Thread(
            target=self._auto_reconnect_loop, daemon=True
        )
        self._reconnect_thread.start()

    def _auto_reconnect_loop(self) -> None:
        """Try to reconnect with exponential backoff."""
        delay = 1.0
        attempts = 0

        while not self._shutting_down and attempts < RECONNECT_MAX_RETRIES:
            attempts += 1
            log.info(
                "Auto-reconnect attempt %d/%d (delay=%.1fs) ...",
                attempts, RECONNECT_MAX_RETRIES, delay,
            )

            if self.connect():
                log.info("Auto-reconnect successful!")
                # Resync state from NAO
                try:
                    self.send_command({"action": "query_state"}, timeout=3.0)
                    log.info(
                        "State resynced: posture=%s",
                        self.nao_state.posture,
                    )
                except Exception:
                    log.warning("State resync failed (non-critical).")

                # Notify callback
                if self.on_reconnect is not None:
                    try:
                        self.on_reconnect()
                    except Exception:
                        log.exception("Error in on_reconnect callback")
                return

            time.sleep(delay)
            delay = min(delay * 2, RECONNECT_BACKOFF_MAX_S)

        log.error(
            "Auto-reconnect failed after %d attempts.", RECONNECT_MAX_RETRIES
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_connection_lost(self) -> None:
        """Handle unexpected connection loss (called from any thread)."""
        was_connected = self._connected.is_set()
        self._connected.clear()
        self._close_socket()
        self._fail_all_pending()

        # Phase 5: auto-reconnect on unexpected loss
        if was_connected and not self._shutting_down:
            self._start_auto_reconnect()

    def _close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _fail_all_pending(self) -> None:
        """Unblock every pending request with ``None`` data."""
        with self._pending_lock:
            for pending in self._pending.values():
                pending.ack_event.set()   # ack_data stays None
                pending.done_event.set()  # done_data stays None
            self._pending.clear()
