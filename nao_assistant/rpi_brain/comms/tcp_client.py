"""
tcp_client.py — Thread-safe, auto-reconnecting JSON-over-TCP client.

Protocol:
    Each message is a UTF-8 JSON object terminated by a newline (\\n).
    The NAO server may send back a JSON acknowledgement on the same socket.

Usage:
    client = NaoTcpClient()
    client.connect()
    client.send_command({"action": "say", "text": "Hello!"})
    client.disconnect()
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any, Dict, Optional

from settings import (
    MSG_DELIMITER,
    NAO_IP,
    NAO_PORT,
    TCP_BUFFER_SIZE,
    TCP_RECONNECT_DELAY_S,
    TCP_TIMEOUT_S,
)

log = logging.getLogger(__name__)


class NaoTcpClient:
    """Manages a persistent TCP connection to the NAO robot server."""

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
        self._lock = threading.RLock()
        self._connected = threading.Event()
        self._recv_buf = b""
        

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish TCP connection. Returns True on success."""
        with self._lock:
            if self._connected.is_set():
                return True
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._timeout)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((self._host, self._port))
                self._sock = sock
                self._connected.set()
                log.info("Connected to NAO at %s:%d", self._host, self._port)
                return True
            except (OSError, socket.error) as exc:
                log.warning("Connection failed: %s", exc)
                self._cleanup_socket()
                return False

    def disconnect(self) -> None:
        """Gracefully close the socket."""
        with self._lock:
            self._cleanup_socket()
            log.info("Disconnected from NAO.")

    def ensure_connected(self, max_retries: int = 10) -> bool:
        """Reconnect if the socket is down. Returns True when ready, False if max_retries exceeded."""
        if self._connected.is_set():
            return True
        log.info("Attempting reconnection to NAO …")
        attempts = 0
        while not self.connect():
            attempts += 1
            if attempts >= max_retries:
                log.error("Max reconnection attempts (%d) exceeded", max_retries)
                return False
            log.info("Reconnect attempt %d/%d failed, retrying in %.1fs …", attempts, max_retries, TCP_RECONNECT_DELAY_S)
            time.sleep(TCP_RECONNECT_DELAY_S)
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Serialize *command* as JSON, send over TCP, and return the
        server's JSON reply (or None on failure).

        Thread-safe: multiple threads may call this concurrently.
        """
        with self._lock:
            if not self.ensure_connected():  # replaces the bare connect() call
                log.error("Cannot send command, no connection to NAO.")
                return None

            payload = (json.dumps(command, separators=(",", ":")) + MSG_DELIMITER).encode("utf-8")

            try:
                self._sock.sendall(payload)
                log.debug("TX → %s", command)
                return self._recv_reply()
            except (OSError, socket.error, socket.timeout) as exc:
                log.error("Send failed: %s — scheduling reconnect", exc)
                self._cleanup_socket()
                return None

    def send_fire_and_forget(self, command: Dict[str, Any]) -> bool:
        """Send without waiting for a reply. Returns True on success."""
        command = dict(command)  # make a copy to avoid mutating caller's dict
        command["no_ack"] = True  # signal to server that no reply is expected
        with self._lock:
            if not self.ensure_connected():
                log.error("Cannot send command, no connection to NAO.")
                return False
            assert self._sock is not None

            payload = (json.dumps(command, separators=(",", ":")) + MSG_DELIMITER).encode("utf-8")

            try:
                self._sock.sendall(payload)
                log.debug("TX (fire&forget) → %s", command)
                return True
            except (OSError, socket.error) as exc:
                log.error("Fire-and-forget failed: %s", exc)
                self._cleanup_socket()
                return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _recv_reply(self) -> Optional[Dict[str, Any]]:
        """Block until a newline-delimited JSON reply arrives."""
        assert self._sock is not None
        try:
            while b"\n" not in self._recv_buf:
                chunk = self._sock.recv(TCP_BUFFER_SIZE)
                if not chunk:
                    raise ConnectionError("Server closed connection")
                self._recv_buf += chunk
            line, self._recv_buf = self._recv_buf.split(b"\n", 1)
            reply = json.loads(line.decode("utf-8"))
            log.debug("RX ← %s", reply)
            return reply
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning("Malformed reply: %s", exc)
            return None
        except (OSError, ConnectionError) as exc:
            log.error("Receive error: %s", exc)
            self._cleanup_socket()
            return None

    def _cleanup_socket(self) -> None:
        self._connected.clear()
        self._recv_buf = b""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
