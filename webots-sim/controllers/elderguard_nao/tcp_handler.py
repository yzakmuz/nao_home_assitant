"""
tcp_handler.py — Non-blocking TCP server for ElderGuard brain communication.

Implements the same JSON-over-TCP protocol as the real NAO server (server.py):
  - Newline-delimited JSON messages
  - Single client connection
  - Non-blocking socket I/O (polled from Webots robot.step loop)
  - Responses include channel state snapshot

Design:
  Uses select() with timeout=0 for non-blocking checks. No background threads.
  This avoids the Webots GIL limitation where threads run ~100x slower.
  All socket I/O happens inside poll_commands() which is called every robot.step.
"""

import json
import select
import socket
import time


class TcpHandler:
    """Non-blocking TCP server for receiving brain commands.

    Usage in the Webots main loop::

        tcp = TcpHandler(port=5555)
        tcp.start()

        while robot.step(timestep) != -1:
            commands = tcp.poll_commands()
            for cmd in commands:
                handle(cmd)
                tcp.send_response(cmd, {"status": "ok"})
    """

    def __init__(self, port=5555):
        self._port = port
        self._server_sock = None
        self._client_sock = None
        self._client_addr = None
        self._recv_buffer = ""
        self._running = False

    def start(self):
        """Bind and listen on the TCP port (non-blocking)."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.setblocking(False)

        try:
            self._server_sock.bind(("0.0.0.0", self._port))
            self._server_sock.listen(2)
            self._running = True
            print("[TCP] Server listening on port %d" % self._port)
        except OSError as e:
            print("[TCP] ERROR: Could not bind port %d: %s" % (self._port, e))
            self._server_sock = None
            self._running = False

    def stop(self):
        """Close all sockets."""
        self._running = False
        if self._client_sock:
            try:
                self._client_sock.close()
            except Exception:
                pass
            self._client_sock = None
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
            self._server_sock = None
        print("[TCP] Server stopped.")

    @property
    def is_connected(self):
        """True if a brain client is connected."""
        return self._client_sock is not None

    def poll_commands(self):
        """Non-blocking: accept new clients and read any pending commands.

        Returns a list of parsed JSON command dicts (may be empty).
        Call this every robot.step() iteration.
        """
        if not self._running:
            return []

        # Try to accept a new client (non-blocking)
        self._try_accept()

        # Read available data from client (non-blocking)
        if self._client_sock is None:
            return []

        return self._read_commands()

    def send_response(self, response_dict):
        """Send a JSON response to the connected client."""
        if self._client_sock is None:
            return

        try:
            data = json.dumps(response_dict) + "\n"
            self._client_sock.sendall(data.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print("[TCP] Send failed (client disconnected): %s" % e)
            self._disconnect_client()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_accept(self):
        """Non-blocking accept of a new client connection."""
        if self._server_sock is None:
            return

        try:
            readable, _, _ = select.select([self._server_sock], [], [], 0)
            if readable:
                conn, addr = self._server_sock.accept()
                conn.setblocking(False)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                # Close any existing client
                if self._client_sock:
                    print("[TCP] Replacing existing client with new connection.")
                    self._disconnect_client()

                self._client_sock = conn
                self._client_addr = addr
                self._recv_buffer = ""
                print("[TCP] Client connected from %s:%d" % addr)
        except Exception:
            pass

    def _read_commands(self):
        """Non-blocking read from client. Returns list of parsed commands."""
        commands = []

        try:
            readable, _, _ = select.select([self._client_sock], [], [], 0)
            if not readable:
                return commands

            data = self._client_sock.recv(4096)
            if not data:
                # Client disconnected cleanly
                print("[TCP] Client disconnected.")
                self._disconnect_client()
                return commands

            self._recv_buffer += data.decode("utf-8", errors="replace")

            # Parse newline-delimited JSON messages
            while "\n" in self._recv_buffer:
                line, self._recv_buffer = self._recv_buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    cmd = json.loads(line)
                    commands.append(cmd)
                except json.JSONDecodeError as e:
                    print("[TCP] Invalid JSON: %s — %s" % (line[:80], e))

        except (ConnectionResetError, ConnectionAbortedError, OSError) as e:
            print("[TCP] Client connection lost: %s" % e)
            self._disconnect_client()

        return commands

    def _disconnect_client(self):
        """Clean up client connection."""
        if self._client_sock:
            try:
                self._client_sock.close()
            except Exception:
                pass
        self._client_sock = None
        self._client_addr = None
        self._recv_buffer = ""
