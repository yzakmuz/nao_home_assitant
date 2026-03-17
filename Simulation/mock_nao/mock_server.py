"""
mock_server.py -- Boots the REAL NAO TCP server with mock proxies.

Imports and uses the actual:
  - NaoStateMachine, ClientConnection, ChannelWorker
  - CommandDispatcher, TcpServer, Watchdog, FallDetector

Only NaoProxies is replaced by MockNaoProxies.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from mock_nao.mock_proxies import MockNaoProxies, set_speed_multiplier

# These imports work because bootstrap.py already:
#   - installed fake naoqi in sys.modules
#   - added nao_body/ to sys.path
from server import (
    NaoStateMachine,
    CommandDispatcher,
    TcpServer,
)


def create_mock_proxies(action_log_queue: queue.Queue,
                        speed_mult: float = 1.0) -> MockNaoProxies:
    """Create mock proxies (call before start_mock_server)."""
    set_speed_multiplier(speed_mult)
    return MockNaoProxies(action_log_queue)


def start_mock_server(
    mock_proxies: MockNaoProxies,
    port: int = 5555,
) -> None:
    """Start the NAO TCP server with mock proxies.

    This function BLOCKS (runs the server accept loop).
    Call it in a daemon thread.

    Args:
        mock_proxies: Pre-created MockNaoProxies instance.
        port: TCP listen port.
    """
    # Create state machine -- start standing (better for demo)
    state = NaoStateMachine(initial_posture="standing")

    # Log startup (mock TTS)
    print("[mock-server] Mock NAO server starting on port %d..." % port)

    # Create dispatcher with the REAL CommandDispatcher class
    dispatcher = CommandDispatcher(mock_proxies, state)
    dispatcher.start_workers()

    # Create and run TCP server (blocking)
    server = TcpServer(port, dispatcher, mock_proxies)
    try:
        server.start()
    except Exception as exc:
        print("[mock-server] Server error: %s" % exc)
    finally:
        print("[mock-server] Server stopped.")
