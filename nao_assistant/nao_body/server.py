#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
server.py -- NAO V5 TCP Server (Python 2.7 / NAOqi).

Listens for newline-delimited JSON commands from the Raspberry Pi
and dispatches them to the appropriate NAOqi ALProxy calls.

Protocol:
    RX: {"action": "say", "text": "Hello"}\\n
    TX: {"status": "ok"}\\n   or   {"status": "error", "message": "..."}\\n

Supported actions:
    say             - ALTextToSpeech
    animated_say    - ALAnimatedSpeech
    move_head       - ALMotion.setAngles (absolute yaw/pitch)
    move_head_relative - ALMotion.setAngles (delta yaw/pitch)
    walk_toward     - ALMotion.moveTo
    stop_walk       - ALMotion.stopMove
    animate         - Named animation (wave, dance)
    pose            - Go to named posture (sit, stand)
    rest            - Safe rest
    wake_up         - Wake up and stand

Usage on NAO:
    $ python server.py
    $ python server.py --port 5555 --nao-ip 127.0.0.1
"""

import json
import socket
import sys
import threading
import time
import traceback

# -- NAOqi imports (ONLY available on the NAO itself) --
from naoqi import ALProxy

import motion_library as motions

# ======================================================================
# Configuration (override via CLI args if needed)
# ======================================================================
DEFAULT_PORT = 5555
DEFAULT_NAO_IP = "127.0.0.1"  # localhost on the NAO
MSG_DELIMITER = "\n"
BUFFER_SIZE = 4096


# ======================================================================
# NAOqi Proxy Manager
# ======================================================================
class NaoProxies(object):
    """Lazy-initialized container for all NAOqi service proxies."""

    def __init__(self, nao_ip):
        self.ip = nao_ip
        self._motion = None
        self._tts = None
        self._animated_tts = None
        self._posture = None
        self._leds = None

    @property
    def motion(self):
        if self._motion is None:
            self._motion = ALProxy("ALMotion", self.ip, 9559)
        return self._motion

    @property
    def tts(self):
        if self._tts is None:
            self._tts = ALProxy("ALTextToSpeech", self.ip, 9559)
        return self._tts

    @property
    def animated_tts(self):
        if self._animated_tts is None:
            self._animated_tts = ALProxy("ALAnimatedSpeech", self.ip, 9559)
        return self._animated_tts

    @property
    def posture(self):
        if self._posture is None:
            self._posture = ALProxy("ALRobotPosture", self.ip, 9559)
        return self._posture

    @property
    def leds(self):
        if self._leds is None:
            self._leds = ALProxy("ALLeds", self.ip, 9559)
        return self._leds


# ======================================================================
# Command Dispatcher
# ======================================================================
class CommandDispatcher(object):
    """Maps JSON action strings to NAOqi calls."""

    def __init__(self, proxies):
        self.px = proxies
        self._handlers = {
            "say":                self._handle_say,
            "animated_say":       self._handle_animated_say,
            "move_head":          self._handle_move_head,
            "move_head_relative": self._handle_move_head_relative,
            "walk_toward":        self._handle_walk_toward,
            "stop_walk":          self._handle_stop_walk,
            "animate":            self._handle_animate,
            "pose":               self._handle_pose,
            "rest":               self._handle_rest,
            "wake_up":            self._handle_wake_up,
        }

    def dispatch(self, command):
        """
        Execute a command dict. Returns a response dict.
        """
        action = command.get("action")
        if action is None:
            return {"status": "error", "message": "Missing 'action' field."}

        handler = self._handlers.get(action)
        if handler is None:
            return {"status": "error", "message": "Unknown action: %s" % action}

        try:
            handler(command)
            return {"status": "ok", "action": action}
        except Exception as exc:
            traceback.print_exc()
            return {"status": "error", "action": action, "message": repr(exc)}

    # -- Individual handlers ------------------------------------------

    def _handle_say(self, cmd):
        text = cmd.get("text", "")
        if text:
            motions.stop_walk(self.px.motion)
            self.px.tts.say(str(text))

    def _handle_animated_say(self, cmd):
        text = cmd.get("text", "")
        if text:
            motions.stop_walk(self.px.motion)
            self.px.animated_tts.say(str(text))

    def _handle_move_head(self, cmd):
        yaw = cmd.get("yaw", 0.0)
        pitch = cmd.get("pitch", 0.0)
        speed = cmd.get("speed", 0.15)
        motions.move_head(self.px.motion, yaw, pitch, speed)

    def _handle_move_head_relative(self, cmd):
        d_yaw = cmd.get("d_yaw", 0.0)
        d_pitch = cmd.get("d_pitch", 0.0)
        speed = cmd.get("speed", 0.12)
        motions.move_head_relative(self.px.motion, d_yaw, d_pitch, speed)

    def _handle_walk_toward(self, cmd):
        x = cmd.get("x", 0.0)
        y = cmd.get("y", 0.0)
        theta = cmd.get("theta", 0.0)
        # Run in a thread so the server isn't blocked during the walk
        t = threading.Thread(
            target=motions.walk_toward,
            args=(self.px.motion, x, y, theta),
        )
        t.daemon = True
        t.start()

    def _handle_stop_walk(self, cmd):
        motions.stop_walk(self.px.motion)

    def _handle_animate(self, cmd):
        name = cmd.get("name", "")
        anim_map = {
            "wave": motions.wave_animation,
            "dance": motions.dance_animation,
        }
        func = anim_map.get(name)
        if func is None:
            raise ValueError("Unknown animation: %s" % name)
        # Run animation in a thread so it doesn't block the socket
        t = threading.Thread(target=func, args=(self.px.motion,))
        t.daemon = True
        t.start()

    def _handle_pose(self, cmd):
        motions.stop_walk(self.px.motion)
        name = cmd.get("name", "stand")
        pose_map = {
            "sit":   motions.go_to_sit,
            "stand": motions.go_to_stand,
        }
        func = pose_map.get(name)
        if func is None:
            raise ValueError("Unknown pose: %s" % name)
        t = threading.Thread(
            target=func,
            args=(self.px.motion, self.px.posture),
        )
        t.daemon = True
        t.start()

    def _handle_rest(self, cmd):
        motions.stop_walk(self.px.motion)
        t = threading.Thread(
            target=motions.safe_rest,
            args=(self.px.motion, self.px.posture),
        )
        t.daemon = True
        t.start()

    def _handle_wake_up(self, cmd):
        t = threading.Thread(
            target=motions.safe_wake_up,
            args=(self.px.motion, self.px.posture),
        )
        t.daemon = True
        t.start()


# ======================================================================
# TCP Server
# ======================================================================
class TcpServer(object):
    """Single-client TCP server with newline-delimited JSON protocol."""

    def __init__(self, port, dispatcher):
        self.port = port
        self.dispatcher = dispatcher
        self._running = False

    def start(self):
        """Bind, listen, and accept one client at a time (blocking)."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(1)
        srv.settimeout(1.0)
        self._running = True

        print("[server] Listening on 0.0.0.0:%d" % self.port)

        while self._running:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break

            print("[server] Client connected: %s:%d" % addr)
            self._handle_client(conn, addr)
            print("[server] Client disconnected: %s:%d" % addr)

        srv.close()
        print("[server] Server shut down.")

    def _handle_client(self, conn, addr):
        """Read newline-delimited JSON messages from a single client."""
        conn.settimeout(None)  # blocking reads
        buf = b""
        disconnected = False

        while self._running and not disconnected:
            try:
                data = conn.recv(BUFFER_SIZE)
            except socket.error:
                break

            if not data:
                break

            buf += data

            if len(buf) > BUFFER_SIZE * 16:  # e.g. 64KB max
                print("[server] Buffer overflow from %s:%d, dropping connection." % addr)
                break

            # Process all complete messages in the buffer
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    response = {"status": "error", "message": "Invalid UTF-8."}
                    resp_bytes = (
                    json.dumps(response, ensure_ascii=True) + MSG_DELIMITER
                    ).encode("utf-8")
                    try:
                        conn.sendall(resp_bytes)
                    except socket.error:
                        disconnected = True
                    break
                line = line.strip()
                if not line:
                    continue

                command = None
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        raise ValueError("Expected JSON object")
                    command = parsed
                except ValueError as exc:
                    response = {"status": "error", "message": repr(exc)}

                else:
                    print("[server] RX: %s" % json.dumps(command))
                    response = self.dispatcher.dispatch(command)

                # Send response
                no_ack = command.get("no_ack", False) if command is not None else False
                if not no_ack:
                    resp_bytes = (
                    json.dumps(response, ensure_ascii=True) + MSG_DELIMITER
                    ).encode("utf-8")
                    try:
                        conn.sendall(resp_bytes)
                    except socket.error:
                        disconnected = True
                        break

        try:
            conn.close()
        except socket.error:
            pass


# ======================================================================
# Main
# ======================================================================
def main():
    port = DEFAULT_PORT
    nao_ip = DEFAULT_NAO_IP

    # Simple CLI arg parsing (no argparse to keep Py2.7 minimal)
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--nao-ip" and i + 1 < len(args):
            nao_ip = args[i + 1]
            i += 2
        else:
            i += 1

    print("[server] NAO IP: %s | Listen port: %d" % (nao_ip, port))

    # Initialize NAOqi proxies
    proxies = NaoProxies(nao_ip)

    # Wake up the robot
    print("[server] Waking up NAO ...")
    try:
        motions.safe_wake_up_seated(proxies.motion, proxies.posture)
        proxies.tts.say("Server started. Waiting for brain.")
    except Exception as exc:
        print("[server] WARNING: Could not wake up NAO: %s" % exc)

    # Start TCP server (blocks)
    dispatcher = CommandDispatcher(proxies)
    server = TcpServer(port, dispatcher)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[server] Interrupted.")
    finally:
        print("[server] Resting NAO ...")
        try:
            current_pos = proxies.posture.getPosture()
            if current_pos == "Sit":
                proxies.motion.rest()
            else:  
                motions.safe_rest(proxies.motion, proxies.posture)
        except Exception:
            pass


if __name__ == "__main__":
    main()
