#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
server.py -- NAO V5 TCP Server (Python 2.7 / NAOqi).

Phase 4 architecture:
    - ChannelWorker: dedicated worker threads for LEGS, SPEECH, ARMS
    - HEAD commands run inline (immediate, never queued)
    - SYSTEM commands (query_state, stop_all) run inline
    - LEGS worker supports walk interruption (new walk cancels current)
    - stop_all: emergency stop all channels + clear all queues
    - set_walk_velocity: continuous velocity-mode walking (inline)
    - NaoStateMachine: per-channel state tracking (from Phase 2)
    - Two-phase ACK protocol: all worker commands get ack + done

Phase 5 additions:
    - Watchdog: safe-sit after WATCHDOG_TIMEOUT_S silence from client
    - FallDetector: polls ALMemory for robotHasFallen, sends event
    - Disconnect cleanup: stop motions, clear queues, reset state
    - listen(2) for reconnection queueing (FLAW 16 fix)
"""

import json
import socket
import sys
import threading
import time
import traceback

try:
    from Queue import Queue, Empty  # Python 2.7
except ImportError:
    from queue import Queue, Empty  # Python 3 (for tests)

# -- NAOqi imports (ONLY available on the NAO itself) --
from naoqi import ALProxy
import motion_library as motions

# ======================================================================
# Configuration
# ======================================================================
DEFAULT_PORT = 5555
DEFAULT_NAO_IP = "127.0.0.1"
MSG_DELIMITER = "\n"
BUFFER_SIZE = 4096
WATCHDOG_TIMEOUT_S = 10.0
FALL_CHECK_INTERVAL_S = 0.5

# Channel routing: action -> worker name (None = inline)
CHANNEL_ROUTING = {
    "walk_toward":  "legs",
    "pose":         "legs",
    "rest":         "legs",
    "wake_up":      "legs",
    "say":          "speech",
    "animated_say": "speech",
    "animate":      "arms",
    "pickup_object": "legs",    # uses crouch + stand (legs channel)
    "offer_object":  "arms",    # arm movements only
    "open_hand":     "arms",
    "close_hand":    "arms",
    "arm_carry":     "arms",
    "arm_reach_down": "arms",
    "arm_offer":     "arms",
    "arm_rest":      "arms",
}

# Actions whose response should include head joint angles
ANGLE_ACTIONS = frozenset([
    "query_state", "heartbeat", "get_posture",
])


# ======================================================================
# NAO State Machine -- Per-Channel State Tracking
# ======================================================================
class NaoStateMachine(object):
    """Tracks NAO's physical state per independent channel.

    Channels
    --------
    posture : resting / sitting / standing / fallen
    head    : idle (server-level; RPI servo manages tracking state)
    legs    : idle / walking / animating
    speech  : idle / speaking
    arms    : idle / animating
    """

    def __init__(self, initial_posture="standing"):
        self._lock = threading.Lock()
        self.posture = initial_posture
        self.head = "idle"
        self.legs = "idle"
        self.speech = "idle"
        self.arms = "idle"

    def can_execute(self, action, params=None):
        """Check if *action* is allowed given current channel states.

        Returns (bool, str_or_None) -- (allowed, rejection_reason).
        """
        with self._lock:
            # HEAD -- always allowed
            if action in ("move_head", "move_head_relative"):
                return (True, None)

            # LEGS -- walk requires standing; rejected while dancing
            if action == "walk_toward":
                if self.posture != "standing":
                    return (False, "must_stand_first")
                if self.legs == "animating":
                    return (False, "legs_busy_dancing")
                # legs=="walking" is OK -- will interrupt current walk
                return (True, None)

            # Continuous velocity walk -- same rules as walk_toward
            if action == "set_walk_velocity":
                if self.posture != "standing":
                    return (False, "must_stand_first")
                if self.legs == "animating":
                    return (False, "legs_busy_dancing")
                return (True, None)

            if action in ("stop_walk", "stop_all"):
                return (True, None)

            # SPEECH -- always allowed (queued FIFO in worker)
            if action in ("say", "animated_say"):
                return (True, None)

            # ARMS / LEGS+ARMS (animations)
            if action == "animate":
                name = params.get("name", "") if params else ""
                if name == "dance":
                    if self.posture != "standing":
                        return (False, "must_stand_first")
                    if self.legs in ("walking", "animating"):
                        return (False, "legs_busy")
                if self.arms == "animating":
                    return (False, "arms_busy")
                return (True, None)

            # POSTURE transitions
            if action in ("pose", "rest", "wake_up"):
                return (True, None)

            # System / query actions
            if action in ("query_state", "heartbeat", "get_posture"):
                return (True, None)

            return (True, None)

    def set_channel(self, channel, value):
        with self._lock:
            setattr(self, channel, value)

    def set_channels(self, updates):
        with self._lock:
            for channel, value in updates.items():
                setattr(self, channel, value)

    def snapshot(self, motion_proxy=None):
        with self._lock:
            snap = {
                "posture": self.posture,
                "head": self.head,
                "legs": self.legs,
                "speech": self.speech,
                "arms": self.arms,
            }

        if motion_proxy is not None:
            try:
                angles = motion_proxy.getAngles(
                    ["HeadYaw", "HeadPitch"], True
                )
                snap["head_yaw"] = round(angles[0], 4)
                snap["head_pitch"] = round(angles[1], 4)
            except Exception:
                pass

        return snap


# ======================================================================
# Client Connection -- Thread-Safe Socket Wrapper
# ======================================================================
class ClientConnection(object):
    def __init__(self, conn):
        self._conn = conn
        self._send_lock = threading.Lock()
        self._closed = False

    def send_response(self, response):
        with self._send_lock:
            if self._closed:
                return False
            try:
                data = (
                    json.dumps(response, ensure_ascii=True) + MSG_DELIMITER
                ).encode("utf-8")
                self._conn.sendall(data)
                return True
            except socket.error:
                self._closed = True
                return False

    def recv(self, bufsize):
        return self._conn.recv(bufsize)

    def settimeout(self, timeout):
        self._conn.settimeout(timeout)

    @property
    def is_closed(self):
        return self._closed

    def close(self):
        with self._send_lock:
            self._closed = True
        try:
            self._conn.close()
        except socket.error:
            pass


# ======================================================================
# Channel Worker -- Dedicated Thread Per Channel
# ======================================================================
class ChannelWorker(object):
    """Runs queued tasks on a dedicated thread.

    Each task is a callable.  Tasks execute sequentially within a
    channel but different channels run in parallel.
    """

    def __init__(self, name):
        self.name = name
        self._queue = Queue()
        self._thread = None
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()
        print("[worker:%s] Started." % self.name)

    def submit(self, task_fn):
        """Add a callable to the work queue."""
        self._queue.put(task_fn)

    def clear_queue(self):
        """Drain all pending tasks.  Returns count of drained tasks."""
        count = 0
        while True:
            try:
                self._queue.get_nowait()
                count += 1
            except Empty:
                break
        return count

    def _loop(self):
        while self._running:
            try:
                task_fn = self._queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                task_fn()
            except Exception:
                traceback.print_exc()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        print("[worker:%s] Stopped." % self.name)


# ======================================================================
# NAOqi Proxy Manager
# ======================================================================
class NaoProxies(object):
    def __init__(self, nao_ip):
        self.ip = nao_ip
        self._motion = None
        self._tts = None
        self._animated_tts = None
        self._posture = None
        self._leds = None
        self._memory = None

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
            self._animated_tts = ALProxy(
                "ALAnimatedSpeech", self.ip, 9559
            )
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

    @property
    def memory(self):
        if self._memory is None:
            self._memory = ALProxy("ALMemory", self.ip, 9559)
        return self._memory


# ======================================================================
# Watchdog -- Safe-Sit on Client Silence
# ======================================================================
class Watchdog(object):
    """Monitors client heartbeat.  Safe-sits NAO if no messages
    are received for WATCHDOG_TIMEOUT_S seconds.

    Does NOT stop channel workers (they must remain ready for
    a reconnecting client).
    """

    def __init__(self, timeout_s, proxies, dispatcher):
        self._timeout_s = timeout_s
        self._px = proxies
        self._dispatcher = dispatcher
        self._last_msg_time = time.time()
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._triggered = False
        self._client_connected = False

    def message_received(self):
        """Called on every incoming TCP message."""
        with self._lock:
            self._last_msg_time = time.time()
            self._triggered = False

    def set_client_connected(self, connected):
        with self._lock:
            self._client_connected = connected
            if connected:
                self._last_msg_time = time.time()
                self._triggered = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()
        print("[watchdog] Started (timeout=%ds)." % int(self._timeout_s))

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _loop(self):
        while self._running:
            time.sleep(1.0)
            with self._lock:
                if not self._client_connected or self._triggered:
                    continue
                elapsed = time.time() - self._last_msg_time

            if elapsed > self._timeout_s:
                self._safe_stop()

    def _safe_stop(self):
        """Stop all motions and sit the robot safely."""
        with self._lock:
            self._triggered = True
        print(
            "[watchdog] No messages for >%ds — safe stopping."
            % int(self._timeout_s)
        )

        # Stop all motions
        try:
            self._px.motion.stopMove()
        except Exception:
            pass
        try:
            self._px.tts.stopAll()
        except Exception:
            pass

        # Clear all worker queues (but don't stop workers)
        for w in self._dispatcher._workers.values():
            w.clear_queue()

        # Reset channel states
        self._dispatcher._state.set_channels({
            "legs": "idle",
            "speech": "idle",
            "arms": "idle",
        })

        # Safe sit
        try:
            self._px.posture.goToPosture("Sit", 0.5)
            self._dispatcher._state.set_channel("posture", "sitting")
        except Exception:
            pass

        try:
            self._px.tts.say("Lost connection. Sitting for safety.")
        except Exception:
            pass


# ======================================================================
# Fall Detector -- ALMemory Polling
# ======================================================================
class FallDetector(object):
    """Polls ALMemory for ``robotHasFallen`` every 0.5s.

    On fall:  sets posture to 'fallen', stops motions, sends an
    async ``event`` message to the connected client.
    """

    def __init__(self, proxies, state_machine):
        self._px = proxies
        self._state = state_machine
        self._thread = None
        self._running = False
        self._client_conn = None
        self._client_lock = threading.Lock()
        self._fall_handled = False

    def set_client(self, client_conn):
        """Update the current client connection reference."""
        with self._client_lock:
            self._client_conn = client_conn

    def start(self):
        try:
            # Test that ALMemory is reachable
            _ = self._px.memory
        except Exception as exc:
            print(
                "[fall-detect] ALMemory unavailable: %s — disabled." % exc
            )
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop)
        self._thread.daemon = True
        self._thread.start()
        print("[fall-detect] Started.")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self):
        while self._running:
            time.sleep(FALL_CHECK_INTERVAL_S)
            try:
                fallen = self._px.memory.getData("robotHasFallen")
                if fallen and not self._fall_handled:
                    self._on_fall()
                elif not fallen:
                    self._fall_handled = False
            except Exception:
                pass

    def _on_fall(self):
        """Handle a detected fall event."""
        self._fall_handled = True
        print("[fall-detect] FALL DETECTED!")

        self._state.set_channel("posture", "fallen")

        try:
            self._px.motion.stopMove()
        except Exception:
            pass

        # Notify client
        with self._client_lock:
            cc = self._client_conn

        if cc is not None:
            cc.send_response({
                "type": "event",
                "event": "fallen",
                "state": self._state.snapshot(),
            })


# ======================================================================
# Command Dispatcher -- Channel-Routed
# ======================================================================
class CommandDispatcher(object):
    def __init__(self, proxies, state_machine):
        self.px = proxies
        self._state = state_machine

        # -- Channel workers --
        self._legs_worker = ChannelWorker("legs")
        self._speech_worker = ChannelWorker("speech")
        self._arms_worker = ChannelWorker("arms")

        self._workers = {
            "legs":   self._legs_worker,
            "speech": self._speech_worker,
            "arms":   self._arms_worker,
        }

        # -- Handler map --
        self._handlers = {
            "say":                self._handle_say,
            "animated_say":       self._handle_animated_say,
            "move_head":          self._handle_move_head,
            "move_head_relative": self._handle_move_head_relative,
            "walk_toward":        self._handle_walk_toward,
            "set_walk_velocity":  self._handle_set_walk_velocity,
            "stop_walk":          self._handle_stop_walk,
            "stop_all":           self._handle_stop_all,
            "animate":            self._handle_animate,
            "pose":               self._handle_pose,
            "rest":               self._handle_rest,
            "wake_up":            self._handle_wake_up,
            "query_state":        self._handle_query_state,
            "heartbeat":          self._handle_heartbeat,
            "get_posture":        self._handle_get_posture,
            "pickup_object":      self._handle_pickup_object,
            "offer_object":       self._handle_offer_object,
            "open_hand":          self._handle_open_hand,
            "close_hand":         self._handle_close_hand,
            "arm_carry":          self._handle_arm_carry,
            "arm_reach_down":     self._handle_arm_reach_down,
            "arm_offer":          self._handle_arm_offer,
            "arm_rest":           self._handle_arm_rest,
        }

    @property
    def state(self):
        return self._state

    def start_workers(self):
        for w in self._workers.values():
            w.start()

    def stop_workers(self):
        for w in self._workers.values():
            w.stop()

    # ------------------------------------------------------------------
    # Dispatch entry point
    # ------------------------------------------------------------------

    def dispatch(self, command, client_conn):
        """Route a command to the correct channel.

        - HEAD commands: executed inline (immediate)
        - SYSTEM commands: executed inline
        - LEGS/SPEECH/ARMS: submitted to channel worker queue
          with ack + done two-phase protocol

        Returns the immediate response dict.
        """
        action = command.get("action")
        request_id = command.get("id")

        if action is None:
            return {"status": "error", "message": "Missing 'action' field."}

        handler = self._handlers.get(action)
        if handler is None:
            return {
                "status": "error",
                "message": "Unknown action: %s" % action,
            }

        # -- State-machine validation --
        ok, reason = self._state.can_execute(action, command)
        if not ok:
            resp = {
                "status": "rejected",
                "reason": reason,
                "action": action,
                "state": self._state.snapshot(),
            }
            if request_id is not None:
                resp["id"] = request_id
                resp["type"] = "ack"
            return resp

        # -- Route by channel --
        channel = CHANNEL_ROUTING.get(action)

        if channel is None:
            # INLINE: HEAD or SYSTEM -- execute immediately
            return self._dispatch_inline(
                handler, command, action, request_id
            )
        else:
            # WORKER: submit to channel queue
            return self._dispatch_to_worker(
                channel, handler, command, action, request_id, client_conn
            )

    # ------------------------------------------------------------------
    # Inline dispatch (HEAD / SYSTEM)
    # ------------------------------------------------------------------

    def _dispatch_inline(self, handler, command, action, request_id):
        try:
            handler(command)

            include_angles = action in ANGLE_ACTIONS
            motion_px = self.px.motion if include_angles else None

            resp = {
                "status": "ok",
                "action": action,
                "state": self._state.snapshot(motion_px),
            }
            if request_id is not None:
                resp["id"] = request_id
                resp["type"] = "done"
            return resp

        except Exception as exc:
            traceback.print_exc()
            resp = {
                "status": "error",
                "action": action,
                "message": repr(exc),
                "state": self._state.snapshot(),
            }
            if request_id is not None:
                resp["id"] = request_id
                resp["type"] = "done"
            return resp

    # ------------------------------------------------------------------
    # Worker dispatch (LEGS / SPEECH / ARMS)
    # ------------------------------------------------------------------

    def _dispatch_to_worker(
        self, channel, handler, command, action, request_id, client_conn
    ):
        worker = self._workers[channel]

        # -- LEGS interruption logic --
        if channel == "legs" and self._state.legs == "walking":
            if action in ("walk_toward", "pose", "rest"):
                # Interrupt current walk: stopMove + clear queue
                try:
                    self.px.motion.stopMove()
                except Exception:
                    pass
                cleared = worker.clear_queue()
                if cleared > 0:
                    print(
                        "[dispatch] Cleared %d queued LEGS task(s)."
                        % cleared
                    )

        # -- Pre-state update (inline, before ack) --
        self._pre_state_update(action, command)

        # -- Build task closure --
        # Python 2.7: use default-arg capture to freeze variables
        def _make_task(h, cmd, act, rid, cc):
            def task():
                status = "ok"
                error_msg = None
                try:
                    h(cmd)
                except Exception as exc:
                    traceback.print_exc()
                    status = "error"
                    error_msg = repr(exc)
                finally:
                    self._post_state_update(act, cmd)

                # Send done response (only if request had an id)
                if rid is not None and cc is not None:
                    done_resp = {
                        "id": rid,
                        "type": "done",
                        "status": status,
                        "action": act,
                        "state": self._state.snapshot(),
                    }
                    if error_msg is not None:
                        done_resp["message"] = error_msg
                    cc.send_response(done_resp)
            return task

        worker.submit(_make_task(handler, command, action, request_id, client_conn))

        # -- Immediate ack response --
        resp = {
            "status": "accepted" if request_id is not None else "ok",
            "action": action,
            "state": self._state.snapshot(),
        }
        if request_id is not None:
            resp["id"] = request_id
            resp["type"] = "ack"
        return resp

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def _pre_state_update(self, action, command):
        """Update channel states inline before ack (reader thread)."""
        if action == "walk_toward":
            self._state.set_channel("legs", "walking")

        elif action in ("say", "animated_say"):
            self._state.set_channel("speech", "speaking")

        elif action == "animate":
            name = command.get("name", "")
            if name == "dance":
                self._state.set_channels({
                    "legs": "animating",
                    "arms": "animating",
                })
            else:
                self._state.set_channel("arms", "animating")

        elif action in ("pose", "rest"):
            # Legs already stopped by interruption logic if needed
            self._state.set_channel("legs", "idle")

    def _post_state_update(self, action, command):
        """Update channel states after worker task finishes."""
        if action == "walk_toward":
            self._state.set_channel("legs", "idle")

        elif action in ("say", "animated_say"):
            self._state.set_channel("speech", "idle")

        elif action == "animate":
            name = command.get("name", "")
            if name == "dance":
                self._state.set_channels({
                    "legs": "idle",
                    "arms": "idle",
                })
            else:
                self._state.set_channel("arms", "idle")

        elif action == "pose":
            name = command.get("name", "stand")
            if name == "sit":
                self._state.set_channel("posture", "sitting")
            elif name == "stand":
                self._state.set_channel("posture", "standing")

        elif action == "rest":
            self._state.set_channels({
                "posture": "resting",
                "legs": "idle",
                "arms": "idle",
                "speech": "idle",
            })

        elif action == "wake_up":
            self._state.set_channel("posture", "standing")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Stop all workers and join threads."""
        try:
            print("[server] Stopping all motions...")
            self.px.motion.stopMove()
        except Exception:
            pass

        try:
            self.px.tts.stopAll()
        except Exception:
            pass

        for w in self._workers.values():
            w.clear_queue()
            w.stop()

        print("[server] All workers stopped.")

    # ------------------------------------------------------------------
    # Handlers -- Inline (HEAD)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Handlers -- Inline (SYSTEM)
    # ------------------------------------------------------------------

    def _handle_set_walk_velocity(self, cmd):
        """Set continuous walk velocity (inline, non-blocking)."""
        x = cmd.get("x", 0.0)
        y = cmd.get("y", 0.0)
        theta = cmd.get("theta", 0.0)
        motions.set_walk_velocity(self.px.motion, x, y, theta)
        self._state.set_channel("legs", "walking")

    def _handle_stop_walk(self, cmd):
        """Stop walking only -- head, speech, arms continue."""
        self._legs_worker.clear_queue()
        try:
            motions.stop_walk_only(self.px.motion)
        except Exception:
            try:
                self.px.motion.stopMove()
            except Exception:
                pass
        self._state.set_channel("legs", "idle")

    def _handle_stop_all(self, cmd):
        """Emergency stop -- kill ALL motions, clear ALL queues."""
        # Stop all NAOqi motions
        try:
            self.px.motion.stopMove()
        except Exception:
            pass

        # Stop speech
        try:
            self.px.tts.stopAll()
        except Exception:
            pass

        # Clear all worker queues
        for w in self._workers.values():
            w.clear_queue()

        # Reset all channel states
        self._state.set_channels({
            "legs": "idle",
            "speech": "idle",
            "arms": "idle",
        })

    def _handle_query_state(self, cmd):
        pass  # state + angles included via _dispatch_inline

    def _handle_heartbeat(self, cmd):
        pass

    def _handle_get_posture(self, cmd):
        pass

    # ------------------------------------------------------------------
    # Handlers -- LEGS worker
    # ------------------------------------------------------------------

    def _handle_walk_toward(self, cmd):
        x = cmd.get("x", 0.0)
        y = cmd.get("y", 0.0)
        theta = cmd.get("theta", 0.0)
        motions.walk_toward(self.px.motion, x, y, theta)

    def _handle_pose(self, cmd):
        name = cmd.get("name", "stand")
        pose_map = {
            "sit":   motions.go_to_sit,
            "stand": motions.go_to_stand,
        }
        func = pose_map.get(name)
        if func is None:
            raise ValueError("Unknown pose: %s" % name)
        func(self.px.motion, self.px.posture)

    def _handle_rest(self, cmd):
        motions.safe_rest(self.px.motion, self.px.posture)

    def _handle_wake_up(self, cmd):
        motions.safe_wake_up(self.px.motion, self.px.posture)

    # ------------------------------------------------------------------
    # Handlers -- SPEECH worker
    # ------------------------------------------------------------------

    def _handle_say(self, cmd):
        text = cmd.get("text", "")
        if text:
            self.px.tts.say(str(text))

    def _handle_animated_say(self, cmd):
        text = cmd.get("text", "")
        if text:
            self.px.animated_tts.say(str(text))

    # ------------------------------------------------------------------
    # Handlers -- ARMS worker
    # ------------------------------------------------------------------

    def _handle_animate(self, cmd):
        name = cmd.get("name", "")
        anim_map = {
            "wave": motions.wave_animation,
            "dance": motions.dance_animation,
        }
        func = anim_map.get(name)
        if func is None:
            raise ValueError("Unknown animation: %s" % name)
        func(self.px.motion)

    # ------------------------------------------------------------------
    # Handlers -- Object Pickup (Improvement 6)
    # ------------------------------------------------------------------

    def _handle_pickup_object(self, cmd):
        """Full pickup sequence: crouch, reach, grab, stand."""
        motions.pickup_sequence(self.px.motion, self.px.posture)

    def _handle_offer_object(self, cmd):
        """Offer object then release."""
        motions.offer_and_release(self.px.motion)

    def _handle_open_hand(self, cmd):
        hand = cmd.get("hand", "right")
        motions.open_hand(self.px.motion, hand)

    def _handle_close_hand(self, cmd):
        hand = cmd.get("hand", "right")
        motions.close_hand(self.px.motion, hand)

    def _handle_arm_carry(self, cmd):
        motions.arm_carry_position(self.px.motion)

    def _handle_arm_reach_down(self, cmd):
        motions.arm_reach_down(self.px.motion)

    def _handle_arm_offer(self, cmd):
        motions.arm_offer_position(self.px.motion)

    def _handle_arm_rest(self, cmd):
        motions.arm_rest_position(self.px.motion)


# ======================================================================
# TCP Server
# ======================================================================
class TcpServer(object):
    def __init__(self, port, dispatcher, proxies):
        self.port = port
        self.dispatcher = dispatcher
        self._proxies = proxies
        self._running = False
        self._watchdog = Watchdog(
            WATCHDOG_TIMEOUT_S, proxies, dispatcher
        )
        self._fall_detector = FallDetector(proxies, dispatcher.state)

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(2)  # FLAW 16: allow reconnecting client to queue
        srv.settimeout(1.0)
        self._running = True

        self._watchdog.start()
        self._fall_detector.start()

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

        self._watchdog.stop()
        self._fall_detector.stop()
        self.dispatcher.shutdown()
        srv.close()
        print("[server] Server shut down.")

    def _handle_client(self, conn, addr):
        client_conn = ClientConnection(conn)
        client_conn.settimeout(2.0)  # Non-blocking recv for stale detection
        self._watchdog.set_client_connected(True)
        self._fall_detector.set_client(client_conn)
        buf = b""

        while self._running and not client_conn.is_closed:
            try:
                data = client_conn.recv(BUFFER_SIZE)
            except socket.timeout:
                continue  # Check if server still running
            except socket.error:
                break

            if not data:
                break

            buf += data

            if len(buf) > BUFFER_SIZE * 16:
                print(
                    "[server] Buffer overflow from %s:%d, dropping."
                    % addr
                )
                break

            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)

                # Update watchdog on every received message
                self._watchdog.message_received()

                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError:
                    response = {
                        "status": "error",
                        "message": "Invalid UTF-8.",
                    }
                    if not client_conn.send_response(response):
                        break
                    continue

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
                    response = {
                        "status": "error",
                        "message": repr(exc),
                    }
                else:
                    print("[server] RX: %s" % json.dumps(command))
                    response = self.dispatcher.dispatch(
                        command, client_conn
                    )

                no_ack = (
                    command.get("no_ack", False)
                    if command is not None
                    else False
                )
                if not no_ack and response is not None:
                    if not client_conn.send_response(response):
                        break

        # Client disconnected — cleanup (FLAW 16 fix)
        self._on_client_disconnect(client_conn)

    def _on_client_disconnect(self, client_conn):
        """Clean up after client disconnects."""
        self._watchdog.set_client_connected(False)
        self._fall_detector.set_client(None)
        client_conn.close()

        print("[server] Cleaning up after disconnect...")

        # Stop all motions
        try:
            self._proxies.motion.stopMove()
        except Exception:
            pass
        try:
            self._proxies.tts.stopAll()
        except Exception:
            pass

        # Clear all worker queues
        for w in self.dispatcher._workers.values():
            w.clear_queue()

        # Reset channel states (keep posture as-is)
        self.dispatcher._state.set_channels({
            "legs": "idle",
            "speech": "idle",
            "arms": "idle",
        })


# ======================================================================
# Main
# ======================================================================
def main():
    port = DEFAULT_PORT
    nao_ip = DEFAULT_NAO_IP

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

    proxies = NaoProxies(nao_ip)
    state = NaoStateMachine(initial_posture="sitting")

    print("[server] Waking up NAO ...")
    try:
        motions.safe_wake_up_seated(proxies.motion, proxies.posture)
        state.set_channel("posture", "sitting")
        proxies.tts.say("Server started. Waiting for brain.")
    except Exception as exc:
        print("[server] WARNING: Could not wake up NAO: %s" % exc)

    dispatcher = CommandDispatcher(proxies, state)
    dispatcher.start_workers()

    server = TcpServer(port, dispatcher, proxies)
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
