#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
motion_library.py -- Reusable motion presets for the NAO V5.

All functions accept an ALMotion proxy and execute blocking or
non-blocking motions. Called by the TCP server dispatcher.

Python 2.7 / NAOqi compatible.
"""

import time
import math

def safe_wake_up_seated(motion, posture):
    motion.wakeUp()
    posture.goToPosture("Sit", 0.5)


def safe_wake_up(motion, posture):
    """Ensure the robot is awake and standing."""
    motion.wakeUp()
    posture.goToPosture("StandInit", 0.5)


def safe_rest(motion, posture):
    """Put the robot into a safe resting pose."""
    posture.goToPosture("Crouch", 0.4)
    motion.rest()


def move_head(motion, yaw, pitch, speed=0.15):
    """
    Set head joint angles (absolute).

    Args:
        yaw:   Head yaw   in radians (positive = turn left).
        pitch: Head pitch  in radians (positive = look down).
        speed: Fraction of max speed [0.0, 1.0].
    """
    # Clamp to safe joint limits
    yaw = max(-2.0857, min(2.0857, yaw))
    pitch = max(-0.6720, min(0.5149, pitch))

    motion.setAngles(
        ["HeadYaw", "HeadPitch"],
        [float(yaw), float(pitch)],
        float(speed),
    )


def move_head_relative(motion, d_yaw, d_pitch, speed=0.12):
    """
    Adjust head angles relative to current position.
    """
    current = motion.getAngles(["HeadYaw", "HeadPitch"], True)
    new_yaw = current[0] + d_yaw
    new_pitch = current[1] + d_pitch
    move_head(motion, new_yaw, new_pitch, speed)


def walk_toward(motion, x, y, theta):
    """
    Start walking toward a relative target.

    Args:
        x:     Forward distance (meters).
        y:     Lateral distance (meters, positive = left).
        theta: Rotation (radians, positive = counter-clockwise).
    """
    motion.moveTo(float(x), float(y), float(theta))


def stop_walk(motion):
    """Immediately stop all locomotion."""
    motion.stopMove()


def wave_animation(motion):
    """Simple wave gesture using the right arm."""
    names = [
        "RShoulderPitch", "RShoulderRoll", "RElbowYaw",
        "RElbowRoll", "RWristYaw", "RHand",
    ]
    # Raise arm
    motion.setAngles(names, [-0.5, -0.3, 1.5, 1.0, 0.0, 1.0], 0.25)
    time.sleep(0.8)

    # Wave (oscillate elbow roll)
    for _ in range(3):
        motion.setAngles(["RElbowRoll"], [0.5], 0.35)
        time.sleep(0.3)
        motion.setAngles(["RElbowRoll"], [1.2], 0.35)
        time.sleep(0.3)

    # Lower arm
    motion.setAngles(names, [1.4, 0.2, 1.2, 0.5, 0.0, 0.3], 0.2)
    time.sleep(0.6)


def dance_animation(motion):
    """A short dance sequence — alternating weight shifts and arm moves."""
    names_legs = ["LHipRoll", "RHipRoll"]
    names_arms = [
        "LShoulderPitch", "LShoulderRoll", "LElbowRoll",
        "RShoulderPitch", "RShoulderRoll", "RElbowRoll",
    ]

    for _ in range(4):
        # Shift left, right arm up
        motion.setAngles(names_legs, [0.2, 0.2], 0.3)
        motion.setAngles(names_arms, [0.0, 0.3, -0.5, -0.5, -0.5, 1.0], 0.35)
        time.sleep(0.5)

        # Shift right, left arm up
        motion.setAngles(names_legs, [-0.2, -0.2], 0.3)
        motion.setAngles(names_arms, [-0.5, 0.5, -1.0, 0.0, -0.3, 0.5], 0.35)
        time.sleep(0.5)

    # Return to stand
    motion.setAngles(names_legs, [0.0, 0.0], 0.2)
    motion.setAngles(names_arms, [1.4, 0.2, -0.5, 1.4, -0.2, 0.5], 0.2)
    time.sleep(0.5)


def go_to_sit(motion, posture):
    """Transition to sitting posture."""
    posture.goToPosture("Sit", 0.5)


def go_to_stand(motion, posture):
    """Transition to standing posture."""
    posture.goToPosture("StandInit", 0.5)
