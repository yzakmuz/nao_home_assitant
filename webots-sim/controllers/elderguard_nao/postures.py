"""
postures.py — NAO V5 joint angle presets for standard postures.

These angles match NAOqi's ALRobotPosture values where possible.
Webots has no built-in goToPosture() so we set joints manually.

IMPORTANT — Webots physics constraints:
  - SIT posture (HipPitch=-1.53) is UNSTABLE when free-standing.
    On the real NAO, "Sit" assumes a chair is present. In Webots,
    the robot topples forward because the center of mass leaves the
    support polygon.
  - We map "sit" → CROUCH (a stable squat) in the POSTURES dict
    so the brain's "sit down" command keeps the robot balanced.
  - Key stability constraint: HipPitch + KneePitch + AnklePitch ≈ 0
    ensures the torso stays above the feet.
"""

STAND_INIT = {
    "HeadYaw": 0.0, "HeadPitch": -0.17,
    "LShoulderPitch": 1.41, "LShoulderRoll": 0.35,
    "LElbowYaw": -1.39, "LElbowRoll": -1.04, "LWristYaw": -0.01,
    "RShoulderPitch": 1.41, "RShoulderRoll": -0.35,
    "RElbowYaw": 1.39, "RElbowRoll": 1.04, "RWristYaw": 0.01,
    "LHipYawPitch": -0.17, "LHipRoll": 0.06, "LHipPitch": -0.44,
    "LKneePitch": 0.69, "LAnklePitch": -0.35, "LAnkleRoll": -0.06,
    "RHipYawPitch": -0.17, "RHipRoll": -0.06, "RHipPitch": -0.44,
    "RKneePitch": 0.69, "RAnklePitch": -0.35, "RAnkleRoll": 0.06,
    "LHand": 0.3, "RHand": 0.3,
}

# Stable crouch: gentle knee bend, center of mass stays over feet
# HipPitch + KneePitch + AnklePitch = -0.58 + 1.00 + (-0.45) = -0.03 ≈ 0  (stable)
# Deliberately shallow — only ~0.14 rad hip change from standing,
# so sit/stand transitions cause minimal momentum shift.
CROUCH = {
    "HeadYaw": 0.0, "HeadPitch": 0.0,
    "LShoulderPitch": 1.41, "LShoulderRoll": 0.20,
    "LElbowYaw": -1.39, "LElbowRoll": -0.80, "LWristYaw": -0.01,
    "RShoulderPitch": 1.41, "RShoulderRoll": -0.20,
    "RElbowYaw": 1.39, "RElbowRoll": 0.80, "RWristYaw": 0.01,
    "LHipYawPitch": -0.17, "LHipRoll": 0.06, "LHipPitch": -0.58,
    "LKneePitch": 1.00, "LAnklePitch": -0.45, "LAnkleRoll": -0.06,
    "RHipYawPitch": -0.17, "RHipRoll": -0.06, "RHipPitch": -0.58,
    "RKneePitch": 1.00, "RAnklePitch": -0.45, "RAnkleRoll": 0.06,
    "LHand": 0.3, "RHand": 0.3,
}

# Deep sit (UNSTABLE without a chair — DO NOT use when free-standing)
SIT_DEEP = {
    "HeadYaw": 0.0, "HeadPitch": 0.0,
    "LShoulderPitch": 1.41, "LShoulderRoll": 0.20,
    "LElbowYaw": -1.39, "LElbowRoll": -0.50, "LWristYaw": -0.01,
    "RShoulderPitch": 1.41, "RShoulderRoll": -0.20,
    "RElbowYaw": 1.39, "RElbowRoll": 0.50, "RWristYaw": 0.01,
    "LHipYawPitch": -0.17, "LHipRoll": 0.06, "LHipPitch": -1.53,
    "LKneePitch": 2.11, "LAnklePitch": -1.18, "LAnkleRoll": -0.06,
    "RHipYawPitch": -0.17, "RHipRoll": -0.06, "RHipPitch": -1.53,
    "RKneePitch": 2.11, "RAnklePitch": -1.18, "RAnkleRoll": 0.06,
    "LHand": 0.3, "RHand": 0.3,
}

# Arm positions for object manipulation (Improvement 6)
ARM_CARRY = {
    "RShoulderPitch": 0.8, "RShoulderRoll": -0.15,
    "RElbowYaw": 1.0, "RElbowRoll": -1.0, "RWristYaw": 0.0,
}

ARM_REACH_DOWN = {
    "RShoulderPitch": 0.8, "RShoulderRoll": -0.2,
    "RElbowYaw": 1.2, "RElbowRoll": -0.5, "RWristYaw": 0.0,
}

ARM_OFFER = {
    "RShoulderPitch": 0.3, "RShoulderRoll": -0.1,
    "RElbowYaw": 1.0, "RElbowRoll": -0.1, "RWristYaw": 0.0,
}

ARM_REST = {
    "RShoulderPitch": 1.4, "RShoulderRoll": -0.2,
    "RElbowYaw": 1.2, "RElbowRoll": 0.5, "RWristYaw": 0.0,
    "RHand": 0.3,
}

# Posture lookup — "sit" maps to CROUCH for Webots stability
POSTURES = {
    "stand": STAND_INIT,
    "Stand": STAND_INIT,
    "StandInit": STAND_INIT,
    "sit": CROUCH,
    "Sit": CROUCH,
    "crouch": CROUCH,
    "Crouch": CROUCH,
}
