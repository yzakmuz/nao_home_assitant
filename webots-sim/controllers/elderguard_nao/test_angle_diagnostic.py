#!/usr/bin/env python3
"""
Diagnostic: determine the correct angle convention for this Webots world.
Queries NAO position/orientation and person position, then tests formulas.
"""
import socket, json, time, math

HOST = "127.0.0.1"
PORT = 5555

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
s.settimeout(5.0)
buf = ""
counter = 0

def send(cmd):
    global buf, counter
    counter += 1
    cmd["id"] = "diag_%d" % counter
    s.send((json.dumps(cmd) + "\n").encode())
    while "\n" not in buf:
        buf += s.recv(4096).decode()
    line, buf = buf.split("\n", 1)
    return json.loads(line)

print("=" * 60)
print("  Angle Convention Diagnostic")
print("=" * 60)

# Query person position (includes NAO position + orientation)
resp = send({"action": "get_person_position"})
person = resp["person_position"]
nao_pos = resp["nao_position"]
nao_orient = resp["nao_orientation"]

print()
print("  Person:   x=%.3f  y=%.3f  z=%.3f" % (person["x"], person["y"], person["z"]))
print("  NAO pos:  x=%.3f  y=%.3f  z=%.3f" % (nao_pos["x"], nao_pos["y"], nao_pos["z"]))
print("  NAO RPY:  roll=%.4f  pitch=%.4f  yaw=%.4f" % (
    nao_orient["roll"], nao_orient["pitch"], nao_orient["yaw"]))
print("  NAO yaw:  %.4f rad = %.1f deg" % (nao_orient["yaw"], math.degrees(nao_orient["yaw"])))

dx = person["x"] - nao_pos["x"]
dy = person["y"] - nao_pos["y"]
dz = person["z"] - nao_pos["z"]
yaw = nao_orient["yaw"]

print()
print("  Delta: dx=%.3f  dy=%.3f  dz=%.3f" % (dx, dy, dz))
print("  Horiz dist (XY): %.3f" % math.sqrt(dx*dx + dy*dy))
print("  Horiz dist (XZ): %.3f" % math.sqrt(dx*dx + dz*dz))
print()

# Test all formulas
formulas = [
    ("atan2(dx, dy)", math.atan2(dx, dy)),
    ("atan2(dy, dx)", math.atan2(dy, dx)),
    ("atan2(-dx, dy)", math.atan2(-dx, dy)),
    ("atan2(dx, -dy)", math.atan2(dx, -dy)),
    ("atan2(-dy, dx)", math.atan2(-dy, dx)),
    ("atan2(dy, -dx)", math.atan2(dy, -dx)),
]

print("  Formula tests (yaw=%.4f):" % yaw)
print("  %-20s  world_angle   rel_angle   interpretation" % "formula")
print("  " + "-" * 70)

for name, world_angle in formulas:
    rel = world_angle - yaw
    rel = math.atan2(math.sin(rel), math.cos(rel))
    deg = math.degrees(rel)

    if abs(deg) < 20:
        interp = "NEARLY AHEAD (%.0f deg) <-- likely correct" % deg
    elif abs(deg) < 45:
        interp = "%.0f deg %s" % (abs(deg), "left" if deg > 0 else "right")
    elif abs(deg) < 135:
        interp = "%.0f deg %s (sideways!)" % (abs(deg), "left" if deg > 0 else "right")
    else:
        interp = "%.0f deg (BEHIND!)" % abs(deg)

    print("  %-20s  %+8.4f      %+8.4f    %s" % (name, world_angle, rel, interp))

print()

# Also test what forward direction yaw implies
cos_y = math.cos(yaw)
sin_y = math.sin(yaw)
print("  Forward vector at yaw=%.2f:" % yaw)
print("    If yaw=0 -> +X: forward = (cos(yaw), sin(yaw)) = (%.3f, %.3f)" % (cos_y, sin_y))
print("    If yaw=0 -> +Y: forward = (-sin(yaw), cos(yaw)) = (%.3f, %.3f)" % (-sin_y, cos_y))
print()
print("  Direction to person (normalized): (%.3f, %.3f)" % (
    dx / math.sqrt(dx*dx + dy*dy), dy / math.sqrt(dx*dx + dy*dy)))

s.close()
print()
print("  The correct formula is the one showing 'NEARLY AHEAD'")
print("  (person should be roughly in front of NAO based on the scene)")
print("=" * 60)
