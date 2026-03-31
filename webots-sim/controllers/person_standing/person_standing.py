"""
person_standing.py — Minimal controller that keeps the Pedestrian standing.

Without a controller, the Pedestrian's joints are free and gravity
collapses the body. This controller simply steps the simulation,
which keeps the motors in position-hold mode at their initial angles.
"""

from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

while robot.step(timestep) != -1:
    pass
