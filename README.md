## Capstone Project for MECH_ENG 449: Robotic Manipulation.

The goal of this project is to control the KUKA youBot, to grasp a cube by a 5R robot arm, at the given start
location, achieve a reasonable odometry for the mobile base, carry the cube to the desired location,
and place it down in correct orientation. And lastly, visualize the whole motion in Coppelia-Sim
software.

### Simulation: [Link](https://github.com/Sayantani-Bhattacharya/Manipulation_Capstone_Project/issues/1")

### Main Segments:
1. Planning a trajectory for the end-effector of the youBot mobile manipulator.
2. Generate the kinematics model of the youBot, consisting of the mobile base with 4 meCanum wheels and the robot arm with 5 joints.
3. Feedback and feedforward control to drive the robot to follow the desired trajectory.
4. Implementing Euler integration.
