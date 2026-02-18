import roboticstoolbox as rtb
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from roboticstoolbox import Robot
from spatialmath import SE3
from spatialgeometry import Mesh
from scipy.spatial.transform import Slerp
import trimesh
import pybullet as p
import pybullet_data
import time
from pybullet_utils import bullet_client
from ompl import base as ob
from ompl import geometric as og
import math
import csv
import random
import io
import sys

matlab = "/Users/kim/Documents/MATLAB/New Folder/path_planning/"
import fcl
import qpsolvers

# Load URDF of Robot
robot = rtb.Robot.URDF(matlab + "imed_robot.urdf")

endEffector = 'tool'  # name of end-effector link

# Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])

# --- Initialize configuration ---
start_pose_deg = [0.1,90.0, 90.0, -90.0, 0.0, -90.0, 0.0]  # Python list or numpy array
q_start_prismatic = start_pose_deg[0]
q_start_revolute = np.radians(start_pose_deg[1:7]) # Convert revolute joints (2–7) to radians
start_pose_rad = np.concatenate(([q_start_prismatic], q_start_revolute)) # Combine back into one joint vector

# park pose
park_pose_deg = [1.4,90.0, 90.0, -90.0, 0.0, -90.0, 0.0]
q_park_prismatic = park_pose_deg[0]
q_park_revolute = np.radians(park_pose_deg[1:7])
park_pose_rad = np.concatenate(([q_park_prismatic], q_park_revolute))

# goal pose
goal_pos = [1.623, 0.577, 0.322]
qx = R.from_rotvec(np.pi * np.array([1, 0, 0]))
qy = R.from_rotvec((np.pi/4) * np.array([0, 1, 0]))
goal_ori = (qy * qx).as_quat()  # returns [x,y,z,w]
T_target = SE3()
T_target.t = goal_pos
T_target.R = R.from_quat(goal_ori).as_matrix()
solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                           q0=park_pose_rad)  # replace with your Python IK function
print('goal_pose', solver.q)
goal_pose_q = solver.q # 1x7

#--------------------#
#  Linear Movement   #
#--------------------#
allConfigTraj =[]
linear_time = 10
linear_movement = np.zeros((linear_time,len(start_pose_rad)))
linear_movement[0:] = start_pose_rad
x_start = start_pose_rad[0]
x_goal = goal_pose_q[0]

linear_distance = x_goal - x_start


step = linear_distance / linear_time
for i in range(linear_time):
    # Prismatic joint moves linearly
    x_current = x_start + step * i

    # Revolute joints stay fixed
    q_current = np.concatenate((
        [x_current],
        start_pose_rad[1:]
    ))

    linear_movement[i] = q_current

park_pose_rad = linear_movement[linear_time-1,:]
#--------------------#
#    GOAL REGION     #
#--------------------#
goal_time_joint = 30
n_joints = len(start_pose_rad)
goal_movement_joint = np.zeros((goal_time_joint,len(start_pose_rad)))
add_time = goal_time_joint*2 # 90 deg for joint (2) and joint (3)

# twist movement joint (2) and joint (3)
ninety_rad = 90 * math.pi / 180
twist_step = ninety_rad/goal_time_joint
twist_start_2 = park_pose_rad[1]
twist_start_3 = park_pose_rad[2]
twist_movement_2 = np.zeros(goal_time_joint)
twist_movement_3 = np.zeros(goal_time_joint)
for j in range(goal_time_joint):
    twist_movement_2[j]= twist_start_2 - twist_step * (j+1)
    twist_movement_3[j] = twist_start_3 + twist_step * (j+1)

goal_time = goal_time_joint*(n_joints-1)+add_time
goal_movement = np.zeros((goal_time,len(start_pose_rad)))
goal_distance =np.zeros(len(start_pose_rad))
step_joint =np.zeros(len(start_pose_rad))
goal_start = np.zeros(len(start_pose_rad))
for i in range(n_joints):

    if i == 1:
        start = park_pose_rad[i] + ninety_rad
    elif i == 2:
        start = park_pose_rad[i] - ninety_rad
    else:
        start = park_pose_rad[i]

    goal_distance = goal_pose_q[i] - start
    step = goal_distance / goal_time_joint

    for j in range(goal_time_joint):
        goal_movement_joint[j, i] = start + step * (j+1)

#--------------------#
#     Trajectory     #
#--------------------#
# end
joint_1= goal_movement_joint[goal_time_joint-1,0]
joint_2 =goal_movement_joint[goal_time_joint-1,1]
joint_3= goal_movement_joint[goal_time_joint-1,2]
joint_4 =goal_movement_joint[goal_time_joint-1,3]
joint_5= goal_movement_joint[goal_time_joint-1,4]
joint_6 =goal_movement_joint[goal_time_joint-1,5]
joint_7= goal_movement_joint[goal_time_joint-1,6]


allConfigTraj = np.zeros((goal_time, len(start_pose_rad)))

step_1 = np.zeros((goal_time,len(start_pose_rad)))
step_2 = np.zeros((goal_time,len(start_pose_rad)))
step_3 = np.zeros((goal_time,len(start_pose_rad)))
step_4 = np.zeros((goal_time,len(start_pose_rad)))
step_5 = np.zeros((goal_time,len(start_pose_rad)))
step_6 = np.zeros((goal_time,len(start_pose_rad)))
step_7 = np.zeros((goal_time,len(start_pose_rad)))
step_8 = np.zeros((goal_time,len(start_pose_rad)))
step_9 = np.zeros((goal_time,len(start_pose_rad)))
q_current = np.zeros(len(start_pose_rad))
for i in range(goal_time_joint):
    # Combine prismatic joint + fixed revolute joints
    step_1 = np.concatenate(([goal_movement_joint[i, 0]], park_pose_rad[1:]))
    allConfigTraj[i, :] = step_1


q_current = allConfigTraj[goal_time_joint-1,:]
print("q_current:",q_current)
# twist 90 joint 2
for i in range(goal_time_joint):

    step_2 = q_current.copy()
    step_2[1] = twist_movement_2[i]  # modify joint 2 only

    allConfigTraj[goal_time_joint + i, :] = step_2

q_current = allConfigTraj[2*goal_time_joint-1,:]
print("q_current:",q_current)
# twist -90° joint 3
for i in range(goal_time_joint):
    step_3 = q_current.copy()
    step_3[2] = twist_movement_3[i]  # modify joint 3 only
    allConfigTraj[2*goal_time_joint+i, :] = step_3

q_current = allConfigTraj[3*goal_time_joint-1,:]
print("q_current:",q_current)
# movement joint 2
for i in range(goal_time_joint):
    step_4 = q_current.copy()
    step_4[1] = goal_movement_joint[i, 1]  # modify joint 3 only
    allConfigTraj[3*goal_time_joint+i, :] = step_4

q_current = allConfigTraj[4*goal_time_joint-1,:]
print("q_current:",q_current)
# movement joint 4
for i in range(goal_time_joint):
    step_6 = q_current.copy()  # copy current config
    step_6[3] = goal_movement_joint[i, 3]  # modify joint 4 only
    step_6[2] = goal_movement_joint[i, 2]
    allConfigTraj[4*goal_time_joint+i, :] = step_6

q_current = allConfigTraj[5*goal_time_joint-1,:]
print("q_current:",q_current)
"""
# movement joint 3
for i in range(goal_time_joint):
    step_5 = q_current.copy()  # copy current config
    step_5[2] = goal_movement_joint[i, 2]  # modify joint 3 only
    allConfigTraj[5*goal_time_joint+i, :] = step_5

q_current = allConfigTraj[6*goal_time_joint-1,:]
print("q_current:",q_current)
"""
# movement joint 5
for i in range(goal_time_joint):
    step_7 = q_current.copy()  # copy current config
    step_7[4] = goal_movement_joint[i, 4]  # modify joint 5 only
    allConfigTraj[5*goal_time_joint+i, :] = step_7

q_current = allConfigTraj[6*goal_time_joint-1,:]
print("q_current:",q_current)
# movement joint 6
for i in range(goal_time_joint):
    step_8 = q_current.copy()  # copy current config
    step_8[5] = goal_movement_joint[i, 5]  # modify joint 6 only
    allConfigTraj[6*goal_time_joint+i, :] = step_8

q_current = allConfigTraj[7*goal_time_joint-1,:]
print("q_current:",q_current)
# movement joint 7
for i in range(goal_time_joint):
    step_9 = q_current.copy()  # copy current config
    step_9[6] = goal_movement_joint[i, 6]  # modify joint 7 only
    allConfigTraj[7*goal_time_joint+i, :] = step_9

q_current = allConfigTraj[8*(goal_time_joint-1),:]

allConfigTraj = np.concatenate((linear_movement,allConfigTraj))
np.savetxt(matlab + "allConfigTraj.csv", allConfigTraj, delimiter=",", fmt="%.6f")




