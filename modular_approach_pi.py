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
from roboticstoolbox import jtraj
import fcl
import time

start_time = time.time()
#--------------------#
#      Pybullet      #
#--------------------#
# Connect to DIRECT
bc = bullet_client.BulletClient(connection_mode=p.DIRECT)

# Optional: set search path for meshes
bc.setAdditionalSearchPath(pybullet_data.getDataPath())

robotId=bc.loadURDF("imed_robot_pi.urdf")
position = [0.35, 0.35, -0.55]        # must be length 3
# Axis-angle
angle = -np.pi/4              # rotation angle in radians
axis = np.array([1, 0, 0])    # rotation axis (must be normalized)
axis = axis / np.linalg.norm(axis)  # just to be safe

# Convert to quaternion
r = R.from_rotvec(axis * angle)  # axis-angle → rotation vector
orientation = r.as_quat()
scale = [1,1,1]  # must be length 3
# Create collision shape and visual shape
collisionShapeId = bc.createCollisionShape(shapeType=bc.GEOM_MESH,
                                          fileName='voxel_alpha.stl',
                                          meshScale=scale)
visualShapeId = bc.createVisualShape(shapeType=bc.GEOM_MESH,
                                    fileName='voxel_alpha.stl',
                                    meshScale=scale)

# Create a multi-body using the mesh
obstacleId = bc.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=position,
                                baseOrientation=orientation)





# Load URDF of Robot
adress = "/home/pi/Desktop/Master_python/"
robot = rtb.Robot.URDF( adress+"imed_robot_pi.urdf")
endEffector = 'tool'  # name of end-effector link

# Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])

# --- Initialize configuration ---
start_pose_deg = [0.3,90.0, 90.0, -90.0, 0.0, -90.0, 0.0]  # Python list or numpy array
q_start_prismatic = start_pose_deg[0]
q_start_revolute = np.radians(start_pose_deg[1:7]) # Convert revolute joints (2–7) to radians
start_pose_rad = np.concatenate(([q_start_prismatic], q_start_revolute)) # Combine back into one joint vector

# park pose
park_pose_deg = [0.7,90.0, 90.0, -90.0, 0.0, -90.0, 0.0]
q_park_prismatic = park_pose_deg[0]
q_park_revolute = np.radians(park_pose_deg[1:7])
park_pose_rad = np.concatenate(([q_park_prismatic], q_park_revolute))

# goal pose
goal_pos = [1.4, 0.3117, 0.37]
qx = R.from_rotvec(np.pi * np.array([1, 0, 0]))
qy = R.from_rotvec((np.pi/4) * np.array([0, 1, 0]))
goal_ori = (qy * qx).as_quat()  # returns [x,y,z,w]
T_target = SE3()
T_target.t = goal_pos
T_target.R = R.from_quat(goal_ori).as_matrix()
solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                           q0=park_pose_rad)  # replace with your Python IK function
while (solver.success== False):
    print("IK solver failed, try again")
    print(park_pose_rad)
    solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                            q0=park_pose_rad)  # replace with your Python
q_sol = solver.q
print(q_sol)
goal_pose_q = q_sol # 1x7

#--------------------#
#  Linear Movement   #
#--------------------#
allConfigTraj =[]
linear_time = 30
linear_movement = np.zeros((linear_time,len(start_pose_rad)))
linear_movement[0:] = start_pose_rad
x_start = start_pose_rad[0]
x_goal = park_pose_rad[0]

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

traj = jtraj(park_pose_rad, goal_pose_q, 500)

trajectory  = np.concatenate((linear_movement,traj.q))

np.savetxt( "allConfigTraj.csv", trajectory, delimiter=",", fmt="%.6f")

# --------------------#
#  COLLISION CHECK   #
# --------------------#
print("Waypoints:", len(trajectory))
# Get joint indices
num_joints = p.getNumJoints(robotId)

# Only actuated joints
joint_indices = [i for i in range(num_joints) if p.getJointInfo(robotId, i)[2] != p.JOINT_FIXED]
real_self_collision = False

for waypoint_idx, waypoint_joints in enumerate(trajectory):
    for joint_index, joint_value in zip(joint_indices, waypoint_joints):
        p.resetJointState(robotId, joint_index, joint_value)

    bc.stepSimulation()

    contacts_env = bc.getClosestPoints(robotId, obstacleId, distance=0.001)
    if contacts_env:
        print(f"Environment Collision at waypoint {waypoint_idx}")

    contacts_self_raw = bc.getClosestPoints(robotId, robotId, distance=0.001)
    unique_pairs = set()
    for c in contacts_self_raw:
        linkA, linkB, distance = c[3], c[4], c[8]

        if distance >= -1e-5 or linkA == linkB:
            continue

        parentA = p.getJointInfo(robotId, linkA)[16] if linkA != -1 else -1
        parentB = p.getJointInfo(robotId, linkB)[16] if linkB != -1 else -1

        if parentA == linkB or parentB == linkA:
            continue

        pair = tuple(sorted((linkA, linkB)))
        if pair not in unique_pairs:
            unique_pairs.add(pair)
            print(f"Self collision between link {linkA+1} and link {linkB+1} at waypoint {waypoint_idx}")
            real_self_collision = True
            
end = time.time()
print("Execution time:", end - start_time, "seconds")
execution_time = end - start_time
with open("execution_time_modular.txt", "a") as file:
    file.write(f"{execution_time}\n")

end_pose = trajectory[-1,:]
FK = robot.fkine(end_pose)
end_position = FK.t
error = np.linalg.norm(goal_pos-end_position)

with open("error_position_modular.txt", "a") as file:
    file.write(f"{error}\n")

# Convert measured rotation to quaternion
R_meas = FK.R
q_meas = R.from_matrix(R_meas).as_quat()  # [x, y, z, w]

# Quaternion difference
# q_err = q_ref^{-1} * q_meas
r_ref_inv = R.from_quat(goal_ori).inv()
r_err = r_ref_inv * R.from_quat(q_meas)

# Orientation error as angle (radians)
theta_err = r_err.magnitude()
theta_err_deg = np.degrees(theta_err)

print("Orientation error (deg):", theta_err_deg)

with open("error_orientation_modular.txt", "a") as file:
    file.write(f"{theta_err_deg}\n")
"""
#--------------------#
#    GOAL REGION     #
#--------------------#
goal_time_joint = 10
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

goal_time = goal_time_joint*n_joints
goal_movement = np.zeros((goal_time,len(start_pose_rad)))
goal_distance =np.zeros(len(start_pose_rad))
step_joint =np.zeros(len(start_pose_rad))
goal_start = np.zeros(len(start_pose_rad))
for i in range(n_joints):

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
# movement joint 1
for i in range(goal_time_joint):
    # Combine prismatic joint + fixed revolute joints
    step_1 = np.concatenate(([goal_movement_joint[i, 0]], park_pose_rad[1:]))
    allConfigTraj[i, :] = step_1


q_current = allConfigTraj[goal_time_joint-1,:]
print("q_current:",q_current)


# movement joint 2
for i in range(goal_time_joint):
    step_4 = q_current.copy()
    step_4[1] = goal_movement_joint[i, 1]  # modify joint 3 only
    allConfigTraj[goal_time_joint+i, :] = step_4

q_current = allConfigTraj[2*goal_time_joint-1,:]
print("q_current:",q_current)
# movement joint 3
for i in range(goal_time_joint):
    step_6 = q_current.copy()  # copy current config
    step_6[2] = goal_movement_joint[i, 2]
    allConfigTraj[2*goal_time_joint+i, :] = step_6

q_current = allConfigTraj[3*goal_time_joint-1,:]
print("q_current:",q_current)

# movement joint 4
for i in range(goal_time_joint):
    step_5 = q_current.copy()  # copy current config
    step_5[3] = goal_movement_joint[i, 3]  # modify joint 3 only
    allConfigTraj[3*goal_time_joint+i, :] = step_5

q_current = allConfigTraj[4*goal_time_joint-1,:]
print("q_current:",q_current)

# movement joint 5
for i in range(goal_time_joint):
    step_7 = q_current.copy()  # copy current config
    step_7[4] = goal_movement_joint[i, 4]  # modify joint 5 only
    allConfigTraj[4*goal_time_joint+i, :] = step_7

q_current = allConfigTraj[5*goal_time_joint-1,:]
print("q_current:",q_current)
# movement joint 6
for i in range(goal_time_joint):
    step_8 = q_current.copy()  # copy current config
    step_8[5] = goal_movement_joint[i, 5]  # modify joint 6 only
    allConfigTraj[5*goal_time_joint+i, :] = step_8

q_current = allConfigTraj[6*goal_time_joint-1,:]
print("q_current:",q_current)
# movement joint 7
for i in range(goal_time_joint):
    step_9 = q_current.copy()  # copy current config
    step_9[6] = goal_movement_joint[i, 6]  # modify joint 7 only
    allConfigTraj[6*goal_time_joint+i, :] = step_9

q_current = allConfigTraj[7*(goal_time_joint-1),:]

trajectory  = np.concatenate((linear_movement,allConfigTraj))
np.savetxt( "allConfigTraj.csv", trajectory, delimiter=",", fmt="%.6f")
#---------------------#
# SAVE FOR EXPERIMENT #
#---------------------#
trajectory[:,0] = trajectory[:,0]*1000
trajectory[:,1:7] = trajectory[:,1:7]*180/np.pi
np.savetxt( "modular_normal.txt", trajectory, delimiter=" ", fmt="%.2f")
"""



