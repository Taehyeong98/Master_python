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

def in_region(start, point, radius):
    sx, sy, sz = start
    x, y, z = point

    distance = math.sqrt((x - sx)**2 + (y - sy)**2+(z - sz)**2)
    return distance <= radius

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
address = "/home/pi/Desktop/Master_python/"
robot = rtb.Robot.URDF( address+"imed_robot_update.urdf")
endEffector = 'tool'  # name of end-effector link

# Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])

# --- Initialize configuration ---
start_pose_deg = [0.2,90.0, 90.0, -90.0, 0.0, -90.0, 0.0]  # Python list or numpy array
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
#  Initial Movement  #
#--------------------#

goal_position = [1.4, 0.3117, 0.37]
INITIAL_RADIUS = 0.3

start_pose_deg = [0.2, 90.0, 90.0, -90.0, 0.0, -90.0, 0.0]  # Python list or numpy array
q_start_prismatic = start_pose_deg[0]
q_start_revolute = np.radians(start_pose_deg[1:7])  # Convert revolute joints (2–7) to radians
q0 = np.concatenate(([q_start_prismatic], q_start_revolute))  # Combine back into one joint vector

FK_start = robot.fkine(q0)
start_position = FK_start.t

initial_region = in_region(start_position, goal_pos, INITIAL_RADIUS)
# check whether the goal pose is within the initial region
inital_skip = False
if initial_region:
    initial_skip = True

start_traj = np.loadtxt('goaltrajectory.txt', delimiter=",")
start_traj_size =start_traj.size()
if start_traj_size >0 and initial_skip is False:
    reversed_traj = start_traj[::-1]


if start_traj_size == 0 and initial_skip is False:
    # --------------------#
    #        STL         #
    # --------------------#
    # LOAD STL
    mesh = trimesh.load('voxel_alpha.stl', force='mesh')

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    V = np.array(mesh.vertices, dtype=np.float64)
    F = np.array(mesh.faces, dtype=np.int32)

    print("Vertices shape:", V.shape)
    print("Faces shape:", F.shape)

    model = fcl.BVHModel()
    model.beginModel(len(V), len(F))
    model.addSubModel(V, F)
    model.endModel()

    obj = fcl.CollisionObject(model)

    # Axis-angle rotation
    axis = np.array([1.0, 0.0, 0.0])
    axis = axis / np.linalg.norm(axis)

    angle = -np.pi / 4

    Rot = R.from_rotvec(axis * angle).as_matrix()

    #  Translation

    translation = np.array([0.35, 0.35, -0.55], dtype=np.float64)

    # Create FCL Transform
    tf_mesh = fcl.Transform(Rot, translation)

    # Apply to collision object
    obj.setTransform(tf_mesh)
    # --------------------#
    #        BOX         #
    # --------------------#
    V_world = (Rot @ V.T).T + translation  # apply R*v + t

    # Compute bounds
    xMin = np.min(V_world[:, 0])
    xMax = np.max(V_world[:, 0])
    yMin = np.min(V_world[:, 1])
    yMax = np.max(V_world[:, 1])
    zMin = np.min(V_world[:, 2])
    zMax = np.max(V_world[:, 2])

    # Compute block dimensions
    blockHeight = max(0.0, zMax)
    blockLength = xMax - xMin
    blockWidth = yMax - yMin

    print("The height is", blockHeight)
    print("The length is", blockLength)
    print("The width is", blockWidth)

    # Optional margin
    # margin = 0.05
    # blockLength += margin
    # blockWidth  += margin

    # --------------------#
    #    Singularity     #
    # --------------------#
    Radius = 0.02
    s1 = fcl.Sphere(Radius)
    s1_obj = fcl.CollisionObject(s1)
    s1_rot = np.eye(3)
    s1_trans = np.array([1.56, 0.47, 0.6])
    s1_tf = fcl.Transform(s1_rot, s1_trans)
    s1_obj.setTransform(s1_tf)

    s2 = fcl.Sphere(Radius)
    s2_obj = fcl.CollisionObject(s2)
    s2_rot = np.eye(3)
    s2_trans = np.array([1.63, 0.68, 0.53])
    s2_tf = fcl.Transform(s2_rot, s2_trans)
    s2_obj.setTransform(s2_tf)

    s3 = fcl.Sphere(Radius)
    s3_obj = fcl.CollisionObject(s3)
    s3_rot = np.eye(3)
    s3_trans = np.array([0.43, 0.22, 0.53])
    s3_tf = fcl.Transform(s3_rot, s3_trans)
    s3_obj.setTransform(s3_tf)

    s4 = fcl.Sphere(Radius)
    s4_obj = fcl.CollisionObject(s4)
    s4_rot = np.eye(3)
    s4_trans = np.array([0.6, 0.14, 0.55])
    s4_tf = fcl.Transform(s4_rot, s4_trans)
    s4_obj.setTransform(s4_tf)

    # Create FCL Box
    groundBlockShape = fcl.Box(blockLength, blockWidth, blockHeight)

    groundBlock = fcl.CollisionObject(groundBlockShape)

    # Compute block center
    blockCenter = np.array([
        0.5 * (xMin + xMax),
        0.5 * (yMin + yMax),
        blockHeight / 2.0
    ], dtype=np.float64)

    # Set transform
    tf_box = fcl.Transform(np.eye(3), blockCenter)
    groundBlock.setTransform(tf_box)

    # --------------------#
    #     ROBOT URDF     #
    # --------------------#
    # Load URDF of Robot
    adress = "/home/pi/Desktop/Master_python/"
    robot = rtb.Robot.URDF(adress + "imed_robot_pi.urdf")

    print("robot urdf initialized")

    # --------------------#
    #    Path Planner    #
    # --------------------#
    point_r = 0.06
    point_shape = fcl.Sphere(point_r)
    path_point = fcl.CollisionObject(point_shape)

    # SPACE
    space = ob.SE3StateSpace()

    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, 0.0)  # x-axis
    bounds.setHigh(0, 2.27)

    bounds.setLow(1, 0.0)  # y-axis
    bounds.setHigh(1, 0.85)

    bounds.setLow(2, 0.0)  # z-axis
    bounds.setHigh(2, 0.62)  # original 0.84m

    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    # START & GOAL
    start = ob.State(space)
    goal = ob.State(space)

    start().setXYZ(start_position[0], start_position[1], start_position[2])


    goal().setXYZ(goal_position[0], goal_position[1], goal_position[2])

    # Equivalent of:
    # qx = AngleAxis(pi, X)
    # qy = AngleAxis(pi/4, Y)
    # q = qy * qx

    qx = R.from_rotvec(np.pi * np.array([1, 0, 0]))
    qy = R.from_rotvec((np.pi / 4) * np.array([0, 1, 0]))

    q = (qy * qx).as_quat()  # returns [x,y,z,w]

    start().rotation().x = q[0]
    start().rotation().y = q[1]
    start().rotation().z = q[2]
    start().rotation().w = q[3]

    goal().rotation().x = q[0]
    goal().rotation().y = q[1]
    goal().rotation().z = q[2]
    goal().rotation().w = q[3]

    # Problem definition
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    # Extract goal position due to definition of goal region
    goalPos = np.array([
        goal().getX(),
        goal().getY(),
        goal().getZ()
    ])

    startPos = np.array([
        start().getX(),
        start().getY(),
        start().getZ()
    ])

    GOAL_RADIUS = 0.3
    GOAL = 0.1
    treeReachedGoalRegion = {"value": False}  # mutable container

    space.setStateSamplerAllocator(ob.StateSamplerAllocator(samplerAllocator))

    si.setStateValidityChecker(ob.StateValidityCheckerFn(validityChecker))
    si.setup()
    start_plan = time.time()
    # Create planner
    planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(ob.timedPlannerTerminationCondition(7.0))

    if solved:

        # -----------------------------
        #       Get solution path
        # -----------------------------
        path = pdef.getSolutionPath()
        path.interpolate(150)  # densify path

        # -----------------------------
        #        Save path to CSV
        # -----------------------------
        path_csv = "path3d.csv"
        with open(path_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(path.getStateCount()):
                s = path.getState(i)
                writer.writerow([
                    s.getX(),
                    s.getY(),
                    s.getZ(),
                    s.rotation().x,
                    s.rotation().y,
                    s.rotation().z,
                    s.rotation().w
                ])
        print(f"Solution path saved to {path_csv}")

        # -----------------------------
        #     Save explored points
        # -----------------------------
        pdata = ob.PlannerData(si)
        planner.getPlannerData(pdata)

        explored_csv = "explored_points.csv"
        with open(explored_csv, "w", newline="") as f:
            writer = csv.writer(f)
            for i in range(pdata.numVertices()):
                v = pdata.getVertex(i).getState()
                writer.writerow([
                    v.getX(),
                    v.getY(),
                    v.getZ(),
                    v.rotation().x,
                    v.rotation().y,
                    v.rotation().z,
                    v.rotation().w
                ])
        print(f"Explored points saved to {explored_csv}")

    else:
        print("No solution found.")

    end_plan = time.time()
    planning_time = end_plan - start_plan
    print("Planning time:", end_plan - start_plan)

    # Load Path
    path_data = np.loadtxt('path3d.csv', delimiter=",")

    # Number of waypoints (rows)
    waypoints = path_data.shape[0]

    endEffector = 'tool'  # name of end-effector link

    # Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
    weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])

    # --- Parameters ---
    quaternions = []
    posTraj = path_data[:, 0:3]

    # --- Quaternion interpolation ---
    # MATLAB: quaternion(w,x,y,z)
    # Python: scipy Rotation.from_quat expects (x, y, z, w)

    for j in range(waypoints):
        # Convert MATLAB [w,x,y,z] to Python [x,y,z,w]
        w, x, y, z = path_data[j, 6], path_data[j, 3], path_data[j, 4], path_data[j, 5]
        quaternions.append(R.from_quat([x, y, z, w]))

    # --- Initialize configuration ---
    qStart = q0  # Python list or numpy array
    prevConfig = np.array(qStart)
    configTraj = np.zeros((waypoints, len(qStart)))
    velJointTraj = np.zeros((waypoints, len(qStart)))
    T_target = SE3()
    qIK = q0
    penalty = 0
    # --- Main loop: IK + null-space optimization + smoothing ---
    for i in range(waypoints):
        # Desired end-effector pose
        T_target.t = posTraj[i, :]
        qk = quaternions[i]
        T_target.R = qk.as_matrix()  # 3x3 rotation matrix
        # Solve IK
        configNow = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                                   q0=qIK)  # replace with your Python IK function
        # Handle IK failure
        if not configNow.success:
            print(f"Warning: IK failed at waypoint {i}, using previous config")
            qIK = prevConfig.copy()
            J = robot.jacob0(qIK, endEffector)
            w = np.sqrt(np.linalg.det(J @ J.T))
            penalty = penalty + 1
        else:
            qIK = configNow.q
        robot.fkine(qIK)
        J = robot.jacob0(qIK, endEffector)
        N = np.eye(len(qIK)) - np.linalg.pinv(J) @ J

        # damped least Square
        T_current = robot.fkine(qIK)
        dx_pos = T_target.t - T_current.t  # 3x1 vector
        R_current = T_current.R  # 3x3
        R_target = T_target.R  # 3x3

        # rotation matrix error
        dx_rot = R.from_matrix(R_target @ R_current.T).as_rotvec()

        dx = np.zeros(6)
        dx[:3] = dx_pos
        dx[3:] = dx_rot

        # lambda_sq = 0.01  # damping factor squared
        # JJT = J @ J.T
        # dq = J.T @ np.linalg.inv(JJT + lambda_sq * np.eye(JJT.shape[0])) @ dx

        dq_null = 0.1 * (N @ (singularity_gradient(qIK, endEffector)))
        qNext = (qIK + dq_null).T

        # --- Map to nearest equivalent angles & smooth ---
        qSmooth = mapToNearest(prevConfig, qNext)
        alpha = 0.5
        qFiltered = alpha * qSmooth + (1 - alpha) * prevConfig
        # Store trajectory
        configTraj[i, :] = qFiltered
        prevConfig = qFiltered

    # --- Compute delta in degrees if needed ---
    deltaJointRad = np.diff(configTraj, axis=0)
    deltaJointDeg = np.rad2deg(deltaJointRad)
    absDeltaJointDeg = np.abs(deltaJointDeg)

    # --------------------#
    #    Cubic Spline    #
    # --------------------#
    # Create array to store smooth trajectory
    t_waypoints = np.arange(waypoints)  # 0,1,2,...,waypoints-1
    num_samples = waypoints * 5  # number of points in final trajectory
    t_samples = np.linspace(0, waypoints - 1, num_samples)
    num_joints = configTraj.shape[1]
    allConfigTraj_start = np.zeros((num_samples, num_joints))

    # Interpolate each joint separately
    for j in range(num_joints):
        cs = CubicSpline(t_waypoints, configTraj[:, j], bc_type='clamped')  # clamped ensures zero slope at ends
        allConfigTraj_start[:, j] = cs(t_samples)

if initial_region is False:
    #--------------------#
    #  Linear Movement   #
    #--------------------#
    allConfigTraj =[]
    linear_time = 250
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
    if start_traj is not None:
        trajectory  = np.concatenate((reversed_traj,linear_movement,traj.q))
    else:
        trajectory  = np.concatenate((allConfigTraj_start,linear_movement,traj.q))

    np.savetxt("goaltrajectory.txt", traj.q,delimiter=",", fmt="%.3f")

    np.savetxt( "allConfigTraj.txt", trajectory, delimiter=",", fmt="%.3f")

else:
    trajectory = jtraj(start_pose_rad, goal_pose_q, 500)
    np.savetxt( "allConfigTraj.txt", trajectory.q, delimiter=",", fmt="%.3f")

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




