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

# FCL
import fcl

matlab = "/Users/kim/Documents/MATLAB/New Folder/path_planning/"

def mapToNearest( q_prev, q_new):
    """
    Map q_new to the nearest equivalent configuration relative to q_prev.

    Parameters:
        q_prev : array-like, previous joint configuration
        q_new : array-like, new joint configuration

    Returns:
        q_mapped : np.ndarray, new configuration mapped closest to q_prev
    """
    q_prev = np.array(q_prev).flatten()
    q_new = np.array(q_new).flatten()
    q_mapped = q_prev.copy()

    for i in range(len(q_new)):
        if i == 0:
            # linear joints: no wrapping
            q_mapped[i] = q_new[i]


        else:
            # angle wrapping into [-pi, pi)
            dq = q_new[i] - q_prev[i]
            dq = (dq + np.pi) % (2 * np.pi) - np.pi
            q_mapped[i] = q_prev[i] + dq

    return q_mapped



def singularity_gradient( q, body_name):
    """
    Compute the gradient of the manipulability measure to avoid singularities.

    Parameters:
        robot : roboticstoolbox Robot object
        q : array-like, joint configuration
        body_name : str, name of the end-effector link to compute Jacobian

    Returns:
        grad : np.ndarray, gradient of manipulability (size: robot.n)
    """
    q = np.array(q).flatten()
    n_joints = robot.n
    eps = 1e-6  # small increment for finite differences

    # Compute Jacobian at current configuration
    J = robot.jacob0(q, end=body_name)  # shape: 6 x n
    w = np.sqrt(np.linalg.det(J @ J.T))

    grad = np.zeros(n_joints)

    for k in range(n_joints):
        dq = np.zeros_like(q)
        dq[k] = eps
        J2 = robot.jacob0(q + dq, end=body_name)
        w2 = np.sqrt(np.linalg.det(J2 @ J2.T))
        grad[k] = (w2 - w) / eps

    return grad

def validityChecker(state):

    # Extract SE3 state
    s = state  # already SE3 state in Python
    pos = np.array([s.getX(), s.getY(), s.getZ()])
    INITIAL_RADIUS = 0.3
    GOAL = 0.1
    # ------------- 2️⃣ Move robot (sphere) -------------
    tf = fcl.Transform(np.eye(3), pos)
    path_point.setTransform(tf)

    inGoalRegion = np.linalg.norm(pos - goalPos) < GOAL_RADIUS
    InitialRegion = np.linalg.norm(pos - startPos) < INITIAL_RADIUS
    if np.linalg.norm(pos - goalPos) < GOAL:
        return True

    if InitialRegion:
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        fcl.collide(path_point, obj, req, res)

        if res.is_collision:
            return False
    # ------------- 3️⃣ Ground collision (always obstacle) -------------
    if not inGoalRegion:
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        fcl.collide(path_point, groundBlock, req, res)

        if res.is_collision:
            return False

    # ------------- 4️⃣ Patient collision (only inside goal region) -------------
    if inGoalRegion:
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        fcl.collide(path_point, obj, req, res)

        if res.is_collision:
            return False

    return True

class GoalRegionSampler(ob.StateSampler):

    def __init__(self, space, goal, radius, reachedFlag):
        super(GoalRegionSampler, self).__init__(space)
        self.space = space
        self.goal = np.array(goal)
        self.radius = radius
        self.reachedFlag = reachedFlag  # use mutable container like dict

    # ----------------------------------
    # sampleUniform
    # ----------------------------------
    def sampleUniform(self, state):

        s = state  # already SE3 state in Python

        if self.reachedFlag["value"]:

            # Sample inside goal region
            x = random.uniform(self.goal[0] - self.radius,
                               self.goal[0] + self.radius)
            y = random.uniform(self.goal[1] - self.radius,
                               self.goal[1] + self.radius)
            z = random.uniform(self.goal[2] - self.radius,
                               self.goal[2] + self.radius)

            s.setXYZ(x, y, z)

            # Keep fixed orientation (identity quaternion)
            s.rotation().x = 0.0
            s.rotation().y = 0.0
            s.rotation().z = 0.0
            s.rotation().w = 1.0

        else:
            # Use default sampler
            default_sampler = self.space.allocDefaultStateSampler()
            default_sampler.sampleUniform(state)

    # ----------------------------------
    # Optional overrides
    # ----------------------------------
    def sampleUniformNear(self, state, near, distance):
        pass

    def sampleGaussian(self, state, mean, stdDev):
        pass

def samplerAllocator(space):
    return GoalRegionSampler(space, goalPos, GOAL, treeReachedGoalRegion)

#--------------------#
#        STL         #
#--------------------#
# LOAD STL
mesh = trimesh.load(matlab+'voxel_alpha.stl')
print('finish!')

# Setting correct pose of STL
#mesh.vertices *= 0.001
"""
T = np.eye(4)
T[:3,:3] = R.from_rotvec(-np.pi/4 * np.array([1,0,0])).as_matrix()
T[:3,3] = [0.35, 0.35, -0.55]

mesh.apply_transform(T)
"""
V = np.array(mesh.vertices, dtype=np.float64)
F = np.array(mesh.faces, dtype=np.int32)

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
#--------------------#
#        BOX         #
#--------------------#
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
blockWidth  = yMax - yMin

print("The height is", blockHeight)
print("The length is", blockLength)
print("The width is", blockWidth)

# Optional margin
# margin = 0.05
# blockLength += margin
# blockWidth  += margin

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

#--------------------#
#    Path Planner    #
#--------------------#
point_r = 0.1
point_shape = fcl.Sphere(point_r)
path_point = fcl.CollisionObject(point_shape)

# SPACE
space = ob.SE3StateSpace()

bounds = ob.RealVectorBounds(3)
bounds.setLow(0, 0.0) # x-axis
bounds.setHigh(0, 2.27)

bounds.setLow(1, 0.0) # y-axis
bounds.setHigh(1, 0.85)

bounds.setLow(2, 0.0) # z-axis
bounds.setHigh(2, 0.62) # original 0.84m

space.setBounds(bounds)

si = ob.SpaceInformation(space)

# START & GOAL
start = ob.State(space)
goal = ob.State(space)

start().setXYZ(0.1,0.1,0.5)
start().rotation().setIdentity()
goal().setXYZ(1.623, 0.577, 0.322)

# Equivalent of:
# qx = AngleAxis(pi, X)
# qy = AngleAxis(pi/4, Y)
# q = qy * qx

qx = R.from_rotvec(np.pi * np.array([1, 0, 0]))
qy = R.from_rotvec((np.pi/4) * np.array([0, 1, 0]))

q = (qy * qx).as_quat()  # returns [x,y,z,w]

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

# Create planner
planner = og.RRTstar(si)
planner.setProblemDefinition(pdef)
planner.setup()

solved = planner.solve(ob.timedPlannerTerminationCondition(20.0))

if solved:

    # -----------------------------
    #       Get solution path
    # -----------------------------
    path = pdef.getSolutionPath()
    path.interpolate(300)  # densify path

    # -----------------------------
    #        Save path to CSV
    # -----------------------------
    path_csv = matlab + "path3d.csv"
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

    explored_csv = matlab + "explored_points.csv"
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

#--------------------#
#      Pybullet      #
#--------------------#
# Connect in DIRECT mode (no GUI)
#pybul_start = p.connect(p.GUI)
physicsClient = p.connect(p.DIRECT)
# Optional: set search path for meshes
p.setAdditionalSearchPath(pybullet_data.getDataPath())

robotId=p.loadURDF(matlab+"imed_robot.urdf",
                   flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                   )
position = [0.35, 0.35, -0.55]        # must be length 3

# Convert to quaternion
r = R.from_rotvec(axis * angle)  # axis-angle → rotation vector
orientation = r.as_quat()
scale = [0.001,0.001,0.001]  # must be length 3
# Create collision shape and visual shape
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName='voxel_alpha.stl',
                                          meshScale=scale)
visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName='voxel_alpha.stl',
                                    meshScale=scale)

# Create a multi-body using the mesh
obstacleId = p.createMultiBody(baseMass=0,
                                baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId,
                                basePosition=position,
                                baseOrientation=orientation
                               )



# Load URDF of Robot
robot = rtb.Robot.URDF(matlab + "imed_robot.urdf")

# Load Path
path_data = np.loadtxt(matlab + 'path3d.csv', delimiter=",")

# Number of waypoints (rows)
waypoints = path_data.shape[0]

print("Path data:\n", path_data)
print(waypoints)

endEffector = 'tool'  # name of end-effector link

# Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])
# Initial joint configuration guess
q0 = np.zeros(robot.n)  # or your previous configuration

# --- Parameters ---
T = 1.0  # total trajectory time
waypoints = path_data.shape[0]  # number of waypoints
tWp = np.linspace(0, T, waypoints)  # time for each waypoint
allConfigTraj = []

# --- Extract XYZ positions ---
posX = path_data[:, 0]
posY = path_data[:, 1]
posZ = path_data[:, 2]

# --- Cubic spline for position (clamped: zero velocity at start/end) ---
# 'bc_type' = 'clamped' ensures zero velocity at endpoints
splineX = CubicSpline(tWp, posX, bc_type='clamped')
splineY = CubicSpline(tWp, posY, bc_type='clamped')
splineZ = CubicSpline(tWp, posZ, bc_type='clamped')

tFine = np.linspace(0, T, waypoints)  # interpolated time
posTraj = np.column_stack((splineX(tFine), splineY(tFine), splineZ(tFine)))

# --- Quaternion interpolation ---
# MATLAB: quaternion(w,x,y,z)
# Python: scipy Rotation.from_quat expects (x, y, z, w)
quaternions = []
for j in range(waypoints):
    # Convert MATLAB [w,x,y,z] to Python [x,y,z,w]
    w, x, y, z = path_data[j, 6], path_data[j, 3], path_data[j, 4], path_data[j, 5]
    quaternions.append(R.from_quat([x, y, z, w]))


slerp = Slerp(tWp, R.concatenate(quaternions))
qInterp = slerp(tFine)  # Rotation objects for each fine time step

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
    qk = qInterp[i]
    T_target.R = qk.as_matrix()  # 3x3 rotation matrix
    # Solve IK
    configNow = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True,method = 'sugihara',k=0.0001, q0=qIK)  # replace with your Python IK function
    # Handle IK failure
    if not configNow.success:
        print(f"Warning: IK failed at waypoint {i}, using previous config")
        qIK = prevConfig.copy()
        J = robot.jacob0(qIK, endEffector)
        w = np.sqrt(np.linalg.det(J @ J.T))
        penalty+=1
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

    #lambda_sq = 0.01  # damping factor squared
    #JJT = J @ J.T
    #dq = J.T @ np.linalg.inv(JJT + lambda_sq * np.eye(JJT.shape[0])) @ dx

    dq_null = 0.1 * (N @ (singularity_gradient( qIK, endEffector)))
    qNext = (qIK + dq_null).T

    # --- Map to nearest equivalent angles & smooth ---
    qSmooth = mapToNearest(prevConfig, qNext)
    alpha = 0.5
    qFiltered = alpha * qSmooth + (1 - alpha) * prevConfig
    # Store trajectory
    configTraj[i, :] = qFiltered
    prevConfig = qFiltered
    # Joint velocity
    if i == 0:
        velJointTraj[i, :] = np.zeros(len(qStart))
    else:
        dt = tFine[i] - tFine[i - 1]
        velJointTraj[i, :] = (configTraj[i, :] - configTraj[i - 1, :]) / dt

# --- Compute delta in degrees if needed ---
deltaJointRad = np.diff(configTraj, axis=0)
deltaJointDeg = np.rad2deg(deltaJointRad)
absDeltaJointDeg = np.abs(deltaJointDeg)

# --- Final smooth trajectory ---
allConfigTraj = np.hstack((configTraj, velJointTraj))
np.savetxt(matlab + "allConfigTraj.csv", allConfigTraj, delimiter=",", fmt="%.6f")

while penalty > 0:
    print("replanning...")
    # Create planner
    planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(ob.timedPlannerTerminationCondition(20.0))

    if solved:

        # -----------------------------
        #       Get solution path
        # -----------------------------
        path = pdef.getSolutionPath()
        path.interpolate(300)  # densify path

        # -----------------------------
        #        Save path to CSV
        # -----------------------------
        path_csv = matlab + "path3d.csv"
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

        explored_csv = matlab + "explored_points.csv"
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

    # Load Path
    path_data = np.loadtxt(matlab + 'path3d.csv', delimiter=",")

    # Number of waypoints (rows)
    waypoints = path_data.shape[0]

    print("Path data:\n", path_data)
    print(waypoints)

    endEffector = 'tool'  # name of end-effector link

    # Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
    weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])
    # Initial joint configuration guess
    q0 = np.zeros(robot.n)  # or your previous configuration

    # --- Parameters ---
    T = 1.0  # total trajectory time
    waypoints = path_data.shape[0]  # number of waypoints
    tWp = np.linspace(0, T, waypoints)  # time for each waypoint
    allConfigTraj = []

    # --- Extract XYZ positions ---
    posX = path_data[:, 0]
    posY = path_data[:, 1]
    posZ = path_data[:, 2]

    # --- Cubic spline for position (clamped: zero velocity at start/end) ---
    # 'bc_type' = 'clamped' ensures zero velocity at endpoints
    splineX = CubicSpline(tWp, posX, bc_type='clamped')
    splineY = CubicSpline(tWp, posY, bc_type='clamped')
    splineZ = CubicSpline(tWp, posZ, bc_type='clamped')

    tFine = np.linspace(0, T, waypoints)  # interpolated time
    posTraj = np.column_stack((splineX(tFine), splineY(tFine), splineZ(tFine)))

    # --- Quaternion interpolation ---
    # MATLAB: quaternion(w,x,y,z)
    # Python: scipy Rotation.from_quat expects (x, y, z, w)
    quaternions = []
    for j in range(waypoints):
        # Convert MATLAB [w,x,y,z] to Python [x,y,z,w]
        w, x, y, z = path_data[j, 6], path_data[j, 3], path_data[j, 4], path_data[j, 5]
        quaternions.append(R.from_quat([x, y, z, w]))

    slerp = Slerp(tWp, R.concatenate(quaternions))
    qInterp = slerp(tFine)  # Rotation objects for each fine time step

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
        qk = qInterp[i]
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
            penalty = +1
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

        lambda_sq = 0.01  # damping factor squared
        JJT = J @ J.T
        dq = J.T @ np.linalg.inv(JJT + lambda_sq * np.eye(JJT.shape[0])) @ dx

        dq_null = 0.1 * (N @ (dq + singularity_gradient(qIK, endEffector)))
        qNext = (qIK + dq_null).T

        # --- Map to nearest equivalent angles & smooth ---
        qSmooth = mapToNearest(prevConfig, qNext)
        alpha = 0.5
        qFiltered = alpha * qSmooth + (1 - alpha) * prevConfig
        # Store trajectory
        configTraj[i, :] = qFiltered
        prevConfig = qFiltered
        # Joint velocity
        if i == 0:
            velJointTraj[i, :] = np.zeros(len(qStart))
        else:
            dt = tFine[i] - tFine[i - 1]
            velJointTraj[i, :] = (configTraj[i, :] - configTraj[i - 1, :]) / dt

    # --- Compute delta in degrees if needed ---
    deltaJointRad = np.diff(configTraj, axis=0)
    deltaJointDeg = np.rad2deg(deltaJointRad)
    absDeltaJointDeg = np.abs(deltaJointDeg)

    # --- Final smooth trajectory ---
    allConfigTraj = np.hstack((configTraj, velJointTraj))
    np.savetxt(matlab + "allConfigTraj.csv", allConfigTraj, delimiter=",", fmt="%.6f")

#--------------------#
#  COLLISION CHECK   #
#--------------------#

# Get joint indices
num_joints = p.getNumJoints(robotId)
joint_indices = list(range(num_joints))
collision_penalty = 0
for waypoints, joint_positions in enumerate(configTraj):
    # Set joint positions (robot, joint(i), theta)
    for joint_index, joint_value in zip(joint_indices, joint_positions):
        p.resetJointState(robotId, joint_index, joint_value)

    # --- Environment collision ---
    contacts_env = p.getClosestPoints(robotId, obstacleId, distance=0)

    # --- Self collision ---
    contacts_self_raw = p.getClosestPoints(robotId, robotId, distance=0)

    # Remove trivial contacts (same link pairs)
    contacts_self = []
    unique_pairs = set()
    ee_link = 7
    for c in contacts_self_raw:
        linkA =c[3]
        linkB =c[4]

        # Only care if EE is involved
        if ee_link not in (linkA, linkB):
            continue

        # Identify the "other" link
        other_link = linkB if linkA == ee_link else linkA

        # Ignore same link
        if other_link == ee_link:
            continue

        # Ignore parent-child contact
        parent = p.getJointInfo(robotId, ee_link)[16]
        if other_link == parent:
            continue

        pair = tuple(sorted((ee_link, other_link)))

        if pair not in unique_pairs:
            unique_pairs.add(pair)
            print(f"End-effector collides with link {other_link}")
            if other_link == 3 | other_link == 4:
                collision_penalty +=1
    if contacts_env:
        print(f"Collision at waypoint {waypoints}")
        print("  → Environment collision")
attempt_collision = 0
while collision_penalty > 0 and attempt_collision < 5:
    print("Collision penalty:", collision_penalty)
    print("retry IK SOLVER")
    print("attempt :", attempt_collision)
    # --- Main loop: IK + null-space optimization + smoothing ---
    for i in range(waypoints):
        # Desired end-effector pose
        T_target.t = posTraj[i, :]
        qk = qInterp[i]
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
            penalty += 1
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

        lambda_sq = 0.01  # damping factor squared
        JJT = J @ J.T
        dq = J.T @ np.linalg.inv(JJT + lambda_sq * np.eye(JJT.shape[0])) @ dx

        dq_null = 0.1 * (N @ (dq + singularity_gradient(qIK, endEffector)))
        qNext = (qIK + dq_null).T

        # --- Map to nearest equivalent angles & smooth ---
        qSmooth = mapToNearest(prevConfig, qNext)
        alpha = 0.5
        qFiltered = alpha * qSmooth + (1 - alpha) * prevConfig
        # Store trajectory
        configTraj[i, :] = qFiltered
        prevConfig = qFiltered
        # Joint velocity
        if i == 0:
            velJointTraj[i, :] = np.zeros(len(qStart))
        else:
            dt = tFine[i] - tFine[i - 1]
            velJointTraj[i, :] = (configTraj[i, :] - configTraj[i - 1, :]) / dt

    # --- Compute delta in degrees if needed ---
    deltaJointRad = np.diff(configTraj, axis=0)
    deltaJointDeg = np.rad2deg(deltaJointRad)
    absDeltaJointDeg = np.abs(deltaJointDeg)

    # --- Final smooth trajectory ---
    allConfigTraj = np.hstack((configTraj, velJointTraj))
    np.savetxt(matlab + "allConfigTraj.csv", allConfigTraj, delimiter=",", fmt="%.6f")

    # --------------------#
    #  COLLISION CHECK   #
    # --------------------#

    # Get joint indices
    num_joints = p.getNumJoints(robotId)
    joint_indices = list(range(num_joints))
    collision_penalty = 0
    for waypoints, joint_positions in enumerate(configTraj):
        # Set joint positions (robot, joint(i), theta)
        for joint_index, joint_value in zip(joint_indices, joint_positions):
            p.resetJointState(robotId, joint_index, joint_value)

        # --- Environment collision ---
        contacts_env = p.getClosestPoints(robotId, obstacleId, distance=0)

        # --- Self collision ---
        contacts_self_raw = p.getClosestPoints(robotId, robotId, distance=0)

        # Remove trivial contacts (same link pairs)
        contacts_self = []
        unique_pairs = set()
        ee_link = 7
        for c in contacts_self_raw:
            linkA = c[3]
            linkB = c[4]

            # Only care if EE is involved
            if ee_link not in (linkA, linkB):
                continue

            # Identify the "other" link
            other_link = linkB if linkA == ee_link else linkA

            # Ignore same link
            if other_link == ee_link:
                continue

            # Ignore parent-child contact
            parent = p.getJointInfo(robotId, ee_link)[16]
            if other_link == parent:
                continue

            pair = tuple(sorted((ee_link, other_link)))

            if pair not in unique_pairs:
                unique_pairs.add(pair)
                print(f"End-effector collides with link {other_link}")
                if other_link == 3 | other_link == 4:
                    collision_penalty += 1
        if contacts_env:
            print(f"Collision at waypoint {waypoints}")
            print("  → Environment collision")
            collision_penalty += 1

    attempt_collision+=1

if attempt_collision ==5:
    print ("choose the other goal pose")