import roboticstoolbox as rtb
import numpy as np
import os
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from roboticstoolbox import Robot
from spatialmath import SE3
import numpy as np
import qpsolvers


def mapToNearest(robot, q_prev, q_new):
    """
    Map q_new to the nearest equivalent configuration relative to q_prev.

    Parameters:
        robot : roboticstoolbox Robot object
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


import numpy as np


def singularity_gradient(robot, q, body_name):
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


# Load URDF of Robot
matlab = "/Users/kim/Documents/MATLAB/New Folder/path_planning/"
robot = rtb.Robot.URDF(matlab + "imed_robot.urdf")
# robot = rtb.Robot.URDF(matlab+ "two_link_robot.urdf")
# Load Path
path_data = np.loadtxt(matlab + 'path3d.csv', delimiter=",")

# Number of waypoints (rows)
waypoints = path_data.shape[0]

print("Path data:\n", path_data)
print(waypoints)

endEffector = 'tool'  # name of end-effector link

# Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])
IK = rtb.IK_LM()
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

# Slerp interpolation
from scipy.spatial.transform import Slerp

slerp = Slerp(tWp, R.concatenate(quaternions))
qInterp = slerp(tFine)  # Rotation objects for each fine time step

# --- Initialize configuration ---
qStart = q0  # Python list or numpy array
prevConfig = np.array(qStart)
configTraj = np.zeros((waypoints, len(qStart)))
velJointTraj = np.zeros((waypoints, len(qStart)))
T_target = SE3()
qIK = q0
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
        # print(f"Warning: IK failed at waypoint {i}, using previous config")
        qIK = prevConfig.copy()
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

    #dq_null = 0.1 * (N @ (0.5*dq+ 0.5*singularity_gradient(robot, qIK, endEffector)))
    dq_null = 0.1 * (N @ singularity_gradient(robot, qIK, endEffector))
    qNext = (qIK + dq_null).T

    # --- Map to nearest equivalent angles & smooth ---
    qSmooth = mapToNearest(robot, prevConfig, qNext)
    alpha = 0.5
    qFiltered = alpha * qSmooth + (1 - alpha) * prevConfig
    # Store trajectory
    configTraj[i, :] = qFiltered
    prevConfig = qFiltered
    print(f"{i}:{qFiltered}")
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
