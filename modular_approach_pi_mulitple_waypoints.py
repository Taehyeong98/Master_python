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

def mapToNearest(q_prev, q_new):
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


def singularity_gradient(q, body_name):
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
    pos = np.array([state.getX(), state.getY(), state.getZ()])

    tf = fcl.Transform(np.eye(3), pos)
    path_point.setTransform(tf)

    dist_goal = np.linalg.norm(pos - goalPos)
    dist_start = np.linalg.norm(pos - startPos)
    GOAL_RADIUS = 0.3
    INITIAL_RADIUS = 0.3
    inGoalRegion = dist_goal < GOAL_RADIUS
    inInitialRegion = dist_start < INITIAL_RADIUS

    # Goal reached
    if dist_goal < GOAL:
        return True

    # Inside special regions → patient collision only
    if inGoalRegion or inInitialRegion:
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        fcl.collide(path_point, obj, req, res)
        #fcl.collide(path_point, box1_obj, req, res)
        #fcl.collide(path_point, box2_obj, req, res)
        if res.is_collision:
            return False
        return True

    # Outside regions → ground collision only
    else:
        req = fcl.CollisionRequest()
        res = fcl.CollisionResult()
        fcl.collide(path_point, groundBlock, req, res)
        #fcl.collide(path_point, box1_obj, req, res)
        #fcl.collide(path_point, box2_obj, req, res)
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



def in_region(start, point, radius):
    sx, sy, sz = start
    x, y, z = point

    distance = math.sqrt((x - sx)**2 + (y - sy)**2+(z - sz)**2)
    return distance <= radius

#--------------------#
#      Pybullet      #
#--------------------#
# Connect to DIRECT
bc = bullet_client.BulletClient(connection_mode=p.DIRECT)
# Optional: set search path for meshes
bc.setAdditionalSearchPath(pybullet_data.getDataPath())


waypoints_input = int(input("Number of waypoints: "))
all_trajectories = []
for i in range(waypoints_input-1):
    if i == 0:
        start_pose_deg = list(map(float, input("Start pose in degrees: ").split()))
        goal_pose_input = list(map(float, input("Next Goal position (x y z qx qy qz): ").split()))
        q_start_prismatic = start_pose_deg[0]
        q_start_revolute = np.radians(start_pose_deg[1:7])  # Convert revolute joints (2–7) to radians
        start_pose_rad = np.concatenate(([q_start_prismatic], q_start_revolute))  # Combine back into one joint vector
    else:
        load_allConfigTraj = np.loadtxt('allConfigTraj.txt', delimiter=",")
        start_pose_rad = load_allConfigTraj[-1]
        print(start_pose_rad)
        goal_pose_input = list(map(float, input("Next Goal position (x y z qx qy qz): ").split()))

    start_time = time.time()

    address = "/home/pi/Desktop/Master_python/"
    #address = "/Users/kim/PycharmProjects/JupyterProject/"
    robotId=bc.loadURDF(address + "imed_robot_update.urdf")
    position = [0.55, 0.35, -0.55]        # must be length 3
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

    # CUBE 1
    x1 = [1.2, 1.27]
    y1 = [0.6, 0.7]
    z1 = [0.3, 0.84]

    size1 = [x1[1] - x1[0], y1[1] - y1[0], z1[1] - z1[0]]
    center1 = [(x1[0] + x1[1]) / 2, (y1[0] + y1[1]) / 2, (z1[0] + z1[1]) / 2]

    # PyBullet uses HALF extents
    half_extents1 = [s / 2 for s in size1]

    colBox1 = bc.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents1
    )

    body1 = bc.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=colBox1,
        basePosition=center1
    )
    # CUBE 2
    # Cube 2 dimensions
    x2 = [1.2, 1.27]
    y2 = [0.27, 0.6]
    z2 = [0.38, 0.48]

    size2 = [x2[1] - x2[0], y2[1] - y2[0], z2[1] - z2[0]]
    center2 = [(x2[0] + x2[1]) / 2, (y2[0] + y2[1]) / 2, (z2[0] + z2[1]) / 2]

    half_extents2 = [s / 2 for s in size2]

    colBox2 = bc.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents2
    )

    body2 = bc.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=colBox2,
        basePosition=center2
    )

    # Load URDF of Robot
    robot = rtb.Robot.URDF( address+"imed_robot_update.urdf")
    endEffector = 'tool'  # name of end-effector link

    # Define weights for position (x,y,z) and orientation (roll, pitch, yaw)
    weights = np.array([0.5, 0.5, 0.5, 1, 1, 1])

    # --- Initialize configuration ---
    #start_pose_rad = [0.2,2.196,1.769,-0.716,0.611,-1.560,-0.782]  # Python list or numpy array

    # goal pose
    goal_pos = [goal_pose_input[0], goal_pose_input[1], goal_pose_input[2]]
    qx = R.from_rotvec(np.deg2rad(goal_pose_input[3]) * np.array([1, 0, 0]))
    qy = R.from_rotvec(np.deg2rad(goal_pose_input[4]) * np.array([0, 1, 0]))
    qz = R.from_rotvec(np.deg2rad(goal_pose_input[5]) * np.array([0, 0, 1]))
    goal_ori = (qx*qy*qz).as_quat()  # returns [x,y,z,w]
    T_target = SE3()
    T_target.t = goal_pos
    T_target.R = R.from_quat(goal_ori).as_matrix()

    # park pose
    const = 0.3
    if goal_pos[0] > start_pose_rad[0]:
        park_0 = goal_pos[0]-const
    else:
        park_0 = goal_pos[0]+const

    park_pose_end = [park_0,90.0, 90.0, -90.0, 0.0, -90.0, 0.0]
    q_park_prismatic = park_pose_end[0]
    q_park_revolute = np.radians(park_pose_end[1:7])
    park_pose_end_rad = np.concatenate(([q_park_prismatic], q_park_revolute))

    FK_park_end = robot.fkine(park_pose_end_rad)
    park_end_position = FK_park_end.t
    quat_park_end = R.from_matrix(FK_park_end.R).as_quat()

    park_pose_goal = [goal_pose_input[0],90.0, 90.0, -90.0, 0.0, -90.0, 0.0]
    q_park_prismatic_goal = park_pose_goal[0]
    q_park_revolute_goal = np.radians(park_pose_goal[1:7])
    park_pose_goal_rad = np.concatenate(([q_park_prismatic_goal], q_park_revolute_goal))
    #--------------------#
    #  Initial Movement  #
    #--------------------#

    INITIAL_RADIUS = 0.4

    FK_start = robot.fkine(start_pose_rad)
    start_position = FK_start.t
    quat_start = R.from_matrix(FK_start.R).as_quat()  # returns [x y z w]

    print("start:",start_position)
    initial_region = in_region(start_position, goal_pos, INITIAL_RADIUS)

    # CHECKING LIST
    initial_check = False
    reverse_check = False
    parkpose_check = False
    startpose_check = False
    collision_check = True
    first_try_check = False
    initial_collision_check = True
    goal_trajectory_check = True
    STL_check = False
    hard_to_reach_check = False
    #----------INITIAL CHECK----------#
    if initial_region:
        print("the goal is in initial region and the initial movement is skipped")
        initial_check = True

    #----------START POSE CHECK----------#
    if np.allclose (start_pose_rad[1:], park_pose_end_rad[1:], atol=1e-3):
        startpose_check = True
        print("park pose trajectory is skipped")
        park_pose_start = start_pose_rad


    start_traj = np.loadtxt('goaltrajectory.txt', delimiter=",")
    start_traj_size =start_traj.shape[0]

    print(start_traj_size)
    park_check_pose = [0, 1.571, 1.571, -1.571, 0.000, -1.571, 0.000]
    if start_traj_size > 0:
        prev_goal_pose = start_traj[-1]
        prev_park_pose = start_traj[0]
    #----------REVERSE POSSIBILITY CHECK----------#
        if np.allclose(prev_goal_pose, start_pose_rad,atol=1e-3):
            reverse_check= True


    #----------PARK POSE CHECK----------#
        if np.allclose(prev_park_pose[1:], park_check_pose[1:], atol=1e-3):
            parkpose_check = True
        else:
            parkpose_check = False


    if start_traj_size >0 and initial_check is False and startpose_check is False and reverse_check is True and parkpose_check is True:
        print("the initial trajectory is the previous goal trajectory")
        reversed_traj = start_traj[::-1]
        np.savetxt("reversed_traj.txt", reversed_traj, delimiter=",", fmt="%.3f")
        park_pose_start = reversed_traj[-1]


    if initial_check is True:
        solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                                q0=start_pose_rad)  # replace with your Python IK function
        counter = 0
        while (solver.success == False):
            print("IK solver failed, try again")
            solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                                    q0=start_pose_rad)  # replace with your Python
            counter = counter + 1
            if counter > 10:
                print("GOAL POSITION is not reachable!!")
                sys.exit()
        q_sol = solver.q
        print(q_sol)
        goal_pose_q = q_sol  # 1x7
    else:
        solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                                q0=park_pose_goal_rad)  # replace with your Python IK function
        counter = 0
        while (solver.success == False):
            print("IK solver failed, try again")
            solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                                    q0=park_pose_goal_rad)  # replace with your Python
            counter = counter + 1
            if counter > 10:
                print("GOAL POSITION is not reachable!!")
                print("move little forward and try again")
                # --------------------#
                #    Move FORWARDS    #
                # --------------------#

                move_forwards_rad = [park_pose_end_rad[0], -0.6896, 2.268, -0.4, -0.514, 1.156, 0.752]
                find_pos_traj_gen = jtraj(park_pose_end_rad, move_forwards_rad, 100)
                find_pos_traj = find_pos_traj_gen.q
                find_pos_q = find_pos_traj[-1]
                FK_park_to_find_pos = robot.fkine(find_pos_q)
                find_pos_position = FK_park_to_find_pos.t
                find_pos_quat = R.from_matrix(FK_park_to_find_pos.R).as_quat()
                post_counter = 0

                solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                                        q0=move_forwards_rad)  # replace with your Python IK function
                while (solver.success == False):
                    print("IK solver failed, try again")
                    solver = robot.ikine_LM(Tep=T_target, mask=weights, joint_limits=True, method='sugihara', k=0.0001,
                                            q0=move_forwards_rad)  # replace with your Python
                    post_counter = post_counter + 1
                    if post_counter > 10:
                        print("GOAL POSITION is not reachable!!")
                        sys.exit()
                hard_to_reach_check = True
                print("Hard To Reach!!")


        q_sol = solver.q
        print("goal pose:",q_sol)
        goal_pose_q = q_sol  # 1x7


    if initial_check is False and startpose_check is False and reverse_check is False or initial_check is False and startpose_check is False and reverse_check is True and parkpose_check is False:
        if first_try_check is False:
            park_pose_start = [start_pose_rad[0], 1.571, 1.571, -1.571, 0.000, -1.571, 0.000]
            initial_traj = jtraj(start_pose_rad, park_pose_start, 50)
            first_try_check = True

        # --------------------#
        #  COLLISION CHECK   #
        # --------------------#
        print("Waypoints:", len(initial_traj.q))
        # Get joint indices
        num_joints = p.getNumJoints(robotId)

        # Only actuated joints
        joint_indices = [i for i in range(num_joints) if p.getJointInfo(robotId, i)[2] != p.JOINT_FIXED]

        for waypoint_idx, waypoint_joints in enumerate(initial_traj.q):
            for joint_index, joint_value in zip(joint_indices, waypoint_joints):
                bc.resetJointState(robotId, joint_index, joint_value)

            bc.stepSimulation()

            contacts_env = bc.getClosestPoints(robotId, obstacleId, distance=-0.05)

            for contact in contacts_env:
                if contact[8] < 0:  # penetration
                    print(f"Collision at waypoint {waypoint_idx}:{contact[8]}")
                    initial_collision_check = False

            if initial_collision_check is True:
                allConfigTraj_start = initial_traj.q

        if first_try_check is True and initial_collision_check is False:
            print("the initial trajectory generation is started")
            STL_check = True
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
            # Cube 1 dimensions

            box1 = fcl.Box(*size1)
            box1_obj = fcl.CollisionObject(box1)

            tf1 = fcl.Transform(np.eye(3), np.array(center1))
            box1_obj.setTransform(tf1)

            box2 = fcl.Box(*size2)
            box2_obj = fcl.CollisionObject(box2)

            tf2 = fcl.Transform(np.eye(3), np.array(center2))
            box2_obj.setTransform(tf2)

            x3 = [0, 0.5]
            y3 = [0, 0.05]
            z3 = [0, 0.84]

            size3 = [x3[1] - x3[0], y3[1] - y3[0], z3[1] - z3[0]]
            center3 = [(x3[0] + x3[1]) / 2, (y3[0] + y3[1]) / 2, (z3[0] + z3[1]) / 2]

            box3 = fcl.Box(*size3)
            box3_obj = fcl.CollisionObject(box3)

            tf3 = fcl.Transform(np.eye(3), np.array(center3))
            box3_obj.setTransform(tf3)

            x4 = [0.2, 0.3]
            y4 = [0.05, 0.1]
            z4 = [0.45, 0.5]

            size4 = [x4[1] - x4[0], y4[1] - y4[0], z4[1] - z4[0]]
            center4 = [(x4[0] + x4[1]) / 2, (y4[0] + y4[1]) / 2, (z4[0] + z4[1]) / 2]

            box4 = fcl.Box(*size4)
            box4_obj = fcl.CollisionObject(box4)

            tf4 = fcl.Transform(np.eye(3), np.array(center4))
            box4_obj.setTransform(tf4)


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

            robot = rtb.Robot.URDF(address + "imed_robot_update.urdf")

            print("robot urdf initialized")

            # --------------------#
            #    Path Planner    #
            # --------------------#
            point_r = 0.005
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

            start().setXYZ(start_position[0], start_position[1], start_position[2]+0.06)

            park_pose_x = start_position[0]
            # PARK POSE #
            goal().setXYZ(park_pose_x, 0.4, 0.5)

            q = (0.7071, -0.7071, 0, 0)  # returns [x,y,z,w]

            start().rotation().x = quat_start[0]
            start().rotation().y = quat_start[1]
            start().rotation().z = quat_start[2]
            start().rotation().w = quat_start[3]

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

            RADIUS = 0.3
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

            solved = planner.solve(ob.timedPlannerTerminationCondition(12.0))

            if solved:

                # -----------------------------
                #       Get solution path
                # -----------------------------
                path = pdef.getSolutionPath()
                path.interpolate(30)  # densify path

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
                sys.exit(0)

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

            qStart = start_pose_rad  # Python list or numpy array
            prevConfig = np.array(qStart)
            configTraj = np.zeros((waypoints, len(qStart)))
            velJointTraj = np.zeros((waypoints, len(qStart)))
            T_target = SE3()
            qIK = start_pose_rad
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
            allConfigTraj_start_prev = np.zeros((num_samples, num_joints))

            # Interpolate each joint separately
            for j in range(num_joints):
                cs = CubicSpline(t_waypoints, configTraj[:, j], bc_type='clamped')  # clamped ensures zero slope at ends
                allConfigTraj_start_prev[:, j] = cs(t_samples)

            # ---------ADJUSTMENT-----------#
            allConfigTraj_end = allConfigTraj_start_prev[-1]
            park_pose_start = [park_pose_x,1.571, 1.571, -1.571, 0.000, -1.571, 0.000]
            adjustment_start = jtraj(allConfigTraj_end, park_pose_start, 50)

            allConfigTraj_start =  np.concatenate((allConfigTraj_start_prev,adjustment_start.q))




    if initial_region is False:
        #--------------------#
        #  Linear Movement   #
        #--------------------#
        allConfigTraj =[]
        linear_time = 250
        linear_movement = np.zeros((linear_time,len(start_pose_rad)))
        linear_movement[0:] = park_pose_start
        x_start = park_pose_start[0]
        x_goal = park_pose_end_rad[0]

        linear_distance = x_goal - x_start


        step = linear_distance / linear_time
        for i in range(linear_time):
            # Prismatic joint moves linearly
            x_current = x_start + step * i

            # Revolute joints stay fixed
            q_current = np.concatenate((
                [x_current],
                park_pose_start[1:]
            ))

            linear_movement[i] = q_current

        park_pose_rad_end_linear = linear_movement[linear_time-1,:]
        if hard_to_reach_check:
            linear_movement = np.concatenate((linear_movement, find_pos_traj))
            park_pose_rad_end_linear_extended = linear_movement[linear_time-1,:]
            traj = jtraj(park_pose_rad_end_linear_extended, goal_pose_q, 500)
        else:
            traj = jtraj(park_pose_rad_end_linear, goal_pose_q, 500)

        # --------------------#
        #  COLLISION CHECK   #
        # --------------------#
        num_joints = p.getNumJoints(robotId)

        # Only actuated joints
        joint_indices = [i for i in range(num_joints) if p.getJointInfo(robotId, i)[2] != p.JOINT_FIXED]

        for waypoint_idx, waypoint_joints in enumerate(traj.q):
            for joint_index, joint_value in zip(joint_indices, waypoint_joints):
                p.resetJointState(robotId, joint_index, joint_value)

            bc.stepSimulation()

            contacts_env = bc.getClosestPoints(robotId, obstacleId, distance=-0.05)
            #contacts_robot1 = bc.getClosestPoints(robotId, body1, distance=-0.03)
            #contacts_robot2 = bc.getClosestPoints(robotId, body2, distance=-0.03)

            for contact in contacts_env:
                if contact[8] < -0.05:  # penetration
                    print(f"Collision at waypoint {waypoint_idx}{contact[8]}")
                    goal_trajectory_check = False
            '''
            for contact_2 in contacts_robot1:
                if contact_2[8] < -0.03:  # penetration
                    print(f"Collision with co-robot at waypoint {waypoint_idx}{contact_2[8]}")
                    goal_trajectory_check = False
            for contact_3 in contacts_robot2:
                if contact_3[8] < -0.03:  # penetration
                    print(f"Collision with co-robot at waypoint {waypoint_idx}{contact_3[8]}")
                    goal_trajectory_check = False
            '''
            contacts_self_raw = bc.getClosestPoints(robotId, robotId, distance=0.001)
            unique_pairs = set()
            for c in contacts_self_raw:
                linkA, linkB, distance = c[3], c[4], c[8]

                if distance >= -1e-5 or linkA == linkB:
                    continue

                if {linkA, linkB} == {3, 5}:
                    continue

                parentA = p.getJointInfo(robotId, linkA)[16] if linkA != -1 else -1
                parentB = p.getJointInfo(robotId, linkB)[16] if linkB != -1 else -1

                if parentA == linkB or parentB == linkA:
                    continue

                pair = tuple(sorted((linkA, linkB)))
                if pair not in unique_pairs:
                    unique_pairs.add(pair)
                    print(f"Self collision between link {linkA + 1} and link {linkB + 1} at waypoint {waypoint_idx}")
                    goal_trajectory_check = False



        if goal_trajectory_check is True:
            if initial_check is False and startpose_check is False and reverse_check is True and parkpose_check is True:
                trajectory  = np.concatenate((reversed_traj,linear_movement,traj.q))
            elif initial_check is False and startpose_check is True:
                trajectory = np.concatenate((linear_movement,traj.q))
            else:
                trajectory  = np.concatenate((allConfigTraj_start,linear_movement,traj.q))

            np.savetxt("goaltrajectory.txt", traj.q, delimiter=",", fmt="%.3f")

            np.savetxt("allConfigTraj.txt", trajectory, delimiter=",", fmt="%.3f")
        else:
            print("Collision in the goal trajectory!!")
            print("Path Planner starts!!")

            if STL_check is False:
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

            # --------------------#
            #    Move FORWARDS    #
            # --------------------#
            move_forwards_rad =[park_pose_end_rad[0],2.452, 1.768, -0.4, -0.514, 1.156, 0.752]
            find_pos_traj_gen = jtraj(park_pose_end_rad, move_forwards_rad, 100)
            find_pos_traj = find_pos_traj_gen.q
            find_pos_q = find_pos_traj[-1]
            FK_park_to_find_pos = robot.fkine(find_pos_q)
            find_pos_position = FK_park_to_find_pos.t
            find_pos_quat = R.from_matrix(FK_park_to_find_pos.R).as_quat()

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
            # Cube 2 dimensions
            inflation =0.075
            x2_inflation = [1.2-inflation, 1.27+inflation]
            y2_inflation = [0.27-inflation, 0.6+inflation]
            z2_inflation = [0.38-inflation, 0.5+inflation]


            size2_inflation = [x2_inflation[1] - x2_inflation[0], y2_inflation[1] - y2_inflation[0], z2_inflation[1] - z2_inflation[0]]
            center2_inflation = [(x2_inflation[0] + x2_inflation[1]) / 2, (y2_inflation[0] + y2_inflation[1]) / 2, (z2_inflation[0] + z2_inflation[1]) / 2]

            box1 = fcl.Box(*size1)
            box1_obj = fcl.CollisionObject(box1)

            tf1 = fcl.Transform(np.eye(3), np.array(center1))
            box1_obj.setTransform(tf1)

            box2 = fcl.Box(*size2_inflation)
            box2_obj = fcl.CollisionObject(box2)

            tf2 = fcl.Transform(np.eye(3), np.array(center2_inflation))
            box2_obj.setTransform(tf2)

            x4 = [0.2, 0.3]
            y4 = [0.05, 0.1]
            z4 = [0.45, 0.5]

            size4 = [x4[1] - x4[0], y4[1] - y4[0], z4[1] - z4[0]]
            center4 = [(x4[0] + x4[1]) / 2, (y4[0] + y4[1]) / 2, (z4[0] + z4[1]) / 2]

            box4 = fcl.Box(*size4)
            box4_obj = fcl.CollisionObject(box4)

            tf4 = fcl.Transform(np.eye(3), np.array(center4))
            box4_obj.setTransform(tf4)

            # --------------------#
            #    Path Planner    #
            # --------------------#
            point_r = 0.01
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

            start().setXYZ(find_pos_position[0], find_pos_position[1], find_pos_position[2] + 0.06)
            # PARK POSE #
            goal().setXYZ(goal_pos[0], goal_pos[1], goal_pos[2])

            q = goal_ori  # returns [x,y,z,w]

            start().rotation().x = find_pos_quat[0]
            start().rotation().y = find_pos_quat[1]
            start().rotation().z = find_pos_quat[2]
            start().rotation().w = find_pos_quat[3]

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

            RADIUS = 0.3
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

            solved = planner.solve(ob.timedPlannerTerminationCondition(12.0))

            if solved:

                # -----------------------------
                #       Get solution path
                # -----------------------------
                path = pdef.getSolutionPath()
                path.interpolate(30)  # densify path

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
                sys.exit(0)

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

            qStart = find_pos_q  # Python list or numpy array
            prevConfig = np.array(qStart)
            configTraj = np.zeros((waypoints, len(qStart)))
            velJointTraj = np.zeros((waypoints, len(qStart)))
            T_target = SE3()
            qIK = find_pos_q
            penalty = 0
            # --- Main loop: IK + null-space optimization + smoothing ---
            for i in range(waypoints):
                # Desired end-effector pose
                T_target.t = posTraj[i, :]
                qk = quaternions[i]
                T_target.R = qk.as_matrix()  # 3x3 rotation matrix

                m = robot.manipulability(qIK)
                if m < 0.01:
                    print("the trajectory is infeasible")
                    sys.exit(0)
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
            allConfigTraj_goal = np.zeros((num_samples, num_joints))

            # Interpolate each joint separately
            for j in range(num_joints):
                cs = CubicSpline(t_waypoints, configTraj[:, j], bc_type='clamped')  # clamped ensures zero slope at ends
                allConfigTraj_goal[:, j] = cs(t_samples)

            find_pos_traj_end = find_pos_traj[-1]
            allConfigTraj_goal_start = allConfigTraj_goal[0]
            transition_traj_gen = jtraj(find_pos_traj_end, allConfigTraj_goal_start, 50)
            transition_traj = transition_traj_gen.q
            if initial_check is False and startpose_check is False and reverse_check is True and parkpose_check is True:
                trajectory  = np.concatenate((reversed_traj,linear_movement,find_pos_traj,transition_traj,allConfigTraj_goal))
            elif initial_check is False and startpose_check is True:
                trajectory = np.concatenate((linear_movement,find_pos_traj,transition_traj,allConfigTraj_goal))
            else:
                trajectory  = np.concatenate((allConfigTraj_start,linear_movement,find_pos_traj,transition_traj,allConfigTraj_goal))

            sampling_traj_goal = np.concatenate((find_pos_traj,transition_traj,allConfigTraj_goal))

            np.savetxt("goaltrajectory.txt", sampling_traj_goal,delimiter=",", fmt="%.3f")

            np.savetxt( "allConfigTraj.txt", trajectory, delimiter=",", fmt="%.3f")

    else:
        trajectory_q = jtraj(start_pose_rad, goal_pose_q, 500)
        trajectory = trajectory_q.q
        np.savetxt( "allConfigTraj.txt", trajectory, delimiter=",", fmt="%.3f")

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

        contacts_env = bc.getClosestPoints(robotId, obstacleId, distance=-0.05)
        #contacts_robot1 = bc.getClosestPoints(robotId, body1, distance=-0.03)
        #contacts_robot2 = bc.getClosestPoints(robotId, body2, distance=-0.03)
        for contact in contacts_env:
            if contact[8] < -0.05:  # penetration
                print(f"Collision at waypoint {waypoint_idx}{contact[8]}")
                initial_collision_check = False
        '''
        for contact_2 in contacts_robot1:
            if contact_2[8] < -0.03:  # penetration
                print(f"Collision with co-robot at waypoint {waypoint_idx}{contact_2[8]}")
                initial_collision_check = False
        for contact_3 in contacts_robot2:
            if contact_3[8] < -0.03:  # penetration
                print(f"Collision with co-robot at waypoint {waypoint_idx}{contact_3[8]}")
                initial_collision_check = False
        '''

        contacts_self_raw = bc.getClosestPoints(robotId, robotId, distance=0.001)
        unique_pairs = set()
        for c in contacts_self_raw:
            linkA, linkB, distance = c[3], c[4], c[8]

            if distance >= -1e-5 or linkA == linkB:
                continue

            if {linkA, linkB} == {3, 5}:
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

    if initial_check is True and collision_check is False:
        print("invalid trajectory")

        # sampling method...

    if initial_check is False and collision_check is False:
        print("invalid trajectory and go further backwards")


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
    all_trajectories.append(trajectory)
allConfigTraj_multiple = np.vstack(all_trajectories)
np.savetxt("allConfigTraj_multiple.txt", allConfigTraj_multiple, delimiter=",", fmt="%.6f")

