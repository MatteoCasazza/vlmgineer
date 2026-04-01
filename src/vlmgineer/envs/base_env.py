import pybullet as p
import pybullet_utils.bullet_client as bc
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from abc import ABC, abstractmethod
from pathlib import Path

# Package root directory for finding models
PACKAGE_ROOT = Path(__file__).parent.parent
MODELS_DIR = str(PACKAGE_ROOT / "models")

# Define gripper constants
FINGER_OPEN_POS = 0.04
FINGER_CLOSE_POS = 0.0

class BaseEnv(ABC):
    def __init__(
        self, 
        robot_urdf_path,
        reset_qpos=None,
        stepsize=1.0e-3, 
        realtime=0, 
        video_save_path=None,
        screenshot_save_path=None,
        position_control_gain_p=None,
        position_control_gain_d=None,
        seed=42,
        max_torque=None,
        robot_pos=[0.0, 0.0, 0.0],
        robot_ori=[0, 0, 0],
        gravity=[0, 0, -9.81],
        camera_distance=0.3,
        camera_yaw=40,
        camera_pitch=-35,
        camera_target=[0.9, -0.35, 0.45],
        camera_fov=90.0,
        enable_gui=True,
        physics_client=None,
        enable_gripper=False,
        end_effector_link_name="panda_virtual",
        blender_recorder=None,
        check_self_collision=True,
    ):
        # Set seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize simulation parameters
        self.t = 0.0
        self.stepsize = stepsize
        self.realtime = realtime
        self.gravity = gravity
        self.check_self_collision = check_self_collision
        
        # Robot parameters
        self.robot_urdf_path = robot_urdf_path
        self.robot_pos = robot_pos
        self.robot_ori_euler = robot_ori
        self.robot_ori_quat = p.getQuaternionFromEuler(robot_ori)
        self.enable_gripper = enable_gripper
        self.end_effector_link_name = end_effector_link_name
        self.end_effector_link_index = 7  # Default link index

        self.blender_recorder = blender_recorder
        
        # Camera settings
        self.camera_settings = {
            'distance': camera_distance, 
            'yaw': camera_yaw,
            'pitch': camera_pitch, 
            'target': camera_target
        }
        
        # UI settings
        self.enable_gui = enable_gui
        self.video_save_path = video_save_path
        self.screenshot_save_path = screenshot_save_path
        
        # Store control parameters for initialization after robot loading
        self._init_control_params = {
            'p_gains': position_control_gain_p,
            'd_gains': position_control_gain_d,
            'max_torques': max_torque
        }

        
        if reset_qpos:
            self.reset_qpos = reset_qpos
        else:
            # global reset ee pose:
            # xyz: [0.5, 0.0, 0.5]
            # rpy: [0.0, 0.0, 0.0]
            self.reset_qpos = [-0.0018, -0.1860, 0.0017, -2.1008, 0.0003, 1.9147, 0.7847] 
        
        # Initialize robot state variables
        self._init_robot_state_vars()
        
        # Initialize physics client
        self._init_physics_client(physics_client)
        
        # Configure and reset simulation
        self._setup_simulation()
        
        # Load robot and environment
        self._load_robot()
        self._setup_environment()
        
        # Initialize controller and reset environment
        self.reset()
        
        # Start video recording if path provided
        if self.video_save_path:
            self.physics_client.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, self.video_save_path)
            
        # Take initial screenshot if path provided
        if self.screenshot_save_path:
            os.makedirs(self.screenshot_save_path, exist_ok=True)
            # for _ in range(int(10e2)):
            #     self.set_robot_target_positions(self.reset_qpos)
            #     self.physics_client.stepSimulation()
            self.take_screenshot(os.path.join(self.screenshot_save_path, f"screenshot.png"))

    def _init_robot_state_vars(self):
        """Initialize robot state variables"""
        self.robot = None
        self.joints = []
        self.q_min = []
        self.q_max = []
        self.dof = 0
        self.target_pos = []
        self.finger_joints = []
        self.finger_joint_indices = []
        self.position_control_gain_p = []
        self.position_control_gain_d = []
        self.max_torque = []
        self.link_id_to_name = {}
        self.link_name_to_id = {}
        
        # Gripper state variables
        self.gripper_constraint = None
        self.gripper_opening = True
        self.gripper_target_reached = True
        self.gripper_moving = False
        self.gripper_target_pos = [FINGER_OPEN_POS, FINGER_OPEN_POS]

    def _init_physics_client(self, physics_client):
        """Initialize the PyBullet physics client"""
        if physics_client:
            self.physics_client = physics_client
        else:
            connection_mode = p.GUI if self.enable_gui or self.video_save_path else p.DIRECT
            self.physics_client = bc.BulletClient(connection_mode=connection_mode)

    def _setup_simulation(self):
        """Configure the simulation environment"""
        # Configure visualizer
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=self.camera_settings['distance'],
            cameraYaw=self.camera_settings['yaw'],
            cameraPitch=self.camera_settings['pitch'],
            cameraTargetPosition=self.camera_settings['target']
        )
        
        # Reset simulation parameters
        self.physics_client.resetSimulation()
        self.physics_client.setTimeStep(self.stepsize)
        self.physics_client.setRealTimeSimulation(self.realtime)
        self.physics_client.setGravity(*self.gravity)
        
        # Set additional search path to package models directory
        self.physics_client.setAdditionalSearchPath(MODELS_DIR)

    def _load_robot(self):
        """Load the robot and configure its joints"""
        # Load the robot
        self.robot = self.physics_client.loadURDF(
            self.robot_urdf_path, 
            self.robot_pos, 
            self.robot_ori_quat,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )
        
        # Process joints and setup joint parameters
        self._process_joints()
        
        # Initialize control parameters based on DOF
        self._init_control_parameters()
        
        # Create mapping from link ID to name
        self._create_link_name_mapping()
        
        # Create gripper constraint if gripper is included
        if self.enable_gripper and len(self.finger_joints) == 2:
            self._setup_gripper_constraint()

        if self.blender_recorder:
            self.blender_recorder.register_robot(self.robot)
    
    def _process_joints(self):
        """Process robot joints to extract joint information"""
        joint_info_list = []
        joint_name_to_id = {}
        
        # Panda finger joint names
        finger_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        
        # Collect joint information
        for j in range(self.physics_client.getNumJoints(self.robot)):
            joint_info = self.physics_client.getJointInfo(self.robot, j)
            joint_type = joint_info[2]
            joint_name = joint_info[1].decode("utf-8")
            
            # Skip fixed joints
            if joint_type == p.JOINT_FIXED:
                continue
            
            joint_info_list.append(joint_info)
            joint_name_to_id[joint_name] = joint_info[0]
            
            # Identify finger joints by name
            if joint_name in finger_joint_names:
                self.finger_joints.append(joint_info[0])
        
        # Set up joint parameters
        for idx, joint_info in enumerate(joint_info_list):
            self.joints.append(joint_info[0])
            self.q_min.append(joint_info[8])
            self.q_max.append(joint_info[9])
            
            # Store the index relative to the self.joints list
            if joint_info[0] in self.finger_joints:
                self.finger_joint_indices.append(idx)
        
        # Set DOF to number of non-finger joints (arm joints only)
        self.dof = len(self.joints) - len(self.finger_joints)
    
    def _init_control_parameters(self):
        """Initialize control parameters based on robot DOF"""
        params = self._init_control_params
        
        # Position control P gains
        if params['p_gains'] is not None:
            if len(params['p_gains']) == self.dof:
                self.position_control_gain_p = params['p_gains']
            else:
                raise ValueError(f"Provided position_control_gain_p length ({len(params['p_gains'])}) does not match robot DOF ({self.dof})")
        else:
            self.position_control_gain_p = [0.01] * self.dof

        # Position control D gains
        if params['d_gains'] is not None:
            if len(params['d_gains']) == self.dof:
                self.position_control_gain_d = params['d_gains']
            else:
                raise ValueError(f"Provided position_control_gain_d length ({len(params['d_gains'])}) does not match robot DOF ({self.dof})")
        else:
            self.position_control_gain_d = [1.0] * self.dof
        
        # Maximum torques
        if params['max_torques'] is not None:
            if len(params['max_torques']) == self.dof:
                self.max_torque = params['max_torques']
            else:
                raise ValueError(f"Provided max_torque length ({len(params['max_torques'])}) does not match robot DOF ({self.dof})")
        else:
            self.max_torque = [100.0] * self.dof
    
    def _setup_gripper_constraint(self):
        """Create a constraint between the two finger joints for synchronized movement"""
        if len(self.finger_joints) == 2:
            self.gripper_constraint = self.physics_client.createConstraint(
                self.robot,
                self.finger_joints[0],
                self.robot,
                self.finger_joints[1],
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 0, 1],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )
            self.physics_client.changeConstraint(
                self.gripper_constraint, 
                gearRatio=-1, 
                maxForce=1000, 
                erp=0.2
            )

    def _create_link_name_mapping(self):
        """Create a mapping from link IDs to names for debugging"""
        self.link_id_to_name = {}
        self.link_name_to_id = {}
        for i in range(self.physics_client.getNumJoints(self.robot)):
            joint_info = self.physics_client.getJointInfo(self.robot, i)
            link_id = joint_info[0]
            link_name = joint_info[12].decode("utf-8")
            self.link_id_to_name[link_id] = link_name
            self.link_name_to_id[link_name] = link_id
        
        # Add base link
        self.link_id_to_name[-1] = "base_link"
        self.link_name_to_id["base_link"] = -1
        
        # Set the end effector link index
        if self.end_effector_link_name in self.link_name_to_id:
            self.end_effector_link_index = self.link_name_to_id[self.end_effector_link_name]
            
    @abstractmethod
    def _setup_environment(self):
        """Abstract method for environment-specific assets"""
        pass
    
    @abstractmethod
    def reward(self):
        """Abstract method for reward calculation"""
        pass
    
    def done(self):
        """Abstract method for determining if episode is done"""
        return self.reward() > 0.99

    def result(self):
        """Return the result of the episode in a dictionary"""
        return {
            "reward": float(self.reward()),
            "done": bool(self.done()),
        }
    
    def _check_self_collision(self):
        """Check if the robot is in self-collision, ignoring finger-to-finger collisions"""
        if not self.check_self_collision:
            return False

        # Get all contact points
        contact_points = self.physics_client.getContactPoints(self.robot, self.robot)
        
        # Filter out finger-to-finger collisions
        for contact in contact_points:
            link_a = contact[3]  # First link index
            link_b = contact[4]  # Second link index
            
            # Get link names
            link_a_name = self.link_id_to_name.get(link_a, "")
            link_b_name = self.link_id_to_name.get(link_b, "")
            
            # Check if both links are finger links
            if ("finger" in link_a_name.lower() and "finger" in link_b_name.lower()):
                continue
                
            # If we find any other self-collision, return True
            return True
            
        return False

    def reset(self):
        """Reset the environment"""
        self.t = 0.0        
        
        # Reset robot joint states if reset_qpos is defined
        if hasattr(self, 'reset_qpos') and len(self.reset_qpos) > 0:
            self.target_pos = list(self.reset_qpos)
            
            for j in range(self.dof):
                self.physics_client.resetJointState(self.robot, self.joints[j], targetValue=self.target_pos[j])
            
            # Directly reset finger joint states to open position
            if self.enable_gripper and self.finger_joints:
                for finger_joint in self.finger_joints:
                    self.physics_client.resetJointState(
                        self.robot, 
                        finger_joint, 
                        targetValue=FINGER_OPEN_POS
                    )
                # Update gripper state variables
                self.gripper_opening = True
                self.gripper_target_reached = True
                self.gripper_moving = False
                self.gripper_target_pos = [FINGER_OPEN_POS, FINGER_OPEN_POS]
            
                # Then call open_gripper to ensure proper control setup
                self.open_gripper()
        
        # Reset controller
        self.reset_robot_controller()
        
    def step(self):
        """Step the simulation forward"""
        self.t += self.stepsize
        self.physics_client.stepSimulation()
        if self._check_self_collision():
            raise RuntimeError("Robot is in self-collision, ending simulation!")
        if self.blender_recorder:
            self.blender_recorder.add_keyframe()
    
    def reset_robot_controller(self):
        """Reset the robot controller to zero forces"""
        self.physics_client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.joints)
        )
    
    def set_gripper_state(self, state: int):
        """Control the gripper state.

        Args:
            state: 0 for closed, 1 for open.
        """
        if state == 1:
            self.open_gripper()
        else:
            self.close_gripper()
    
    def open_gripper(self):
        """Open the gripper"""
        if not self.finger_joints:
            print("Warning: Finger joints not identified. Cannot control gripper.")
            return
            
        self.gripper_opening = True
        self.gripper_moving = False
        self.gripper_target_reached = False
        self.gripper_target_pos = [FINGER_OPEN_POS, FINGER_OPEN_POS]
        
        # Apply direct position control with high force
        self.physics_client.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.finger_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.gripper_target_pos,
            forces=[200, 200],  # Increased force
            positionGains=[0.05, 0.05]  # Higher position gain for faster response
        )
    
    def close_gripper(self):
        """Close the gripper"""
        if not self.finger_joints:
            print("Warning: Finger joints not identified. Cannot control gripper.")
            return
            
        # Set state variables
        self.gripper_opening = False
        self.gripper_moving = False
        self.gripper_target_reached = False
        self.gripper_target_pos = [FINGER_CLOSE_POS, FINGER_CLOSE_POS]
        
        # Apply direct position control with high force
        self.physics_client.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.finger_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.gripper_target_pos,
            forces=[200, 200],  # Increased force
            positionGains=[0.05, 0.05]  # Higher position gain for faster response
        )
    
    def set_robot_target_positions(self, target_pos):
        """Set target positions for position control"""
        self.target_pos = target_pos
        self.physics_client.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.target_pos,
            forces=self.max_torque + [200, 200] if self.enable_gripper else self.max_torque,
            positionGains=self.position_control_gain_p + [0.02, 0.02] if self.enable_gripper else self.position_control_gain_p,
            velocityGains=self.position_control_gain_d + [1.0, 1.0] if self.enable_gripper else self.position_control_gain_d
        )
    
    def get_robot_joint_states(self):
        """Get the current joint positions and velocities"""
        joint_states = self.physics_client.getJointStates(self.robot, self.joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        return joint_pos, joint_vel
    
    def get_robot_ee_state(self):
        """Get the position and orientation of the end effector"""
        link_state = self.physics_client.getLinkState(self.robot, self.end_effector_link_index, computeForwardKinematics=True)
        position = link_state[0]  # (x, y, z)
        orientation = link_state[1]  # (x, y, z, w) quaternion
        return position, orientation
    
    def convert_euler_to_quat(self, euler_angles):
        """Convert Euler angles to a quaternion"""
        return p.getQuaternionFromEuler(euler_angles)
    
    def solve_robot_inverse_kinematics(self, pos, ori=None, end_effector_link=None):
        """Solve inverse kinematics"""
        # Use provided end_effector_link if specified, otherwise use the configured end_effector_link_index
        ee_link = end_effector_link if end_effector_link is not None else self.end_effector_link_index
        
        if ori is not None:
            return list(self.physics_client.calculateInverseKinematics(self.robot, ee_link, pos, ori))
        else:
            return list(self.physics_client.calculateInverseKinematics(self.robot, ee_link, pos))
    
    def take_screenshot(self, save_path):
        """Take a screenshot of the current simulation state"""
        # Calculate view matrix
        view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_settings['target'],
            distance=self.camera_settings['distance'],
            yaw=self.camera_settings['yaw'],
            pitch=self.camera_settings['pitch'],
            roll=0,
            upAxisIndex=2  # 2 for Z axis up
        )
        
        # Calculate projection matrix
        projection_matrix = self.physics_client.computeProjectionMatrixFOV(
            fov=90.0,
            aspect=1440/1080,
            nearVal=0.1,
            farVal=100.0
        )
        
        width, height, rgb_img, depth_img, seg_img = self.physics_client.getCameraImage(
            width=1440, height=1080,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
        plt.imsave(save_path, rgb_array.astype(np.uint8))
    
    def update_camera_settings(self, distance=None, yaw=None, pitch=None, target=None):
        """Dynamically update camera parameters during environment rollout
        
        Args:
            distance (float, optional): Camera distance from target
            yaw (float, optional): Camera yaw angle in degrees
            pitch (float, optional): Camera pitch angle in degrees  
            target (list, optional): Camera target position [x, y, z]
        """
        # Update internal camera settings with provided values
        if distance is not None:
            self.camera_settings['distance'] = distance
        if yaw is not None:
            self.camera_settings['yaw'] = yaw
        if pitch is not None:
            self.camera_settings['pitch'] = pitch
        if target is not None:
            self.camera_settings['target'] = target
            
        # Apply the updated settings to the PyBullet visualizer
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=self.camera_settings['distance'],
            cameraYaw=self.camera_settings['yaw'],
            cameraPitch=self.camera_settings['pitch'],
            cameraTargetPosition=self.camera_settings['target']
        )
    
    def get_camera_settings(self):
        """Get the current camera settings
        
        Returns:
            dict: Current camera settings containing distance, yaw, pitch, and target
        """
        return self.camera_settings.copy()
    
    def close(self):
        """Clean up resources"""
        self.physics_client.disconnect()
        