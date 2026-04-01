from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cube import Cube
import numpy as np
import pybullet as p

class ClosePlierEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):
        self.cube_size = 0.04
        self.cube_pos = np.array([0.8, 0.0, self.cube_size / 2])
        self.start_plier_pos = self.cube_pos + np.array([-0.08, 0.0, 0.05])
        self.target_plier_pos = self.cube_pos + np.array([0.0, 0.0, 0.3])
        self.start_plier_joint_pos = 0.5
        self.target_plier_joint_pos = 0.0
        self.plierOri = p.getQuaternionFromEuler([0, 0, np.pi-0.15])
        self.reset_qpos = [-0.0018, -0.1860, 0.0017, -2.1008,
                          0.0003,  1.9147,  0.7847]
        super().__init__(
            **kwargs,
        )
    
    def _setup_environment(self):
        # Load plane
        planePos = [0, 0, -0.625]
        planeOri = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.plane = p.loadURDF("plane/plane.urdf", planePos, planeOri, useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=0.95)
        
        # Load table
        tablePos = [0.6, 0, -0.625]
        tableOri = p.getQuaternionFromEuler([0, 0, 0])
        self.table = p.loadURDF("table/table.urdf", tablePos, tableOri, useFixedBase=True)
        
        # Load cube
        self.cube = Cube(self.cube_size, self.cube_pos).get_shape()
        # Set cube color to white
        p.changeVisualShape(self.cube, -1, rgbaColor=[0.3, 0.3, 0.3, 1.0])
        # Set cube mass to 100 kg
        p.changeDynamics(self.cube, -1, mass=1e10)
        
        # Load plier with scaling
        # Position the plier on top of the cube
        self.plier = p.loadURDF(
            "plier2/mobility.urdf", 
            self.start_plier_pos, 
            self.plierOri, 
            useFixedBase=False,
            globalScaling=0.2  # Adjust this value to scale the plier
        )
        if self.plier is None:
            raise RuntimeError("Failed to load plier URDF.")
        
        # Color the plier components
        # Main handle/body - metallic blue
        p.changeVisualShape(self.plier, -1, rgbaColor=[0.2, 0.4, 0.6, 1.0])
        
        # Get the number of joints in the plier
        num_joints = p.getNumJoints(self.plier)
        
        # Color each link/joint differently
        # Red for the grip parts
        for joint_idx in range(num_joints):
            if joint_idx == 1:  # Main movable joint - red
                p.changeVisualShape(self.plier, joint_idx, rgbaColor=[0.8, 0.2, 0.2, 1.0])
            else:  # Other parts - different shade of blue
                p.changeVisualShape(self.plier, joint_idx, rgbaColor=[0.3, 0.5, 0.7, 1.0])
        
        # Initialize joint positions
        p.resetJointState(self.plier, 1, targetValue=self.start_plier_joint_pos)  # Example: set joint_0 to 0.1 radians
        # Add more joints if needed
    
    def reward(self):
        # Get the current joint position
        joint_state = p.getJointState(self.plier, 1)
        current_joint_pos = joint_state[0]
        
        # Calculate joint reward that is 0.0 at start_plier_joint_pos and 1.0 at target_plier_joint_pos
        # Normalize the joint position between start and target
        joint_range = abs(self.start_plier_joint_pos - self.target_plier_joint_pos)
        
        # This will reward progress toward closing the plier (the target joint position)
        joint_reward = max(0, 1.0 - abs(current_joint_pos - self.target_plier_joint_pos) / joint_range)
        
        return joint_reward
    
    def reset(self):
        super().reset()
        # Reset cube position
        p.resetBasePositionAndOrientation(
            self.cube, 
            self.cube_pos,  # Reset to initial cube position
            p.getQuaternionFromEuler([0, 0, 0])
        )
        # Reset plier position
        p.resetBasePositionAndOrientation(
            self.plier, 
            self.start_plier_pos,  # Reset to initial plier position on top of the cube
            self.plierOri
        )
        # Reset joint position
        p.resetJointState(self.plier, 1, targetValue=self.start_plier_joint_pos)  # Set joint_0 to 90 degrees in radians