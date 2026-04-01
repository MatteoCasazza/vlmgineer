from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cube import Cube
import numpy as np
import pybullet as p
import os

class RetrieveHighObjectEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):
        # Shelf position in front of the robot
        self.shelf_pos = np.array([1.3, 0.0, 0.0])
        self.shelf_ori = p.getQuaternionFromEuler([0, 0, np.pi/2.0])
        
        # Wooden box position beside the shelf and between shelf and robot
        self.box_pos = np.array([0.8, 0.0, 0.1])
        self.box_ori = p.getQuaternionFromEuler([0, 0, np.pi/2.0])
        
        # Get wooden box dimensions from the URDF
        # From wooden_box.urdf: Box outer dimensions are 1.20 × 0.38 × 0.10 meters
        self.box_length = 1.20
        self.box_width = 0.38
        self.box_height = 0.1
        
        # Define box region for success detection (inner volume where cube can fall)
        # The bottom panel is positioned 0.0925m below the origin, so the inner bottom is at -0.0925 + box_pos[2]
        half_length = self.box_length/2 - 0.015  # Subtract wall thickness (0.015m)
        half_width = self.box_width/2 - 0.015    # Subtract wall thickness (0.015m)
        
        self.box_min = np.array([
            self.box_pos[0] - half_width,
            self.box_pos[1] - half_length,
            self.box_pos[2] - 0.0925 + 0.015  # Bottom panel is 0.0925m below origin, add thickness
        ])
        
        self.box_max = np.array([
            self.box_pos[0] + half_width,
            self.box_pos[1] + half_length,
            self.box_pos[2] + 0.05  # Walls extend 0.05m up from bottom
        ])
        
        # Cube position on top of the shelf
        self.cube_size = 0.05
        self.cube_start_pos = self.shelf_pos + np.array([-0.1, 0.0, + 0.90 + self.cube_size/2])  # On top of the shelf
        
        # Calculate initial distance from cube to box center
        self.box_center = np.array([
            (self.box_min[0] + self.box_max[0])/2,
            (self.box_min[1] + self.box_max[1])/2,
            (self.box_min[2] + self.box_max[2])/2
        ])
        
        super().__init__(
            camera_distance=0.5,
            camera_yaw=-30,
            camera_pitch=-30,
            camera_target=[0.3, -0.5, 0.7],
            **kwargs,
        )
    
    def _setup_environment(self):
        # Load plane
        planePos = [0, 0, -0.01]
        planeOri = p.getQuaternionFromEuler([0, 0, 0])
        self.plane = p.loadURDF("plane/plane.urdf", planePos, planeOri, useFixedBase=True)
        
        # Load shelf
        shelf_urdf_path = os.path.join("shelf", "shelf.urdf")
        self.shelf = p.loadURDF(shelf_urdf_path, self.shelf_pos, self.shelf_ori, useFixedBase=True)
        
        # Load wooden box, which is the cube holder and should be placed beside the shelf
        box_urdf_path = os.path.join("box", "wooden_box.urdf")
        self.box = p.loadURDF(box_urdf_path, self.box_pos, self.box_ori, useFixedBase=True)
        
        # Load cube and put cube on the top of the shelf
        self.cube = Cube(self.cube_size, self.cube_start_pos, mass=0.1, color=[0, 1, 0, 1]).get_shape()
        
        # Create a visual marker for the goal box region (semi-transparent)
        # self._create_goal_box_visual()
    
    def _create_goal_box_visual(self):
        """Create a visual marker for the reward box region"""
        box_center = self.box_pos
        
        box_half_extents = [
            (self.box_max[0] - self.box_min[0])/2,  # x half length
            (self.box_max[1] - self.box_min[1])/2,  # y half width
            (self.box_max[2] - self.box_min[2])/2   # z half height
        ]
        
        # Create visual shape for reward region (semi-transparent green)
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=box_half_extents,
            rgbaColor=[0, 1, 0, 0.3]  # Semi-transparent green
        )
        
        # Create multibody for visualization (no physics, just visual)
        self.goal_box_visual = p.createMultiBody(
            baseMass=0,  # Zero mass means it's static
            baseVisualShapeIndex=visual_id,
            basePosition=box_center,
            baseCollisionShapeIndex=-1  # No collision shape
        )
    
    def reward(self):
        # Get current cube position
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        cube_pos = np.array(cube_pos)
        
        # Check if cube is in the wooden box
        in_box = (
            self.box_min[0] <= cube_pos[0] <= self.box_max[0] and
            self.box_min[1] <= cube_pos[1] <= self.box_max[1] and
            self.box_min[2] <= cube_pos[2] <= self.box_max[2]
        )
        
        if in_box:
            reward = 1.0  # Maximum reward when the cube is in the box
        else:
            # Calculate current distance to box center
            current_cube_to_box_dist = np.linalg.norm(cube_pos - self.box_center, ord=1)
            initial_cube_to_box_dist = np.linalg.norm(self.cube_start_pos - self.box_center, ord=1)
            
            # Calculate normalized distance ratio (1.0 when at start, approaches 0 as cube gets closer to box)
            distance_ratio = current_cube_to_box_dist / initial_cube_to_box_dist
            
            # Reward is 1.0 - distance_ratio, so it's normalized between 0 and 1
            # 1.0 when cube is at start position, approaches 1.0 as cube gets closer to box
            reward = max(0.0, 1.0 - distance_ratio)
            
            # Add bonus for getting cube off the shelf (below shelf height)
            if cube_pos[2] < 0.85:
                reward = min(1.0, reward + 0.2)  # Add 0.2 bonus but cap at 1.0
        
        return reward
    
    def reset(self):
        super().reset()
        # Reset cube position
        p.resetBasePositionAndOrientation(
            self.cube, 
            self.cube_start_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
    def is_success(self):
        """Check if the cube is in the wooden box"""
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        cube_pos = np.array(cube_pos)
        
        in_box = (
            self.box_min[0] <= cube_pos[0] <= self.box_max[0] and
            self.box_min[1] <= cube_pos[1] <= self.box_max[1] and
            self.box_min[2] <= cube_pos[2] <= self.box_max[2]
        )
        
        return in_box