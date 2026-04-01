from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cube import Cube
import numpy as np
import pybullet as p

class PlaceCubesInsideEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        # Cube positions on the table
        self.cube1_start_pos = np.array([0.6, -0.1, 0.035])
        self.cube2_start_pos = np.array([0.6, 0.1, 0.035])
        
        # Box position at the end of the table
        self.box_pos = np.array([1.0, 0, 0.1])

        # Box boundary will be dynamically read from URDF
        self.box_half_extents = None
        
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
        
        # Load two cubes
        cubeSize = 0.05
        self.cube1 = Cube(cubeSize, self.cube1_start_pos, color=[0.78, 0.08, 0.52, 1]).get_shape()
        self.cube2 = Cube(cubeSize, self.cube2_start_pos, color=[0.78, 0.08, 0.52, 1]).get_shape()
        
        # Load box at the end of the table
        boxOri = p.getQuaternionFromEuler([0, 0, 0])
        self.box = p.loadURDF("models/real_box/real_box.urdf", self.box_pos, boxOri, useFixedBase=True)

        # Get box boundary dimensions from URDF
        self._get_box_dimensions()

    def _get_box_dimensions(self):
        """Retrieve box boundary dimensions from the URDF model"""
        # Get visual shape data to extract dimensions
        visual_data = p.getVisualShapeData(self.box)
        
        for i, data in enumerate(visual_data):
            geometry_type = data[2]  # 2 = GEOM_BOX
            if geometry_type == p.GEOM_BOX:
                # For box geometries, data[3] contains the half-extents
                self.box_extents = np.array(data[3])
                break
        
    def is_cube_in_box(self, cube_pos, box_pos):
        """Check if a cube is inside the box boundary"""
        # Get relative position of the cube to the box
        rel_pos = np.array(cube_pos) - np.array(box_pos)
        
        # Check if the cube is within the box boundaries
        return (abs(rel_pos[0]) < self.box_extents[0]/2 and
                abs(rel_pos[1]) < self.box_extents[1]/2 and
                abs(rel_pos[2]) < self.box_extents[2]/2)
    
    def reward(self):
        """
        Calculate reward based on whether both cubes are in the box.
        Returns a value normalized between 0 and 1, where:
        - 0: both cubes are far from the box
        - 1: both cubes are inside the box
        """
        # Get positions of both cubes
        cube1_pos, _ = p.getBasePositionAndOrientation(self.cube1)
        cube2_pos, _ = p.getBasePositionAndOrientation(self.cube2)
        
        # Check if cubes are in the box using boundary dimensions
        cube1_in_box = self.is_cube_in_box(cube1_pos, self.box_pos)
        cube2_in_box = self.is_cube_in_box(cube2_pos, self.box_pos)
        
        # If cubes are not in the box, calculate distance-based partial reward
        if not cube1_in_box:
            # Calculate distance from the cube to the box center
            dist1 = np.linalg.norm(np.array(cube1_pos) - np.array(self.box_pos))
            cube1_reward = max(0, 1.0 - dist1 / 0.5)  # Normalize distance
        else:
            cube1_reward = 1.0
            
        if not cube2_in_box:
            # Calculate distance from the cube to the box center
            dist2 = np.linalg.norm(np.array(cube2_pos) - np.array(self.box_pos))
            cube2_reward = max(0, 1.0 - dist2 / 0.5)  # Normalize distance
        else:
            cube2_reward = 1.0
        
        # Final reward combines how close both cubes are to being in the box
        return 0.5 * (cube1_reward + cube2_reward)
    
    def reset(self):
        super().reset()
        # Reset cube positions
        p.resetBasePositionAndOrientation(
            self.cube1, 
            self.cube1_start_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        p.resetBasePositionAndOrientation(
            self.cube2, 
            self.cube2_start_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        ) 