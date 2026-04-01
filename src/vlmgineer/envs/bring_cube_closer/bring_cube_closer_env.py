from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cube import Cube
import numpy as np
import pybullet as p

class BringCubeCloserEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        self.start_cube_pos = np.array([1.0, 0, 0.035])
        self.target_cube_pos = np.array([0.6, 0, 0.035])
        
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
        cubeSize = 0.07
        self.cube = Cube(cubeSize, self.start_cube_pos).get_shape()
        
        # Load visual goal
        self.goal = p.loadURDF("visual_goal/goal.urdf", self.target_cube_pos-np.array([0, 0, 0.035]), useFixedBase=True)

        if self.blender_recorder:
            self.blender_recorder.register_object(self.table, "table")
            self.blender_recorder.register_object(self.cube, "cube")
            self.blender_recorder.register_object(self.goal, "goal")
    
    def reward(self):
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        current_distance = np.linalg.norm(self.target_cube_pos - np.array(cube_pos), ord=1)
        initial_distance = np.linalg.norm(self.target_cube_pos - self.start_cube_pos, ord=1)
        
        # Normalize reward to be 0 at initial position and 1 at target
        normalized_reward = max(0, 1 - current_distance / initial_distance)
        return normalized_reward
    
    def reset(self):
        super().reset()
        # Reset cube position
        p.resetBasePositionAndOrientation(
            self.cube, 
            self.start_cube_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )