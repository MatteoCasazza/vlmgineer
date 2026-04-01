from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cube import Cube
import numpy as np
import pybullet as p

class DislodgeCubeEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        # Reset EE Pose
        # xyz: [0.4, 0.0, 0.1]
        # rpy: [0.0, 0.0, 0.0]
        self.reset_qpos = [-0.0025, 0.4313, 0.0010, -2.6514, -0.0067, 3.0826, 0.7901] 
        self.pipe_pos = np.array([1.1, 0.2, 0.005])
        self.start_cube_pos = np.array([1.24, 0.15, 0.05])
        
        # Define pipe dimensions based on pipe.urdf
        self.pipe_length = 0.4
        self.pipe_width = 0.12 + 0.07
        self.pipe_height = 0.12
        
        super().__init__(
            reset_qpos=self.reset_qpos,
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

        pipe_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.pipe = p.loadURDF("pipe/pipe.urdf", self.pipe_pos, pipe_ori, useFixedBase=True)
    
    def reward(self):
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        cube_pos = np.array(cube_pos)

        # near-side boundaries
        pipe_min_x = self.pipe_pos[0] - self.pipe_length/2
        pipe_min_y = self.pipe_pos[1] - self.pipe_width

        max_dx = self.start_cube_pos[0] - pipe_min_x
        dx = cube_pos[0] - pipe_min_x
        reward_x = np.clip(1.0 - dx/max_dx, 0.0, 1.0)

        max_dy = self.start_cube_pos[1] - pipe_min_y
        dy = cube_pos[1] - pipe_min_y
        reward_y = np.clip(1.0 - dy/max_dy, 0.0, 1.0)

        final_reward = max(reward_x, reward_y)
        return final_reward
    
    def reset(self):
        super().reset()
        # Reset cube position
        p.resetBasePositionAndOrientation(
            self.cube, 
            self.start_cube_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )