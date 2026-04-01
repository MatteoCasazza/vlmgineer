from vlmgineer.envs.base_env import BaseEnv
import numpy as np
import pybullet as p

class ScoopWithSpatulaEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
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
        
        # Load scene
        scenePos = [0.35, 0.0, 0.0]
        sceneOri = p.getQuaternionFromEuler([0, 0, 0])
        # self.spatula= p.loadURDF("scoop_with_spatula/spatula.urdf", scenePos, sceneOri, useFixedBase=False)
        self.cube = p.loadURDF("rlbench_scoop_with_spatula/cube.urdf", scenePos, sceneOri, useFixedBase=False)
        self.start_cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        
    def reward(self):
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube)
        goal_pos = np.array(self.start_cube_pos) + np.array([0, 0, 0.1])
        # goal_pos = np.array([0.600000, -0.075000, 0.162])
        current_distance = np.linalg.norm(np.array(goal_pos) - np.array(cube_pos), ord=1)
        initial_distance = np.linalg.norm(np.array(goal_pos) - self.start_cube_pos, ord=1)
        
        # Normalize reward to be 0 at initial position and 1 at target
        normalized_reward = max(0, 1 - current_distance / initial_distance)
        return normalized_reward
    
    def reset(self):
        super().reset()
        # Reset scene
        # p.resetBasePositionAndOrientation(
        #     self.spatula, 
        #     [0.603853, 0.104745, 0.013827], 
        #     p.getQuaternionFromEuler([-3.135589, 4.501131, 1.570820])
        # )
        p.resetBasePositionAndOrientation(
            self.cube, 
            [0.700000, -0.075000, 0.062], 
            p.getQuaternionFromEuler([3.141592, 3.141615, 3.141593])
        )