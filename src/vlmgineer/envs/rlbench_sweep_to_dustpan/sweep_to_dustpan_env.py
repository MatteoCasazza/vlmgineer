from vlmgineer.envs.base_env import BaseEnv
import numpy as np
import pybullet as p

class SweepToDustpanEnv(BaseEnv):
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
        scenePos = [0.30, 0.0, 0.0]
        dustpanPos = [0.30, 0.0, -0.01]
        sceneOri = p.getQuaternionFromEuler([0, 0, 0])
        self.broom_holder= p.loadURDF("rlbench_sweep_to_dustpan/broom_holder.urdf", scenePos, sceneOri, useFixedBase=True)
        # self.broom = p.loadURDF("sweep_to_dustpan/broom.urdf", scenePos, sceneOri, useFixedBase=False)
        self.dustpan = p.loadURDF("rlbench_sweep_to_dustpan/dustpan.urdf", dustpanPos, sceneOri, useFixedBase=True)
        self.dirt0 = p.loadURDF("rlbench_sweep_to_dustpan/dirt0.urdf", scenePos, sceneOri, useFixedBase=False)
        self.dirt1 = p.loadURDF("rlbench_sweep_to_dustpan/dirt1.urdf", scenePos, sceneOri, useFixedBase=False)
        self.dirt2 = p.loadURDF("rlbench_sweep_to_dustpan/dirt2.urdf", scenePos, sceneOri, useFixedBase=False)
        self.dirt3 = p.loadURDF("rlbench_sweep_to_dustpan/dirt3.urdf", scenePos, sceneOri, useFixedBase=False)
        self.dirt4 = p.loadURDF("rlbench_sweep_to_dustpan/dirt4.urdf", scenePos, sceneOri, useFixedBase=False)
        self.dirt0_pos, _ = p.getBasePositionAndOrientation(self.dirt0)
        self.dirt1_pos, _ = p.getBasePositionAndOrientation(self.dirt1)
        self.dirt2_pos, _ = p.getBasePositionAndOrientation(self.dirt2)
        self.dirt3_pos, _ = p.getBasePositionAndOrientation(self.dirt3)
        self.dirt4_pos, _ = p.getBasePositionAndOrientation(self.dirt4)
        self.goal_pos, _ = p.getBasePositionAndOrientation(self.dustpan)
        
        
    def reward(self):
        dirt0, _ = p.getBasePositionAndOrientation(self.dirt0)
        dirt1, _ = p.getBasePositionAndOrientation(self.dirt1)
        dirt2, _ = p.getBasePositionAndOrientation(self.dirt2)
        dirt3, _ = p.getBasePositionAndOrientation(self.dirt3)
        dirt4, _ = p.getBasePositionAndOrientation(self.dirt4)
        
        current_distance = np.linalg.norm(np.array(self.goal_pos) - np.array(dirt0), ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - np.array(dirt1), ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - np.array(dirt2), ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - np.array(dirt3), ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - np.array(dirt4), ord=1)
        initial_distance = np.linalg.norm(np.array(self.goal_pos) - self.dirt0_pos, ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - self.dirt1_pos, ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - self.dirt2_pos, ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - self.dirt3_pos, ord=1)+\
            np.linalg.norm(np.array(self.goal_pos) - self.dirt4_pos, ord=1)
        
        # Normalize reward to be 0 at initial position and 1 at target
        normalized_reward = max(0, 1 - current_distance / initial_distance)
        return normalized_reward
    
    def reset(self):
        super().reset()
        # Reset scene
        # p.resetBasePositionAndOrientation(
        #     self.broom, 
        #     [0.379309, 0.119323, 0.07206], 
        #     p.getQuaternionFromEuler([3.141299, 3.558271, 1.570976])
        # )
        p.resetBasePositionAndOrientation(
            self.dirt0, 
            [0.60, -0.049998, 0.025004], 
            p.getQuaternionFromEuler([3.141592, 3.141615, 3.141593])
        )
        p.resetBasePositionAndOrientation(
            self.dirt1, 
            [0.575, 2e-06, 0.025004], 
            p.getQuaternionFromEuler([3.141592, 3.141615, 3.141593])
        )
        p.resetBasePositionAndOrientation(
            self.dirt2, 
            [0.55, -0.049998, 0.025004], 
            p.getQuaternionFromEuler([3.141592, 3.141615, 3.141593])
        )
        p.resetBasePositionAndOrientation(
            self.dirt3, 
            [0.60, 2e-06, 0.025004], 
            p.getQuaternionFromEuler([3.141592, 3.141615, 3.141593])
        )
        p.resetBasePositionAndOrientation(
            self.dirt4, 
            [0.55, -0.074998, 0.025004], 
            p.getQuaternionFromEuler([3.141592, 3.141615, 3.141593])
        )