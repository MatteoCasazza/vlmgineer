from vlmgineer.envs.base_env import BaseEnv
import numpy as np
import pybullet as p

class HockeyEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        self.start_cube_pos = np.array([0.9, 0, 0.035])
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
        
        # Load hockey scene
        scenePos = [0.30, 0.0, 0.0]
        sceneOri = p.getQuaternionFromEuler([0, 0, 0])
        self.goal = p.loadURDF("rlbench_hockey/hockey_goal.urdf", scenePos, sceneOri, useFixedBase=True)
        # self.stick = p.loadURDF("hockey/hockey_stick.urdf", scenePos, sceneOri, useFixedBase=False)
        self.ball = p.loadURDF("rlbench_hockey/hockey_ball.urdf", scenePos, sceneOri, useFixedBase=False)
        self.start_ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        
    def reward(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        goal_pos, _ = p.getBasePositionAndOrientation(self.goal)
        current_distance = np.linalg.norm(np.array(goal_pos) - np.array(ball_pos), ord=1)
        initial_distance = np.linalg.norm(np.array(goal_pos) - self.start_ball_pos, ord=1)
        
        # Normalize reward to be 0 at initial position and 1 at target
        normalized_reward = max(0, 1 - current_distance / initial_distance)
        return normalized_reward
    
    def reset(self):
        super().reset()
        # Reset scene
        # p.resetBasePositionAndOrientation(
        #     self.stick, 
        #     [0.586699, 0.085937, 0.064047], 
        #     p.getQuaternionFromEuler([-0.710316, 3.601585, 1.594259])
        # )
        p.resetBasePositionAndOrientation(
            self.ball, 
            [0.558245, -0.188135, 0.013814], 
            p.getQuaternionFromEuler([3.141591, 3.141592, -1.549853])
        )