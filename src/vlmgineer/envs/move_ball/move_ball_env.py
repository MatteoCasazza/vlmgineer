from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.sphere import Sphere
import numpy as np
import pybullet as p

class MoveBallEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        self.start_ball_pos = np.array([0.6, 0.3, 0.08])  # Starting position of the ball
        self.curr_ball_pose = np.array([0.6, 0.3, 0.08])
        self.ball_vel = np.zeros(3)
        self.target_ball_pos = np.array([0.6, -0.3, 0.001])  # Target position of the ball
        
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
        
        # Load sphere
        sphere_radius = 0.08
        self.ball = Sphere(sphere_radius, self.start_ball_pos).get_shape()

        self.goal = p.loadURDF("visual_goal/goal.urdf", self.target_ball_pos, useFixedBase=True)
    
    def reward(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        ball_target_dist = np.linalg.norm(self.target_ball_pos - np.array(ball_pos), ord=1)
        max_distance = np.linalg.norm(self.target_ball_pos - self.start_ball_pos, ord=1)
        ball_speed = np.linalg.norm(self.ball_vel)

        return max(0.0, 1.0 - (ball_target_dist / max_distance) - ball_speed)  # Clip to [0, 1]
    
    def step(self):
        super().step()
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        ball_pos = np.array(ball_pos)
        pos_diff = ball_pos - self.curr_ball_pose
        self.ball_vel = pos_diff / self.stepsize
        self.curr_ball_pose = ball_pos
    
    def reset(self):
        super().reset()
        # Reset ball position
        p.resetBasePositionAndOrientation(
            self.ball, 
            self.start_ball_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )
