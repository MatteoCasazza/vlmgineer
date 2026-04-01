from vlmgineer.envs.base_env import BaseEnv
import numpy as np
import pybullet as p

class BringScreenDownEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):
        self.start_screen_pos = [0.35, 0, 1.1]
        self.target_screen_pos = [0.55, 0.0, 0.4]
        
        super().__init__(
            **kwargs,
        )
    
    def _setup_environment(self):
        # Set gravity to make the screen go upwards
        p.setGravity(0, 0, 9.81)  # Reverse gravity by setting a positive value in the z-direction
        
        # Load plane (floor)
        planePos = [0, 0, 0]
        planeOri = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.plane = p.loadURDF("plane/plane.urdf", planePos, planeOri, useFixedBase=True)
        p.changeDynamics(self.plane, -1, restitution=0.95)
        p.changeVisualShape(self.plane, -1, rgbaColor=[0.2, 0.2, 0.2, 1])  # Set color to grey
        
        # ceiling height
        ceiling_z = 1.2  # whatever Z you want your ceiling at

        # flip the plane so its "up" is actually down
        ceiling_ori = p.getQuaternionFromEuler([np.pi, 0, 0])

        # load a second plane as ceiling
        self.ceiling = p.loadURDF(
            "plane/plane.urdf",
            [0, 0, ceiling_z],
            ceiling_ori,
            useFixedBase=True
        )
        # give it the same bounciness / color if you like
        p.changeDynamics(self.ceiling, -1, restitution=0.95)
        p.changeVisualShape(self.ceiling, -1, rgbaColor=[0.9, 0.9, 0.9, 1])

        # Load screen
        screenOri = p.getQuaternionFromEuler([1.5708, 1.5708, 0])
        self.screen = p.loadURDF("screen/screen.urdf", self.start_screen_pos, screenOri, useFixedBase=False, globalScaling=0.5)
        
    
    def reward(self):
        screen_pos, _ = p.getBasePositionAndOrientation(self.screen)
        distance = np.linalg.norm(self.target_screen_pos - np.array(screen_pos), ord=1)
        return np.exp(-distance) # Normalize reward between 0 and 1
    
    def reset(self):
        super().reset()
        # Reset screen position
        p.resetBasePositionAndOrientation(
            self.screen, 
            self.start_screen_pos, 
            p.getQuaternionFromEuler([1.5708, 1.5708, 0])
        )
