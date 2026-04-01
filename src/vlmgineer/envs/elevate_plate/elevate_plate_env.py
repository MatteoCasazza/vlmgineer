from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cylinder import Cylinder
import numpy as np
import pybullet as p

class ElevatePlateEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):
        self.start_plate_pos = np.array([0.75, 0.0, 0.0])
        self.target_plate_pos = np.array([0.75, 0.0, 0.4])
        
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
        
        # Load plate
        plateOri = p.getQuaternionFromEuler([0, 0, 0])
        self.plate = p.loadURDF("plate/plate.urdf", self.start_plate_pos, plateOri, useFixedBase=False)

    def reward(self):
        plate_pos, _ = p.getBasePositionAndOrientation(self.plate)
        distance = np.linalg.norm(self.target_plate_pos - np.array(plate_pos), ord=1)
        max_distance = np.linalg.norm(self.target_plate_pos - self.start_plate_pos, ord=1)
        return max(0.0, 1.0 - (distance / max_distance))  # Clip to [0, 1]
    
    def reset(self):
        super().reset()
        # Reset plate position
        p.resetBasePositionAndOrientation(
            self.plate, 
            self.start_plate_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )
