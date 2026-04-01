from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.sphere import Sphere
import numpy as np
import pybullet as p
import random

class GatherStrawberriesEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        self.object_name = "strawberry"
        self.num_objects = 30
        self.object_radius = 0.04
        self.height_threshold = 0.3
        self.num_object_target = 4
        self.container_width = 0.3
        self.container_position = np.array([0.7, 0, 0])
        
        # Will be populated after environment setup
        self.object_ids = []
        self.container_id = None
        
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
        
        # Load container
        containerPos = self.container_position
        containerOri = p.getQuaternionFromEuler([0, 0, 0])
        self.container_id = p.loadURDF("container/container.urdf", containerPos, containerOri, useFixedBase=True)

        num_links = p.getNumJoints(self.container_id)
        
        # Change dynamics for each link
        for link_idx in range(num_links):
            p.changeDynamics(self.container_id, link_idx, 
                            lateralFriction=0.001,
                            spinningFriction=0.001,
                            rollingFriction=0.001,
                            linearDamping=0.001,
                            angularDamping=0.001)
        
        ### Spawn objects inside the container: moved to reset! ###
    
    def _spawn_objects(self):
        """Spawn multiple objects inside the container using a grid approach with randomization"""
        self.object_ids = []
        
        # Get container dimensions
        safe_distance = self.object_radius * 2.2 # Ensure objects don't overlap
        
        # Calculate how many objects can fit in each row/column
        objects_per_side = int(self.container_width / safe_distance)
        
        # Adjust if we need multiple layers
        objects_per_layer = objects_per_side * objects_per_side
        num_layers = (self.num_objects + objects_per_layer - 1) // objects_per_layer
        
        # Maximum random offset (keep small to avoid collisions)
        max_random_offset = self.object_radius * 0.2
        
        # Place objects in grid
        count = 0
        for layer in range(num_layers):
            for col in range(objects_per_side):
                for row in range(objects_per_side):
                    if count >= self.num_objects:
                        break
                    
                    # Calculate grid position
                    x = (col - (objects_per_side-1)/2) * safe_distance
                    y = (row - (objects_per_side-1)/2) * safe_distance
                    z = self.object_radius + layer * safe_distance
                    
                    # Add random offset for natural look
                    x += random.uniform(-max_random_offset, max_random_offset)
                    y += random.uniform(-max_random_offset, max_random_offset)
                    z += random.uniform(0, max_random_offset)
                    
                    # Place object relative to container position
                    object_pos = np.array([
                        self.container_position[0] + x,
                        self.container_position[1] + y,
                        self.container_position[2] + z
                    ])
                    
                    # Create object
                    object = p.loadURDF(f"{self.object_name}/{self.object_name}.urdf", object_pos)
                    self.object_ids.append(object)
                    count += 1
                
                if count >= self.num_objects:
                    break
            
            if count >= self.num_objects:
                break
        
    
    def reward(self):
        """Calculate normalized reward based on the total height of all objects.
        - Reward is normalized between 0 and 1.
        - Normalization is done by dividing the current total height by the target total height (all objects at height_threshold).
        """
        total_height = 0
        
        for object_id in self.object_ids:
            object_pos, _ = p.getBasePositionAndOrientation(object_id)
            height_contribution = max(0, object_pos[2]) 
            total_height += height_contribution
            
        # Calculate the maximum possible reward (target height)
        max_possible_reward = self.num_objects * self.height_threshold
            
        # Normalize the reward
        normalized_reward = total_height / max_possible_reward
        
        return normalized_reward
    
    def reset(self):
        """Reset the environment"""
        super().reset()
        
        # Reset all objects
        for object_id in self.object_ids:
            p.removeBody(object_id)
        
        # Spawn new objects
        self._spawn_objects()