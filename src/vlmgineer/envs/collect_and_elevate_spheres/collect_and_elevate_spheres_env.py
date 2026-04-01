from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.sphere import Sphere
import numpy as np
import pybullet as p
import random

class CollectAndElevateSpheresEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        self.num_spheres = 30
        self.sphere_radius = 0.02
        self.height_threshold = 0.3
        self.num_sphere_target = 4
        self.container_width = 0.3
        self.container_position = np.array([0.7, 0, 0])
        
        # Will be populated after environment setup
        self.sphere_ids = []
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
        
        ### Spawn spheres inside the container: moved to reset! ###
    
    def _spawn_spheres(self):
        """Spawn multiple spheres inside the container using a grid approach with randomization"""
        self.sphere_ids = []
        
        # Get container dimensions
        safe_distance = self.sphere_radius * 2.2 # Ensure spheres don't overlap
        
        # Calculate how many spheres can fit in each row/column
        spheres_per_side = int(self.container_width / safe_distance)
        
        # Adjust if we need multiple layers
        spheres_per_layer = spheres_per_side * spheres_per_side
        num_layers = (self.num_spheres + spheres_per_layer - 1) // spheres_per_layer
        
        # Maximum random offset (keep small to avoid collisions)
        max_random_offset = self.sphere_radius * 0.2
        
        # Place spheres in grid
        count = 0
        for layer in range(num_layers):
            for col in range(spheres_per_side):
                for row in range(spheres_per_side):
                    if count >= self.num_spheres:
                        break
                    
                    # Calculate grid position
                    x = (col - (spheres_per_side-1)/2) * safe_distance
                    y = (row - (spheres_per_side-1)/2) * safe_distance
                    z = self.sphere_radius + layer * safe_distance
                    
                    # Add random offset for natural look
                    x += random.uniform(-max_random_offset, max_random_offset)
                    y += random.uniform(-max_random_offset, max_random_offset)
                    z += random.uniform(0, max_random_offset)
                    
                    # Place sphere relative to container position
                    sphere_pos = np.array([
                        self.container_position[0] + x,
                        self.container_position[1] + y,
                        self.container_position[2] + z
                    ])
                    
                    # Create sphere
                    r, g, b = random.random()*0.5+0.5, random.random()*0.5+0.5, random.random()*0.5+0.5
                    sphere = Sphere(self.sphere_radius, sphere_pos, mass=0.005, color=np.array([r*255, g*255, b*255, 255])/255).get_shape()
                    self.sphere_ids.append(sphere)
                    count += 1
                
                if count >= self.num_spheres:
                    break
            
            if count >= self.num_spheres:
                break
        
    
    def reward(self):
        """Calculate normalized reward based on the total height of all spheres.
        - Reward is normalized between 0 and 1.
        - Each sphere's height contribution is capped at height_threshold to prevent reward inflation.
        """
        total_height = 0
        
        for sphere_id in self.sphere_ids:
            sphere_pos, _ = p.getBasePositionAndOrientation(sphere_id)
            # Cap height contribution at height_threshold
            height_contribution = min(max(0, sphere_pos[2]), self.height_threshold)
            total_height += height_contribution
            
        # Calculate the maximum possible reward (target height)
        max_possible_reward = self.num_spheres * self.height_threshold
            
        # Normalize the reward (will now strictly be between 0 and 1)
        normalized_reward = total_height / max_possible_reward
        
        return normalized_reward
    
    def reset(self):
        """Reset the environment"""
        super().reset()
        
        # Reset all spheres
        for sphere_id in self.sphere_ids:
            p.removeBody(sphere_id)
        
        # Spawn new spheres
        self._spawn_spheres()