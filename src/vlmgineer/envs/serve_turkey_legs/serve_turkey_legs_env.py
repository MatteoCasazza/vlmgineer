from vlmgineer.envs.base_env import BaseEnv
import numpy as np
import pybullet as p
import random
import os

class ServeTurkeyLegsEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        # Reset EE Pose
        # xyz: [0.4, -0.2, 0.2]
        # rpy: [0.0, 0.0, 0.0]
        self.reset_qpos = [-0.2984, 0.1758, -0.1533, -2.5745, 0.0695, 2.7474, 0.2718, 0.04, 0.04]

        # Environment parameters
        self.num_turkey_legs = 5  # Number of turkey legs to spawn
        self.turkey_leg_scale = 0.05  # Scale factor for turkey legs
        self.pot_height_threshold = 0.2  # Target height for lifting the pot
        self.pot_position = np.array([0.75, -0.2, 0.08])  # Initial pot position 
        
        # Chef box parameters
        self.chef_box_position = np.array([0.8, 0.25, 0])  # Position on the left side of table
        self.initial_pot_to_box_distance = np.linalg.norm(self.chef_box_position - self.pot_position, ord=1)
        
        # Pot dimensions (from URDF)
        self.pot_radius = 0.12  # Pot radius from URDF
        self.pot_height = 0.0875  # Height of pot walls from URDF (reduced by 50% from 0.175)
        
        # Reward parameters
        self.height_reward_weight = 0.6  # Weight for height reward component
        self.containment_reward_weight = 0.4  # Weight for containment reward component
        
        # Will be populated after environment setup
        self.turkey_leg_ids = []
        self.pot_id = None
        
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
        
        # Load pot
        potPos = self.pot_position
        potOri = p.getQuaternionFromEuler([0, 0, 0])
        
        # Get the path to the pot URDF
        self.pot_id = p.loadURDF("pot/pot.urdf", potPos, potOri, useFixedBase=False)
        
        # Load chef box
        chef_box_pos = self.chef_box_position
        chef_box_ori = p.getQuaternionFromEuler([0, 0, np.pi/2])  # Rotate 90 degrees
        self.chef_box_id = p.loadURDF("chef_box/chef_box.urdf", chef_box_pos, chef_box_ori, useFixedBase=True)
        
        # Store the initial height of the pot bottom for reference
        pot_pos, _ = p.getBasePositionAndOrientation(self.pot_id)
        
        # Adjust pot dynamics for better stability
        p.changeDynamics(self.pot_id, -1, 
                         lateralFriction=0.5,  # Reduced from 0.9
                         spinningFriction=0.05,  # Reduced from 0.1
                         rollingFriction=0.05,  # Reduced from 0.1
                         mass=1.0)  # Set appropriate mass for the pot
    
    def _spawn_turkey_legs(self):
        """Spawn turkey legs in a grid pattern within the pot"""
        self.turkey_leg_ids = []
        
        # Get pot dimensions with safety margin
        pot_radius = self.pot_radius * 0.8  # Slightly larger safety margin
        
        # Get current pot position
        pot_pos, _ = p.getBasePositionAndOrientation(self.pot_id)
        
        # Maximum random offset
        max_random_offset = 0.05
        
        # Place turkey legs in a grid pattern
        grid_size = int(np.sqrt(self.num_turkey_legs))  # Calculate grid size
        spacing = (pot_radius * 2) / (grid_size + 1)  # Calculate spacing between legs
        
        for i in range(self.num_turkey_legs):
            # Calculate grid position
            row = i // grid_size
            col = i % grid_size
            
            # Calculate base position in grid
            x_pos = pot_pos[0] - pot_radius + spacing * (col + 1)
            y_pos = pot_pos[1] - pot_radius + spacing * (row + 1)
            z_pos = pot_pos[2] + 0.05  # Start slightly above pot bottom
            
            # Add small random offset
            x_pos += random.uniform(-max_random_offset * 0.5, max_random_offset * 0.5)
            y_pos += random.uniform(-max_random_offset * 0.5, max_random_offset * 0.5)
            z_pos += random.uniform(0, max_random_offset * 0.5)
            
            # Fixed downward orientation with slight random variation
            ori = p.getQuaternionFromEuler([
                np.pi + random.uniform(-0.3, 0.3),  # More variation around straight down
                random.uniform(-0.3, 0.3),  # More roll variation
                random.uniform(-0.3, 0.3)   # More yaw variation
            ])
            
            # Load turkey leg
            turkey_leg_id = p.loadURDF(
                "turkey_leg/turkey_leg.urdf",
                [x_pos, y_pos, z_pos],
                ori,
                globalScaling=self.turkey_leg_scale
            )
            
            # Adjust dynamics for better stability with less friction
            p.changeDynamics(turkey_leg_id, -1,
                           lateralFriction=0.2,  # Reduced from 0.3
                           spinningFriction=0.02,  # Reduced from 0.05
                           rollingFriction=0.02,  # Reduced from 0.05
                           mass=0.6)  # Set appropriate mass for turkey leg
            
            self.turkey_leg_ids.append(turkey_leg_id)
            
            # Let physics stabilize every few legs
            if i % 2 == 0:
                for _ in range(10):
                    p.stepSimulation()
    
    def _transform_point_to_box_frame(self, point, box_pos, box_ori):
        """Transform a point from world coordinates to box-relative coordinates."""
        # Convert quaternion to rotation matrix
        rot_matrix = np.array(p.getMatrixFromQuaternion(box_ori)).reshape(3, 3)
        
        # Convert tuples to numpy arrays before subtraction
        point_array = np.array(point)
        box_pos_array = np.array(box_pos)
        
        # Translate point to box frame
        translated = point_array - box_pos_array
        
        # Rotate point to box frame
        rotated = np.dot(rot_matrix.T, translated)
        
        return rotated
    
    def _is_point_in_chef_box(self, point):
        """Check if a point is inside the chef box boundaries."""
        # Chef box dimensions from URDF
        box_width = 0.265  # x dimension
        box_length = 0.525  # y dimension
        box_height = 0.185  # z dimension (including lip)
        
        # Get chef box position and orientation
        box_pos, box_ori = p.getBasePositionAndOrientation(self.chef_box_id)
        
        # Transform point to box frame
        point_in_box_frame = self._transform_point_to_box_frame(point, box_pos, box_ori)
        
        # Check if point is within box bounds
        x, y, z = point_in_box_frame
        return (abs(x) <= box_width/2 and 
                abs(y) <= box_length/2 and 
                abs(z) <= box_height/2)

    def reward(self):
        """Calculate reward based on turkey legs in box and pot position."""
        # Get chef box position and pot position
        chef_box_pos, _ = p.getBasePositionAndOrientation(self.chef_box_id)
        pot_pos, _ = p.getBasePositionAndOrientation(self.pot_id)
        
        # Check if pot is inside chef box
        if self._is_point_in_chef_box(pot_pos):
            pot_reward = 0.7  # Penalize 30% if pot is inside chef box
        else:
            pot_reward = 1.0  # No penalty if pot is outside chef box
        
        # Calculate individual rewards for each turkey leg
        turkey_leg_rewards = []
        
        for turkey_leg_id in self.turkey_leg_ids:
            turkey_leg_pos, _ = p.getBasePositionAndOrientation(turkey_leg_id)
            turkey_leg_center = np.array(turkey_leg_pos)
            
            # Check if turkey leg center is in chef box
            if self._is_point_in_chef_box(turkey_leg_center):
                turkey_leg_rewards.append(1.0)  # Full reward for turkey legs in box
            else:
                turkey_leg_rewards.append(0.0)  # No reward for turkey legs outside box
        
        # Calculate mean turkey leg reward
        turkey_leg_reward = np.mean(turkey_leg_rewards)
        
        # Combine rewards (equal weight for turkey legs and pot position)
        total_reward = turkey_leg_reward * pot_reward
        
        return total_reward
    
    def reset(self):
        """Reset the environment"""
        super().reset()
        
        # Reset all turkey legs
        for turkey_leg_id in self.turkey_leg_ids:
            p.removeBody(turkey_leg_id)
        
        # Reset pot position
        if self.pot_id is not None:
            p.resetBasePositionAndOrientation(
                self.pot_id, 
                self.pot_position, 
                p.getQuaternionFromEuler([0, 0, 0])
            )
        
        # Spawn new turkey legs
        self._spawn_turkey_legs()
        
        # Let physics stabilize
        for _ in range(100):
            p.stepSimulation() 