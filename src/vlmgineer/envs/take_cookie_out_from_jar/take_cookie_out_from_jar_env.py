from vlmgineer.envs.base_env import BaseEnv
import numpy as np
import pybullet as p
import random
import os

class TakeCookieOutFromJarEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):
        # Environment parameters
        self.jar_position = np.array([0.58, 0, 0.0])
        self.num_cookies = 12
        self.jar_height_threshold = 0.3  # Target height for jar lifting
        
        # Objects IDs to track
        self.jar_id = None
        self.cookie_ids = []
        
        super().__init__(
            **kwargs,
        )
    
    def _setup_environment(self):
        # Load plane
        planePos = [0, 0, -0.625]
        planeOri = p.getQuaternionFromEuler([0, 0, np.pi/2])
        self.plane = p.loadURDF("plane/plane.urdf", planePos, planeOri, useFixedBase=True)
        
        # Load table
        tablePos = [0.6, 0, -0.625]
        tableOri = p.getQuaternionFromEuler([0, 0, 0])
        self.table = p.loadURDF("table/table.urdf", tablePos, tableOri, useFixedBase=True)
        
        # Initialize jar and cookies in reset method
        
    def _spawn_jar_and_cookies(self):
        """Spawn the jar and cookies"""
        # Remove existing objects if any
        if self.jar_id is not None:
            p.removeBody(self.jar_id)
        
        for cookie_id in self.cookie_ids:
            p.removeBody(cookie_id)
        
        self.cookie_ids = []
        s = 0.0025  # Scale factor for jar and cookies
        # Load jar
        jar_position = self.jar_position
        jar_orientation = p.getQuaternionFromEuler([3.14159265, 0, 0])
        # self.jar_id = p.loadURDF(
        #     "cookie_jar/jar/jar.urdf", 
        #     jar_position, 
        #     jar_orientation, 
        #     useFixedBase=False,
        #     globalScaling=0.0025
        # )
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="cookie_jar/jar/jar.obj",
            meshScale=[s, s, 0.7 * s],
            rgbaColor=[1, 1, 1, 0.3],      # R, G, B, Alpha (0=transparent → 1=opaque)
            specularColor=[0.5, 0.5, 0.5]        # optional: controls highlight shininess
        )

        # (2) Use that visual shape in your MultiBody
        col = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="cookie_jar/jar/jar.obj",
            meshScale=[s, s, 0.7*s],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.jar_id = p.createMultiBody(
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=jar_position,
            baseOrientation=jar_orientation
        )
        # Adjust jar dynamics to make it easier to grasp
        # p.changeDynamics(
        #     self.jar_id, 
        #     -1,  # Base link
        #     lateralFriction=0.9,
        #     spinningFriction=0.1,
        #     rollingFriction=0.1,
        #     restitution=0.1,
        #     mass=0.5,  # Match mass from URDF
        # )
        
        # Spawn cookies inside the jar in circular patterns
        cookies_per_layer = 2  # Number of cookies in each circular layer
        num_layers = (self.num_cookies + cookies_per_layer - 1) // cookies_per_layer
        jar_radius = 0.1  # Approximate radius of jar interior
        cookie_height = 0.03  # Approximate height of each cookie
        
        cookies_spawned = 0
        for layer in range(num_layers):
            # Calculate z-position for this layer
            z_position = jar_position[2] + layer * cookie_height
            
            # Calculate how many cookies for this layer (might be fewer for the last layer)
            layer_cookies = min(cookies_per_layer, self.num_cookies - cookies_spawned)
            
            for i in range(layer_cookies):
                # Calculate position in a circle
                angle = 2 * np.pi * i / cookies_per_layer
                radius = jar_radius * 0.7  # Use 70% of jar radius to ensure cookies fit
                
                x_offset = 0.4* radius * np.cos(angle)+0.02*np.random.rand()
                y_offset = radius * np.sin(angle)+0.02*np.random.rand()
                
                cookie_pos = [
                    jar_position[0] + x_offset,
                    jar_position[1] + y_offset,
                    z_position
                ]
                
                # Consistent orientation for stability, with small random variation
                cookie_ori = p.getQuaternionFromEuler([
                    np.pi/2,  # Small random tilt
                    0,  # Small random tilt
                    angle  # Face according to position in circle
                ])
                
                # Load cookie
                cookie_id = p.loadURDF(
                    "cookie_jar/cookie.urdf",
                    cookie_pos,
                    cookie_ori,
                    useFixedBase=False,
                    globalScaling=0.03
                )
                
                # Add friction to cookie
                p.changeDynamics(
                    cookie_id,
                    -1,
                    lateralFriction=0.8,
                    mass=0.02
                )
                
                self.cookie_ids.append(cookie_id)
                cookies_spawned += 1

    def reward(self):
        """Calculate reward based on jar height"""

        # if any cookie has risen above the jar's opening, it's a win
        jar_opening_z = self.jar_position[2] + self.jar_height_threshold
        highest_cookie_z = 0.0
        for cid in self.cookie_ids:
            cookie_z = p.getBasePositionAndOrientation(cid)[0][2]
            # print(cid, cookie_z)
            if cookie_z > jar_opening_z:
                return 1.0
            else:
                if cookie_z > highest_cookie_z:
                    highest_cookie_z = cookie_z
                    
        return highest_cookie_z/self.jar_height_threshold


    def reset(self):
        """Reset the environment"""
        super().reset()
        
        # Spawn jar and cookies
        self._spawn_jar_and_cookies()
