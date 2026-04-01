from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cube import Cube
import numpy as np
import pybullet as p
import os

class TakeOneBookOutEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):  
        # Book dimensions (approximate from standard books)
        self.est_book_width = 0.03  # meters
        self.real_book_width = 0.04  # meters
        self.book_height = 0.28  # meters
        self.book_depth = 0.088  # meters
        
        # Positions
        self.books_base_pos = [0.6, 0.0, 0.20]  # Center position on table
        self.books_high_zaxis = 0.18
        self.books_low_zaxis = 0.10
        
        # Define positions for books (will be set in setup)
        self.books_positions = []
        
        # Target position for book 3 (where we want to pull it to)
        self.target_book3_pos = np.array([0.4, 0, 0.04])
        
        # Success criteria
        self.success_distance = np.sqrt((self.book_height/2)**2 + (self.book_depth/2)**2)  # Distance threshold for success
        
        # Create a visual marker for the target position
        self.visual_marker_id = None
        
        super().__init__(
            camera_distance=0.5,
            camera_yaw=0,
            camera_pitch=-35,
            camera_target=[0.6, -0.4, 0.5],
            **kwargs,
        )
        
        self.ini_book_poses = []
        for i in range(5):
            init_book_pos, _ = p.getBasePositionAndOrientation(self.books[i])
            self.ini_book_poses.append(np.array(init_book_pos))
            
        self._create_target_marker()
            
    
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
        
        # Calculate positions for books
        total_books_width = 5 * self.real_book_width
        start_y = self.books_base_pos[1] - total_books_width/2 + self.real_book_width/2

        for i in range(5):
            y = start_y + i * self.real_book_width
            if i >=3:
                self.books_positions.append([self.books_base_pos[0], y , self.books_high_zaxis])
            else:
                self.books_positions.append([self.books_base_pos[0], y , self.books_low_zaxis])
        
        # Load left book holder
        left_holder_pos = [self.books_base_pos[0], start_y, 0.08]
        left_holder_ori = p.getQuaternionFromEuler([0, 0, -np.pi/2])  # Rotate to hold books
        self.left_holder = p.loadURDF("book_holder/model.urdf", left_holder_pos, left_holder_ori, useFixedBase=True)
        
        # Load books 1-5
        self.books = []
        for i in range(5):
            book_path = f"books/book_{i+1}/model.urdf"
            book_pos = self.books_positions[i]
            book_ori = p.getQuaternionFromEuler([0, 0, 0])  # Books standing upright
            book = p.loadURDF(book_path, book_pos, book_ori, useFixedBase=False)
            self.books.append(book)
        
        
        # Load right book holder
        right_holder_pos = [self.books_base_pos[0], start_y + 5 * self.est_book_width, 0.08]
        right_holder_ori = p.getQuaternionFromEuler([0, 0, np.pi/2])  # Rotate to hold books
        self.right_holder = p.loadURDF("book_holder/model.urdf", right_holder_pos, right_holder_ori, useFixedBase=True)
        
        # Create visual marker for target position (where to pull book 3)
    
    def _create_target_marker(self):
        """Create a visual marker for the target position"""
        book3_pos, _ = p.getBasePositionAndOrientation(self.books[2])
        book3_pos = np.array(book3_pos)
        self.target_book3_pos = book3_pos
        self.target_book3_pos[1] = self.target_book3_pos[1] + self.real_book_width/2
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.book_height/2, self.real_book_width/2, self.book_depth],
            rgbaColor=[0, 1, 0, 0.3]  # Semi-transparent green
        )
        
        self.visual_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=self.target_book3_pos,
            baseCollisionShapeIndex=-1  # No collision
        )
    
    def reward(self):
        # Get the position of book 3 (the middle book)
        book3_pos, _ = p.getBasePositionAndOrientation(self.books[2])
        book3_pos = np.array(book3_pos)
        # Get robot end effector position
        ee_state = self.get_robot_ee_state()
        ee_pos = np.array(ee_state[0])
        # Calculate distances
        ee_to_book_dist = np.linalg.norm(ee_pos - book3_pos)
        book_pulled_dist = np.linalg.norm(book3_pos - self.ini_book_poses[2])
        # Check if book 3 has been pulled out
        is_pulled_out = book_pulled_dist > self.success_distance
        # Other books movement
        other_books_move_dist = 0
        for i in range(5):
            if i != 2:
                other_book_pos, _ = p.getBasePositionAndOrientation(self.books[i])
                other_book_pos = np.array(other_book_pos)
                other_books_move_dist += np.linalg.norm(other_book_pos - self.ini_book_poses[i])
        # Calculate reward
        if is_pulled_out:
            # High reward for pulling the book out
            reward = np.exp(-other_books_move_dist)
        else:
            # Encourage getting end effector to the book
            reward = book_pulled_dist/self.success_distance - (-np.exp(-other_books_move_dist)+1)
        reward = np.clip(reward, 0.0, 1.0)
        return reward
    
    def reset(self):
        super().reset()
        # Reset books positions
        for i, book in enumerate(self.books):
            p.resetBasePositionAndOrientation(
                book, 
                self.books_positions[i], 
                p.getQuaternionFromEuler([0, 0, 0])
            )
    
    def is_success(self):
        """Check if book 3 has been successfully pulled out"""
        book3_pos, _ = p.getBasePositionAndOrientation(self.books[2])
        book3_pos = np.array(book3_pos)
        
        # Calculate distance from initial position
        initial_pos = np.array(self.books_positions[2])
        book_pulled_dist = np.linalg.norm(book3_pos - initial_pos)
        
        return book_pulled_dist > self.success_distance
