import pybullet as p
import pybullet_data
import time

class Cube:
    def __init__(self, length, pose, mass=1.0, color=[1, 0, 0, 1]):
        # Define cube parameters.
        self.length = length
        self.pose = pose
        self.mass = mass 
        self.color = color

        box_half_extents = [self.length / 2.0, self.length / 2.0, self.length / 2.0]
        box_collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=box_half_extents)
        box_visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=box_half_extents, rgbaColor=self.color)
        self.box_body = p.createMultiBody(
            baseMass=self.mass, 
            baseCollisionShapeIndex=box_collision_id, 
            baseVisualShapeIndex=box_visual_id, 
            basePosition=self.pose
        )
        
    def get_shape(self):
        return self.box_body

