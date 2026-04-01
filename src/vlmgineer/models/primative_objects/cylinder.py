import pybullet as p
import pybullet_data
import time

class Cylinder:
    def __init__(self, cylinder_radius, cylinder_height, cylinder_pos):
        # Define cylinder parameters.

        cylinder_collision = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                                    radius=cylinder_radius,
                                                    height=cylinder_height)

        cylinder_visual = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                            radius=cylinder_radius,
                                            length=cylinder_height,  # Some PyBullet versions use 'length' for visuals.
                                            rgbaColor=[0, 0, 1, 1])

        self.cylinder_body = p.createMultiBody(baseMass=1, 
                                        baseCollisionShapeIndex=cylinder_collision,
                                        baseVisualShapeIndex=cylinder_visual,
                                        basePosition=cylinder_pos)  # Position the cylinder above the plane.
        
    def get_shape(self):
        return self.cylinder_body