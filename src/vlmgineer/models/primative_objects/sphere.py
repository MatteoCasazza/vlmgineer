import pybullet as p
import pybullet_data
import time

class Sphere:
    def __init__(self, radius, pose, mass=1.0, color=[1, 0, 0, 1]):
        # Define sphere parameters.
        self.radius = radius
        self.pose = pose
        self.mass = mass 
        self.color = color

        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius, rgbaColor=self.color)
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)
        self.sphere = p.createMultiBody(baseMass=self.mass,
                                        baseCollisionShapeIndex=sphere_collision,
                                        baseVisualShapeIndex=sphere_visual,
                                        basePosition=self.pose)
        
    def get_shape(self):
        return self.sphere