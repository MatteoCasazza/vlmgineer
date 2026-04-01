import time
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

class EnvRunner:
    def __init__(self, env, waypoints, enable_orn=True, enable_gripper=False):
        if not isinstance(waypoints, np.ndarray):
            waypoints = np.array(waypoints)
        assert len(waypoints.shape) == 2
        self.env = env
        self.waypoints = waypoints
        self.waypoints = np.vstack([
            self.waypoints,
            np.repeat(self.waypoints[-1][None, :], 2, axis=0)
        ])
        self.enable_orn = enable_orn
        self.enable_gripper = enable_gripper
        if self.enable_orn:
            if self.enable_gripper:
                assert waypoints.shape[1] == 7
            else:
                assert waypoints.shape[1] == 6
        else:
            if self.enable_gripper:
                assert waypoints.shape[1] == 4
            else:
                assert waypoints.shape[1] == 3
        self.num_interpolate_points = 500
        self.default_orn = np.array([0, 0, 0])

        # eval
        self.total_waypoint_distance = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1).sum()

    def interpolate_waypoints(self, start, end):
        # Create parameter space
        t = np.linspace(0, 1, self.num_interpolate_points)
    
        # Interpolate positions linearly 
        start_pos = start[:3]
        end_pos = end[:3]
        positions = np.zeros((self.num_interpolate_points, 3))
        for i in range(self.num_interpolate_points):
            positions[i] = start_pos * (1 - t[i]) + end_pos * t[i]
        
        result = positions
        
        # Handle orientation if enabled
        if self.enable_orn:
            start_euler = start[3:6]
            end_euler = end[3:6]
            
            key_rots = Rotation.from_euler('xyz', np.vstack((start_euler, end_euler)))
            key_times = [0, 1]
            slerp = Slerp(key_times, key_rots)
            
            interp_rots = slerp(t).as_euler('xyz')
            result = np.hstack((result, interp_rots))
            
        return result

    def run(self):
        for i in range(len(self.waypoints) - 1):
            segment = self.interpolate_waypoints(self.waypoints[i], self.waypoints[i + 1])
            for pose in segment:
                pos = pose[:3]
                
                if self.enable_orn:
                    orn = pose[3:6]
                else:
                    orn = self.default_orn
                
                orn = self.env.convert_euler_to_quat(orn)
                joint_angles = self.env.solve_robot_inverse_kinematics(pos, orn)
                self.env.set_robot_target_positions(joint_angles)
                if self.enable_gripper:
                    gripper = int(self.waypoints[i, -1])
                    self.env.set_gripper_state(gripper)
                self.env.step()
                time.sleep(self.env.stepsize)

        if self.env.blender_recorder:
            self.env.blender_recorder.save()
            
        result = self.env.result()
        result['total_dis'] = self.total_waypoint_distance
        result['total_time'] = self.env.t
        return result

