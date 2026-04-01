from vlmgineer.envs.base_env import BaseEnv
from vlmgineer.models.primative_objects.cube import Cube
import numpy as np
import pybullet as p
import colorsys

class CleanTableTopEnv(BaseEnv):
    """
    A pushing environment that can spawn N cubes at random table positions.
    """
    def __init__(
        self,
        n_cubes: int       = 8,
        cube_size: float   = 0.03,
        goal_center: tuple = (1.1, 0.0),  # (x, y) center of the goal circle
        goal_radius: float = 0.15,          # radius of the goal circle
        **kwargs,
    ):
        # ---------- user‐configurable ----------
        self.n_cubes      = n_cubes
        self.cube_size    = cube_size
        self.table_xy_rng = dict(x=(0.5, 0.9),  # x- and y-ranges on the table
                                 y=(-0.2, 0.2))
        # ---------------------------------------

        # IDs of all cubes that will be spawned in _setup_environment
        self.cubes      = []
        # Store initial positions for reward calculation
        self.initial_positions = None

        # --- pushing goal ---------------------------------------------------
        self.goal_center = goal_center     # center of the goal circle
        self.goal_radius = goal_radius     # radius of the goal circle
        self._goal_circle_id = None        # handle for the debug circle

        super().__init__(**kwargs)
    
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
        
        # Spawn N cubes at random (non-overlapping) positions on the table
        occupied = []                                      # keep track of placed cubes
        def _sample_pos() -> np.ndarray:
            """Sample a random (x, y, z) inside the user-defined table bounds."""
            return np.array([
                np.random.uniform(*self.table_xy_rng["x"]),
                np.random.uniform(*self.table_xy_rng["y"]),
                self.cube_size / 2
            ])

        def _random_low_saturation_color() -> tuple:
            """Generate a random color with low saturation."""
            h = np.random.rand()  # Random hue
            s = np.random.uniform(0.1, 0.3)  # Low saturation
            v = np.random.uniform(0.7, 1.0)  # High value
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            return (r, g, b)

        for _ in range(self.n_cubes):
            # --- rejection sample to avoid initial overlap between cubes ----
            for __ in range(100):                          # give up after 100 tries
                pos = _sample_pos()
                if all(np.linalg.norm(pos[:2] - o[:2]) > self.cube_size * 2
                       for o in occupied):
                    occupied.append(pos)
                    break
            else:
                raise RuntimeError("Failed to place cube without overlap.")

            cube_id = Cube(self.cube_size, pos).get_shape()
            self.cubes.append(cube_id)

            # Set random low saturation color
            color = _random_low_saturation_color()
            p.changeVisualShape(cube_id, -1, rgbaColor=color + (1,))  # Add alpha channel

        # --------------------------------------------------------------------
        # Draw / refresh the visual "goal circle" so the user can see it.
        # (Remove the old one first if we are re-initialising.)
        if self._goal_circle_id is not None:
            p.removeUserDebugItem(self._goal_circle_id)

        # Draw a circle by approximating it with a polygon
        num_segments = 36
        # Create multiple concentric circles for thickness
        radius_offsets = [-0.001, 0.0, 0.001]  # Multiple radius offsets for thickness
        for radius_offset in radius_offsets:
            points = [
                (
                    self.goal_center[0] + (self.goal_radius + radius_offset) * np.cos(2 * np.pi * i / num_segments),
                    self.goal_center[1] + (self.goal_radius + radius_offset) * np.sin(2 * np.pi * i / num_segments),
                    0.001
                )
                for i in range(num_segments + 1)
            ]

            for i in range(num_segments):
                self._goal_circle_id = p.addUserDebugLine(
                    points[i], points[i + 1],
                    lineColorRGB=[0, 1, 0],  # green line
                )
    
    def reward(self) -> float:
        """
        Calculate the reward for the current state of the environment.

        The reward is designed to encourage cubes to move into a specified circular target zone.

        Rules:
        -----
        1. If a cube is within the goal circle, its reward is 1.0
        2. If a cube is outside the goal circle, its reward is the ratio of:
           current distance from circle center / initial distance from circle center
        3. The final reward is the mean of all per-cube rewards, normalized between 0.0 and 1.0

        Returns:
        -------
        float
            The calculated reward for the current state, normalized between 0.0 and 1.0
        """
        if self.initial_positions is None:
            # Store initial positions if not already stored
            self.initial_positions = [
                p.getBasePositionAndOrientation(cube_id)[0][:2]
                for cube_id in self.cubes
            ]

        # Get current positions
        current_positions = [
            p.getBasePositionAndOrientation(cube_id)[0][:2]
            for cube_id in self.cubes
        ]

        rewards = []
        for init_pos, curr_pos in zip(self.initial_positions, current_positions):
            curr_dist = np.linalg.norm(np.array(curr_pos[:2]) - np.array(self.goal_center[:2]), ord=1)
            init_dist = np.linalg.norm(np.array(init_pos[:2]) - np.array(self.goal_center[:2]), ord=1)
            
            if curr_dist <= self.goal_radius + 0.01:
                # Cube is in the goal circle
                rewards.append(1.0)
            else:
                # Calculate ratio of distances, ensuring it's between 0 and 1
                ratio = max(0.0, min(1.0, 1.0 - (curr_dist / init_dist)))
                rewards.append(ratio)

        return np.mean(rewards)
    
    def reset(self):
        super().reset()
        # Reset initial positions
        self.initial_positions = None
        # Re-sample *new* random positions for every cube
        occupied = []
        for cube_id in self.cubes:
            for __ in range(100):
                pos = np.array([
                    np.random.uniform(*self.table_xy_rng["x"]),
                    np.random.uniform(*self.table_xy_rng["y"]),
                    self.cube_size / 2
                ])
                if all(np.linalg.norm(pos[:2] - o[:2]) > self.cube_size * 2
                       for o in occupied):
                    occupied.append(pos)
                    break
            p.resetBasePositionAndOrientation(
                cube_id,
                pos,
                p.getQuaternionFromEuler([0, 0, 0])
            )