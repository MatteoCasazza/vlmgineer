import math
import pybullet as p
import pybullet_data


class PyBulletLayoutEvaluator:
    def __init__(self, gui=False):
        self.gui = gui
        self.ee_link_index = 11
        self.allowed_contact_types = []
    
    def snap_objects_to_table_surface(self, layout):
        """
        Normalize object z positions so task objects, targets, and obstacles
        are placed on top of the table surface.

        PyBullet boxes use position as the geometric center.
        VLM outputs may ambiguously describe object positions as surface poses.
        """
        tables = [obj for obj in layout["objects"] if obj["object_type"] == "table"]

        if not tables:
            return layout

        table = tables[0]
        table_top_z = table["position"][2] + table["dimensions"][2] / 2.0

        surface_object_types = ["task_object", "target_area", "obstacle"]

        for obj in layout["objects"]:
            if obj["object_type"] in surface_object_types:
                obj_height = obj["dimensions"][2]
                obj["position"][2] = table_top_z + obj_height / 2.0

        return layout

    def evaluate_layout(self, layout):
        layout = self.snap_objects_to_table_surface(layout)

        connection_mode = p.GUI if self.gui else p.DIRECT
        physics_client = p.connect(connection_mode)

        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")

            robot_id = None
            object_body_ids = {}
            object_types = {}

            for obj in layout["objects"]:
                if obj["object_type"] == "robot_base":
                    robot_base_position = [
                        obj["position"][0],
                        obj["position"][1],
                        0.0,
                    ]

                    robot_id = p.loadURDF(
                        "franka_panda/panda.urdf",
                        basePosition=robot_base_position,
                        baseOrientation=p.getQuaternionFromEuler(obj["orientation"]),
                        useFixedBase=True,
                        flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                    )
                else:
                    body_id = self.spawn_box(obj)
                    object_body_ids[obj["object_name"]] = body_id
                    object_types[obj["object_name"]] = obj["object_type"]

            if robot_id is None:
                return {
                    "physical_feasibility_score": 0.0,
                    "ik_reachability_score": 0.0,
                    "collision_free_score": 0.0,
                    "details": [],
                    "error": "No robot_base found",
                }

            details = self.check_targets(
                robot_id,
                layout["objects"],
                object_body_ids,
                object_types,
            )

            clearance_result = self.check_robot_table_clearance(layout)

            if not details:
                return {
                    "physical_feasibility_score": 0.0,
                    "ik_reachability_score": 0.0,
                    "collision_free_score": 0.0,
                    "details": [],
                    "error": "No task_object or target_area found",
                }

            ik_score = sum(1 for d in details if d["reachable"]) / len(details)
            collision_score = sum(1 for d in details if d["collision_free"]) / len(details)
            feasibility_score = sum(1 for d in details if d["physically_feasible"]) / len(details)

            clearance_score = 1.0 if clearance_result["clearance_ok"] else 0.0

            combined_physical_score = (
                0.4 * feasibility_score
                + 0.25 * ik_score
                + 0.25 * collision_score
                + 0.1 * clearance_score
            )

            if not clearance_result["clearance_ok"]:
                combined_physical_score *= 0.5

            return {
                "combined_physical_score": combined_physical_score,
                "physical_feasibility_score": feasibility_score,
                "ik_reachability_score": ik_score,
                "collision_free_score": collision_score,
                "robot_table_clearance": clearance_result,
                "details": details,
            }

        finally:
            p.disconnect(physics_client)

    def spawn_box(self, obj):
        position = obj["position"]
        orientation_rpy = obj["orientation"]
        dimensions = obj["dimensions"]

        half_extents = [
            dimensions[0] / 2,
            dimensions[1] / 2,
            dimensions[2] / 2,
        ]

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
        )

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
        )

        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=p.getQuaternionFromEuler(orientation_rpy),
        )

        return body_id

    def reset_robot_home(self, robot_id):
        """
        Reset Panda to a neutral home configuration before each IK/collision check.
        This avoids state contamination between targets.
        """
        home_joints = [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8]

        for joint_idx, joint_value in enumerate(home_joints):
            p.resetJointState(robot_id, joint_idx, joint_value)

        p.stepSimulation()

    def check_targets(self, robot_id, objects, object_body_ids, object_types):
        results = []

        for obj in objects:
            if obj["object_type"] not in ["task_object", "target_area"]:
                continue

            object_top_z = obj["position"][2] + obj["dimensions"][2] / 2.0

            target_pos = [
                obj["position"][0],
                obj["position"][1],
                object_top_z + 0.18,
            ]

            approach_pos = [
                target_pos[0],
                target_pos[1],
                target_pos[2] + 0.20,
            ]

            target_orientation = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

            self.reset_robot_home(robot_id)

            trajectory_collision = self.check_approach_trajectory(
                robot_id,
                approach_pos,
                target_pos,
                target_orientation,
                object_body_ids,
                object_types,
            )

            self.reset_robot_home(robot_id)

            ik_solution = p.calculateInverseKinematics(
                robot_id,
                self.ee_link_index,
                target_pos,
                targetOrientation=target_orientation,
                maxNumIterations=100,
                residualThreshold=1e-4,
            )

            if ik_solution is None or len(ik_solution) == 0:
                reachable = False
                error = None
                ik_joint_solution = None
                forbidden_collisions = []
                allowed_contacts = []
                collision_free = False
            else:
                for joint_idx in range(7):
                    p.resetJointState(robot_id, joint_idx, ik_solution[joint_idx])

                ik_joint_solution = list(ik_solution[:7])
                p.stepSimulation()

                ee_state = p.getLinkState(robot_id, self.ee_link_index)
                ee_pos = ee_state[0]

                error = (
                    (ee_pos[0] - target_pos[0]) ** 2
                    + (ee_pos[1] - target_pos[1]) ** 2
                    + (ee_pos[2] - target_pos[2]) ** 2
                ) ** 0.5

                reachable = error < 0.08

                forbidden_collisions, allowed_contacts = self.check_robot_collisions(
                    robot_id,
                    object_body_ids,
                    object_types,
                )

                safety_margin_result = self.check_robot_safety_margin(
                    robot_id,
                    object_body_ids,
                    object_types,
                )

                safety_margin_ok = safety_margin_result["safety_margin_ok"]

                collision_free = len(forbidden_collisions) == 0

            physically_feasible = (
                reachable
                and collision_free
                and not trajectory_collision
                and safety_margin_ok
            )

            results.append({
                "object_name": obj["object_name"],
                "object_type": obj["object_type"],
                "target_position": target_pos,
                "reachable": reachable,
                "ik_error": error,
                "collision_free": collision_free,
                "allowed_contacts": allowed_contacts,
                "forbidden_collisions": forbidden_collisions,
                "physically_feasible": physically_feasible,
                "trajectory_collision": trajectory_collision,
                "safety_margin": safety_margin_result,
                "ik_joint_solution": ik_joint_solution,
            })

        return results

    def check_approach_trajectory(
        self,
        robot_id,
        approach_pos,
        target_pos,
        target_orientation,
        object_body_ids,
        object_types,
        n_steps=10,
    ):
        """
        Check whether the robot can move from an approach pose to the target pose
        without forbidden collisions.
        """

        self.reset_robot_home(robot_id)

        for step in range(n_steps + 1):
            alpha = step / n_steps

            interpolated_pos = [
                approach_pos[0] * (1 - alpha) + target_pos[0] * alpha,
                approach_pos[1] * (1 - alpha) + target_pos[1] * alpha,
                approach_pos[2] * (1 - alpha) + target_pos[2] * alpha,
            ]

            ik_solution = p.calculateInverseKinematics(
                robot_id,
                self.ee_link_index,
                interpolated_pos,
                targetOrientation=target_orientation,
                maxNumIterations=100,
                residualThreshold=1e-4,
            )

            if ik_solution is None or len(ik_solution) == 0:
                safety_margin_result = {
                    "safety_margin_ok": False,
                    "required_margin": 0.05,
                    "min_distance": None,
                    "too_close_objects": [],
                }
                safety_margin_ok = False
                return True

            for joint_idx in range(7):
                p.resetJointState(robot_id, joint_idx, ik_solution[joint_idx])

            p.stepSimulation()

            forbidden_collisions, _ = self.check_robot_collisions(
                robot_id,
                object_body_ids,
                object_types,
            )

            if forbidden_collisions:
                return True

        return False

    def check_robot_collisions(self, robot_id, object_body_ids, object_types):
        p.performCollisionDetection()

        forbidden_collisions = []
        allowed_contacts = []

        for obj_name, body_id in object_body_ids.items():
            contact_points = p.getClosestPoints(
                robot_id,
                body_id,
                distance=0.03,
            )

            if len(contact_points) == 0:
                continue

            real_collision = False
            min_distance = float("inf")

            for cp in contact_points:
                distance = cp[8]

                min_distance = min(min_distance, distance)

                if distance < 0.0:
                    real_collision = True

            obj_type = object_types[obj_name]

            contact_info = {
                "object_name": obj_name,
                "object_type": obj_type,
                "num_contacts": len(contact_points),
                "min_distance": min_distance,
            }

            if real_collision:
                forbidden_collisions.append(contact_info)

        return forbidden_collisions, allowed_contacts

    def check_robot_safety_margin(
        self,
        robot_id,
        object_body_ids,
        object_types,
        safety_margin=0.05,
    ):
        """
        Check minimum distance between the robot and forbidden objects.
        Even if there is no contact, configurations too close to tables/obstacles
        are considered unsafe.
        """

        forbidden_types = ["table", "obstacle"]

        min_distance = float("inf")
        too_close_objects = []

        for obj_name, body_id in object_body_ids.items():
            obj_type = object_types[obj_name]

            if obj_type not in forbidden_types:
                continue

            closest_points = p.getClosestPoints(
                bodyA=robot_id,
                bodyB=body_id,
                distance=safety_margin,
            )

            if closest_points:
                distances = [point[8] for point in closest_points]
                local_min_distance = min(distances)

                min_distance = min(min_distance, local_min_distance)

                too_close_objects.append({
                    "object_name": obj_name,
                    "object_type": obj_type,
                    "min_distance": local_min_distance,
                })

        if min_distance == float("inf"):
            min_distance = None

        return {
            "safety_margin_ok": len(too_close_objects) == 0,
            "required_margin": safety_margin,
            "min_distance": min_distance,
            "too_close_objects": too_close_objects,
        }

    def check_robot_table_clearance(self, layout, min_clearance=0.25):
        robot = None
        tables = []

        for obj in layout["objects"]:
            if obj["object_type"] == "robot_base":
                robot = obj
            elif obj["object_type"] == "table":
                tables.append(obj)

        if robot is None or not tables:
            return {
                "clearance_ok": False,
                "min_distance": None,
                "required_clearance": min_clearance,
            }

        robot_x, robot_y, _ = robot["position"]

        min_distance = float("inf")

        for table in tables:
            table_x, table_y, _ = table["position"]
            table_l, table_w, _ = table["dimensions"]

            dx = abs(robot_x - table_x) - table_l / 2
            dy = abs(robot_y - table_y) - table_w / 2

            # distanza 2D approssimata tra base robot e bounding box del tavolo
            clearance_x = max(dx, 0)
            clearance_y = max(dy, 0)

            distance = (clearance_x ** 2 + clearance_y ** 2) ** 0.5
            min_distance = min(min_distance, distance)

        return {
            "clearance_ok": min_distance >= min_clearance,
            "min_distance": min_distance,
            "required_clearance": min_clearance,
        }