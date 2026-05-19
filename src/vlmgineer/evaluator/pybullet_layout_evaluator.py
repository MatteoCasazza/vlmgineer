import pybullet as p
import pybullet_data


class PyBulletLayoutEvaluator:
    def __init__(self, gui=False):
        self.gui = gui
        self.ee_link_index = 11
        self.allowed_contact_types = ["task_object", "target_area"]

    def evaluate_layout(self, layout):
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
                    robot_id = p.loadURDF(
                        "franka_panda/panda.urdf",
                        basePosition=obj["position"],
                        baseOrientation=p.getQuaternionFromEuler(obj["orientation"]),
                        useFixedBase=True,
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

    def check_targets(self, robot_id, objects, object_body_ids, object_types):
        results = []

        for obj in objects:
            if obj["object_type"] not in ["task_object", "target_area"]:
                continue

            target_pos = [
                obj["position"][0],
                obj["position"][1],
                obj["position"][2] + 0.15,
            ]

            ik_solution = p.calculateInverseKinematics(
                robot_id,
                self.ee_link_index,
                target_pos,
            )

            if ik_solution is None or len(ik_solution) == 0:
                reachable = False
                error = None
                forbidden_collisions = []
                allowed_contacts = []
                collision_free = False
            else:
                for joint_idx in range(7):
                    p.resetJointState(robot_id, joint_idx, ik_solution[joint_idx])

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

                collision_free = len(forbidden_collisions) == 0

            physically_feasible = reachable and collision_free

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
            })

        return results

    def check_robot_collisions(self, robot_id, object_body_ids, object_types):
        p.performCollisionDetection()

        forbidden_collisions = []
        allowed_contacts = []

        for obj_name, body_id in object_body_ids.items():
            contact_points = p.getContactPoints(robot_id, body_id)

            if len(contact_points) == 0:
                continue

            obj_type = object_types[obj_name]

            contact_info = {
                "object_name": obj_name,
                "object_type": obj_type,
                "num_contacts": len(contact_points),
            }

            if obj_type in self.allowed_contact_types:
                allowed_contacts.append(contact_info)
            else:
                forbidden_collisions.append(contact_info)

        return forbidden_collisions, allowed_contacts

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