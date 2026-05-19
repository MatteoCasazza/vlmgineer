import math


class LayoutEvaluator:
    def __init__(self):
        # workspace semplice (box)
        self.workspace_limits = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (0.0, 1.5),
        }

    def evaluate_layout(self, layout):
        objects = layout["objects"]

        collision_penalty = self.check_collisions(objects)
        workspace_penalty = self.check_workspace(objects)
        reachability_score = self.check_reachability(objects)

        # score finale (semplice)
        score = (
            0.5 * (1 - collision_penalty)
            + 0.3 * (1 - workspace_penalty)
            + 0.2 * reachability_score
        )

        return {
            "layout_score": score,
            "collision_penalty": collision_penalty,
            "workspace_penalty": workspace_penalty,
            "reachability_score": reachability_score,
        }

    # -----------------------
    # 1. Collision check (AABB)
    # -----------------------
    def check_collisions(self, objects):
        collisions = 0
        total_pairs = 0

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                total_pairs += 1
                if self.aabb_overlap(objects[i], objects[j]):
                    collisions += 1

        if total_pairs == 0:
            return 0

        return collisions / total_pairs

    def aabb_overlap(self, obj1, obj2):
        for axis in range(3):
            p1 = obj1["position"][axis]
            p2 = obj2["position"][axis]

            d1 = obj1["dimensions"][axis] / 2
            d2 = obj2["dimensions"][axis] / 2

            if abs(p1 - p2) > (d1 + d2):
                return False

        return True

    # -----------------------
    # 2. Workspace check
    # -----------------------
    def check_workspace(self, objects):
        violations = 0

        for obj in objects:
            x, y, z = obj["position"]

            if not (self.workspace_limits["x"][0] <= x <= self.workspace_limits["x"][1]):
                violations += 1
            if not (self.workspace_limits["y"][0] <= y <= self.workspace_limits["y"][1]):
                violations += 1
            if not (self.workspace_limits["z"][0] <= z <= self.workspace_limits["z"][1]):
                violations += 1

        return violations / len(objects)

    # -----------------------
    # 3. Reachability (semplificata)
    # -----------------------
    def check_reachability(self, objects):
        robot = None

        for obj in objects:
            if obj["object_type"] == "robot_base":
                robot = obj
                break

        if robot is None:
            return 0

        reachable = 0
        total = 0

        for obj in objects:
            if obj["object_type"] == "task_object":
                total += 1
                dist = self.distance(robot["position"], obj["position"])

                if dist < 1.0:  # raggio semplice
                    reachable += 1

        if total == 0:
            return 0

        return reachable / total

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2
            + (p1[1] - p2[1]) ** 2
            + (p1[2] - p2[2]) ** 2
        )