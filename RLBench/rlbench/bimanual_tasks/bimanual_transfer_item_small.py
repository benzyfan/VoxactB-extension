from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.conditions import NothingGrasped,GraspedCondition
from rlbench.backend.conditions import ConditionSet
from rlbench.backend.task import BimanualTask,DABimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition


class BimanualTransferItemSmall(DABimanualTask):

    use_dominant_assistive = True
    def init_task(self) -> None:

        self.item = Shape('item')
        self.register_graspable_objects([self.item])
        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(8):
            self.waypoint_mapping[f'waypoint{i}'] = 'left'
        self.waypoint_mapping.update({'waypoint0': 'right'})

        self.boundaries = Shape('transfer_item_boundary')
        self.end_boundaries = Shape('end_item_boundary')
        self.end_sensor = ProximitySensor('end_place')
        # release_wp_index = 6
        # self.register_waypoint_ability_end(
        #     release_wp_index,
        #     lambda wp: self.robot.release_gripper('left')
        # )

        #self.mid_sensor = ProximitySensor('success_middle')

    def init_episode(self, index: int) -> List[str]:

        self._variation_index = index

        self._dominant = 'right'
        # Have to force generate some waypoints first ,otherwise will cause fault!
        _ = self.get_waypoints()
        self.reorder_waypoints(self._dominant)

        # print("In the task , we have the self._dominant", self._dominant)

        b = SpawnBoundary([self.boundaries])
        b.clear()
        b.sample(self.item, min_distance=0.1)

        b2 = SpawnBoundary([self.end_boundaries])
        b2.clear()
        b2.sample(self.end_sensor, min_rotation = (0,0,0), max_rotation=(0,0,0), min_distance=0.1)

        if self._dominant == 'left':
            seq = [
                GraspedCondition(self.robot.right_gripper, self.item),
                DetectedCondition(self.item, ProximitySensor('middle_place')),
                NothingGrasped(self.robot.right_gripper),
                GraspedCondition(self.robot.left_gripper,  self.item),
                DetectedCondition(self.item, ProximitySensor('end_place')),
            ]
        else:
            seq = [
                GraspedCondition(self.robot.left_gripper,  self.item),
                DetectedCondition(self.item, ProximitySensor('middle_place')),
                NothingGrasped(self.robot.left_gripper),
                GraspedCondition(self.robot.right_gripper, self.item),
                DetectedCondition(self.item, ProximitySensor('end_place')),
            ]
        self.register_success_conditions([ConditionSet(seq, order_matters=True)])
 
        # self.register_stop_at_waypoint(8)

        return ['pick item from the left',
                'place the item in the middle',
                'pick the item from the middle',
                'put item on the right side']

    def variation_count(self) -> int:
        return 1

    def boundary_root(self) -> Object:
        return Shape('transfer_item_boundary')

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]

    def is_static_workspace(self) -> bool:
        #Set the place of item fixed
        return True
