# # from collections import defaultdict
# # from typing import List, Tuple

# # import numpy as np
# # from pyrep.objects.dummy import Dummy
# # from pyrep.objects.shape import Shape
# # from pyrep.objects.proximity_sensor import ProximitySensor
# # from rlbench.backend.conditions import Condition, DetectedCondition
# # from rlbench.backend.task import BimanualTask
# # from rlbench.backend.spawn_boundary import SpawnBoundary


# # class LiftedCondition(Condition):
# #     def __init__(self, item: Shape, min_height: float):
# #         self.item = item
# #         self.min_height = min_height

# #     def condition_met(self):
# #         # 检查物体高度是否达到阈值
# #         return self.item.get_position()[2] >= self.min_height, False


# # class PlacedCondition(Condition):
# #     def __init__(self, item: Shape, target_xy: Tuple[float, float], tol: float):
# #         self.item = item
# #         self.target_xy = target_xy
# #         self.tol = tol

# #     def condition_met(self):
# #         pos = self.item.get_position()
# #         dx = abs(pos[0] - self.target_xy[0])
# #         dy = abs(pos[1] - self.target_xy[1])
# #         in_place = (dx <= self.tol and dy <= self.tol)
# #         return in_place, False


# # class BimanualTransferItem(BimanualTask):
# #     def init_task(self) -> None:
# #         # 定义可抓取物体
# #         self.item = Shape('item')
# #         self.register_graspable_objects([self.item])


# #         # 实例化并命名所有与左臂相关的 waypoint
# #         self.wp_pre_grasp  = Dummy('left_pick_pre')   # 预抓取点
# #         self.wp_grasp      = Dummy('left_pick')       # 抓取点
# #         self.wp_post_grasp = Dummy('left_pick_post')  # 抓取后撤离点
# #         self.wp_pre_place  = Dummy('place_pre')       # 预放置点
# #         self.wp_place      = Dummy('place')           # 放置点
# #         self.wp_post_place = Dummy('place_post')      # 放置后撤离点

# #         # 将所有 waypoint 分配给左臂并注册为实际的 waypoint 列表
# #         names = [wp.get_name() for wp in (
# #             self.wp_pre_grasp, self.wp_grasp, self.wp_post_grasp,
# #             self.wp_pre_place, self.wp_place, self.wp_post_place)]
# #         # 映射到 left-arm
# #         self.waypoint_mapping = {n: 'left' for n in names}
# #         # 显式注册这些为任务的 waypoint 列表
# #         self._waypoints = [self.wp_pre_grasp, self.wp_grasp, self.wp_post_grasp,
# #                            self.wp_pre_place, self.wp_place, self.wp_post_place]
# #         names = [wp.get_name() for wp in (
# #             self.wp_pre_grasp, self.wp_grasp, self.wp_post_grasp,
# #             self.wp_pre_place, self.wp_place, self.wp_post_place)]
# #         self.waypoint_mapping = {n: 'left' for n in names}

# #     def init_episode(self, index: int) -> List[str]:
# #         self._variation_index = index
# #         right_success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
# #         left_success_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

# #         target_xy = tuple(self.wp_place.get_position()[:2])
# #         tol = 0.05  # 5cm 容差

# #         # Success: Grasp and height at 0.8 , put it down 
# #         self.register_success_conditions([
# #             DetectedCondition(self.item, left_success_sensor, negated=True),
# #             LiftedCondition(self.item, 0.80), 
# #             PlacedCondition(self.item, target_xy, tol)     
# #         ])

# #         return ['pick up the item', 'put it in the middle']

# #     def variation_count(self) -> int:
# #         return 1

# #     def boundary_root(self) -> Shape:
# #         return Shape('transfer_item_boundary')

# #     def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
# #         return [0, 0, -np.pi/8], [0, 0, np.pi/8]

# from collections import defaultdict
# from typing import List, Tuple

# import numpy as np
# from pyrep.objects.dummy import Dummy
# from pyrep.objects.shape import Shape
# from pyrep.objects.proximity_sensor import ProximitySensor
# from rlbench.backend.conditions import Condition, DetectedCondition
# from rlbench.backend.task import BimanualTask
# from rlbench.backend.spawn_boundary import SpawnBoundary


# class LiftedCondition(Condition):
#     def __init__(self, item: Shape, min_height: float):
#         self.item = item
#         self.min_height = min_height

#     def condition_met(self):
#         # 检查物体高度是否达到阈值
#         return self.item.get_position()[2] >= self.min_height, False


# class PlacedCondition(Condition):
#     def __init__(self, item: Shape, target_xy: Tuple[float, float], tol: float):
#         self.item = item
#         self.target_xy = target_xy
#         self.tol = tol

#     def condition_met(self):
#         pos = self.item.get_position()
#         dx = abs(pos[0] - self.target_xy[0])
#         dy = abs(pos[1] - self.target_xy[1])
#         in_place = (dx <= self.tol and dy <= self.tol)
#         return in_place, False


# class BimanualTransferItem(BimanualTask):
#     def init_task(self) -> None:
#         # 不调用 super().init_task()，禁止触发 NotImplementedError

#         # 定义可抓取物体
#         self.item = Shape('item')
#         self.register_graspable_objects([self.item])

#         # 实例化并命名所有与左臂相关的 waypoint
#         self.wp_pre_grasp  = Dummy('left_pick_pre')   # 预抓取点
#         self.wp_grasp      = Dummy('left_pick')       # 抓取点
#         self.wp_post_grasp = Dummy('left_pick_post')  # 抓取后撤离点
#         self.wp_pre_place  = Dummy('place_pre')       # 预放置点
#         self.wp_place      = Dummy('place')           # 放置点
#         self.wp_post_place = Dummy('place_post')      # 放置后撤离点

#         # 将所有 waypoint 分配给左臂并注册为实际的 waypoint 列表
#         names = [wp.get_name() for wp in (
#             self.wp_pre_grasp, self.wp_grasp, self.wp_post_grasp,
#             self.wp_pre_place, self.wp_place, self.wp_post_place)]
#         self.waypoint_mapping = {n: 'left' for n in names}
#         self._waypoints = [
#             self.wp_pre_grasp, self.wp_grasp, self.wp_post_grasp,
#             self.wp_pre_place, self.wp_place, self.wp_post_place
#         ]

#     def init_episode(self, index: int) -> List[str]:
#         self._variation_index = index

#         # 使用 SpawnBoundary 确保在 transfer_item_boundary 区域内放置物体
#         boundary = SpawnBoundary(self.boundary_root())
#         boundary.clear()
#         boundary.sample(self.item)

#         right_success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
#         left_success_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

#         target_xy = tuple(self.wp_place.get_position()[:2])
#         tol = 0.05  # 5cm 容差

#         self.register_success_conditions([
#             DetectedCondition(self.item, left_success_sensor, negated=True),
#             LiftedCondition(self.item, 0.80),
#             PlacedCondition(self.item, target_xy, tol)
#         ])

#         return ['pick up the item', 'put it in the middle']

#     def variation_count(self) -> int:
#         return 1

#     def boundary_root(self) -> Shape:
#         return Shape('transfer_item_boundary')

#     def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
#         return [0, 0, -np.pi/8], [0, 0, np.pi/8]

from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.conditions import NothingGrasped,GraspedCondition
from rlbench.backend.conditions import ConditionSet
from rlbench.backend.task import BimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.conditions import Condition


class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False

class BimanualTransferItem(BimanualTask):

    def init_task(self) -> None:

        self.item = Shape('item')

        self.register_graspable_objects([self.item])

        self.waypoint_mapping = defaultdict(lambda: 'right')
        for i in range(8):
            self.waypoint_mapping[f'waypoint{i}'] = 'left'

        self.waypoint_mapping.update({'waypoint0': 'right'})

        self.boundaries = Shape('transfer_item_boundary')

        # release_wp_index = 6
        # self.register_waypoint_ability_end(
        #     release_wp_index,
        #     lambda wp: self.robot.release_gripper('left')
        # )

        #self.mid_sensor = ProximitySensor('success_middle')

    def init_episode(self, index: int) -> List[str]:

        self._variation_index = index

        right_success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
        left_success_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

        #b = SpawnBoundary([self.boundaries])
        #b.clear()
        #    b.sample(item, min_distance=0.1)


        seq = [
 
            GraspedCondition(self.robot.left_gripper, self.item),   # 1) Left catch 
            NothingGrasped(self.robot.left_gripper),               # 2) Left put in the middle 

            GraspedCondition(self.robot.right_gripper, self.item), # 3) Right catch
            NothingGrasped(self.robot.right_gripper)               # 5) Right put on the side 
        ]


        self.register_success_conditions([
            ConditionSet(seq, order_matters=True)
        ])
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
