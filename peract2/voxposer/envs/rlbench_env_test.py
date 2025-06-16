import os
import numpy as np
import open3d as o3d
import json
from rlbench.action_modes.action_mode import MoveArmThenGripper, SelectableMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import ArmActionMode, EndEffectorPoseViaPlanning, SelectableUnimanualArmEndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete, GripperActionMode, SelectableUnimanualDiscrete
from rlbench.environment import Environment
import rlbench.tasks as tasks
from pyrep.const import ObjectType
from pyrep.objects.shape import Shape
from voxposer.utils import normalize_vector, bcolors
import math
from PIL import Image
import io
import base64
import requests
import time
from voxposer.LLM_cache import DiskCache

class CustomSelectableMoveArmThenGripper(SelectableMoveArmThenGripper):
    """
    Migrates CustomMoveArmThenGripper2Robots logic onto SelectableMoveArmThenGripper.
    Adds deduplication of identical arm actions and fault-tolerance.
    """
    def __init__(self, arm_action_mode, gripper_action_mode):
        super().__init__(arm_action_mode, gripper_action_mode)
        self._prev_arm_action = None

    def action(self, scene, action, which_arm):
        # Normalize aliases
        if which_arm == 'right hand':
            which_arm = 'right'
        if which_arm == 'left hand':
            which_arm = 'left'
        if which_arm not in ['right hand', 'left hand', 'left', 'right']:
            raise NotImplementedError(f"Unsupported which_arm: {which_arm}")

        # Split action into arm + gripper (rest)
        arm_act_size = int(np.prod(self.arm_action_mode.action_shape(scene)))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])

        # Deduplicate: skip arm if same as previous
        if self._prev_arm_action is not None and np.allclose(arm_action, self._prev_arm_action):
            self.gripper_action_mode.action(scene, ee_action, which_arm)
        else:
            # Fault-tolerant arm execution
            try:
                self.arm_action_mode.action(scene, arm_action, which_arm=which_arm)
            except Exception as e:
                print(f"[Custom] Ignoring failed arm action; Exception: {e}")
            # Always execute gripper
            self.gripper_action_mode.action(scene, ee_action, which_arm)

        # Update cache
        self._prev_arm_action = arm_action.copy()

    def action_peract(self, scene, action, which_arm):
        # Same decomposition for PerAct
        arm_act_size = int(np.prod(self.arm_action_mode.action_shape(scene)))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        ignore_collisions = bool(action[arm_act_size+1:arm_act_size+2])

        self.arm_action_mode.action(scene, arm_action, ignore_collisions, which_arm)
        self.gripper_action_mode.action(scene, ee_action, which_arm)

    def action_shape(self, scene):
        # Inherit shape: arm + gripper + collision flag
        arm_dim = int(np.prod(self.arm_action_mode.action_shape(scene)))
        grip_dim = int(np.prod(self.gripper_action_mode.action_shape(scene)))
        return arm_dim + grip_dim + 1
    

class CustomVoxPoserRLBenchRobots():
    def __init__(self, visualizer=None, observation_config=None, dataset_root=None, headless=None, task_name=None, dominant_assitive_policy=False, custom_ttt_file=''):
        """
        Initializes the VoxPoserRLBench environment.

        Args:
            visualizer: Visualization interface, optional.
        """
        action_mode = CustomSelectableMoveArmThenGripper(arm_action_mode=SelectableUnimanualArmEndEffectorPoseViaPlanning(),
                        gripper_action_mode=SelectableUnimanualDiscrete())
        if observation_config is not None:
            # VoxPoser + PerAct
            self.rlbench_env = Environment(action_mode=action_mode, obs_config=observation_config, dataset_root=dataset_root, headless=headless, task_name=task_name)
        else:
            self.rlbench_env = Environment(action_mode, task_name=task_name)
        if custom_ttt_file != '':
            self.rlbench_env._TTT_FILE = custom_ttt_file
        self.rlbench_env.launch()
        self.task = None
        self.task_name = task_name

        self.workspace_bounds_min = np.array([self.rlbench_env._scene._workspace_minx, self.rlbench_env._scene._workspace_miny, self.rlbench_env._scene._workspace_minz])
        self.workspace_bounds_max = np.array([self.rlbench_env._scene._workspace_maxx, self.rlbench_env._scene._workspace_maxy, self.rlbench_env._scene._workspace_maxz])
        self.visualizer = visualizer

        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)


        
        # self.camera_names = ['front', 'wrist_left', 'wrist_right']
        self.camera_names = list(self.rlbench_env._scene.camera_sensors.keys())
        forward_vector = np.array([0, 0, 1])
        self.lookat_vectors = {}
        # calculate lookat vector for all cameras (for normal estimation)
        # name2cam = {
        #     'front': self.rlbench_env._scene._cam_front,
        #     'wrist_left': self.rlbench_env._scene._cam_wrist_left,
        #     'wrist_right': self.rlbench_env._scene._cam_wrist_right,
        # }
        name2cam = self.rlbench_env._scene.camera_sensors
        for cam_name in self.camera_names:
            extrinsics = name2cam[cam_name].get_matrix()
            lookat = extrinsics[:3, :3] @ forward_vector
            self.lookat_vectors[cam_name] = normalize_vector(lookat)


        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task_object_names.json')
        with open(path, 'r') as f:
            self.task_object_names = json.load(f)

        self._reset_task_variables()

        # dominant, assistive policies
        self._dominant_assitive_policy = dominant_assitive_policy
        self._dominant_arm = ''
        self._dominant_arm_for_ep_reset = ''

        ##### ChatGPT for determining acting arm
        # self.rlbench_env.add_highres_front_cam_for_llm()
        # self._image_cache = DiskCache(cache_dir='../../../../voxposer/cache', load_cache=True)
        # self._env_image_folder_path = f'../../../../voxposer/env_images'
        # self._env_image_saved_path = f'{self._env_image_folder_path}/direct_view.jpg'
        # if not os.path.exists(self._env_image_folder_path):
        #     os.makedirs(self._env_image_folder_path)
