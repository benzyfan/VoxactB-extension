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
    def set_dominant_hand_for_ep_reset(self, ep_number):
        """
        Set _dominant_arm_for_ep_reset variable based on current episode number. This function is used to properly set up the environment, specifically the base_rotation_bounds for each task.
        """
        if self._dominant_assitive_policy:
            # episode numbers < 12 are all left-armed dominant, and episode numbers  >= 12 to 24 are all right-armed dominant
            if ep_number < 12:
                self._dominant_arm_for_ep_reset = 'left'
            else:
                self._dominant_arm_for_ep_reset = 'right'
    def determine_dominant_hand(self):
        print("Now we are in the voxposer/rlbench_env.py, the function determine_dominant_hand, which is not useful for bimanual_transfer_item")
        if self.task_name == 'OpenDrawer':
            raise NotImplementedError(f"determine_dominant_hand not made for  “{self.task_name}” ")
            ##### VLM for determining acting arm
            # compute the angle between the look-at vector and the bottom drawer handle (normals vector)
            bottom_drawer_handle_normals = self.get_3d_normals_by_name('bottom drawer handle')
            bottom_drawer_handle_normals_avg = np.mean(bottom_drawer_handle_normals, axis=0)
            self.angle_bottom_drawer_handle_and_lookat = math.degrees(np.arccos(self.lookat_vectors['front'][0] * bottom_drawer_handle_normals_avg[0] + self.lookat_vectors['front'][1] * bottom_drawer_handle_normals_avg[1] + self.lookat_vectors['front'][2] * bottom_drawer_handle_normals_avg[2]))

            # 135 is the threshold angle used to determine which arm to use
            if self.angle_bottom_drawer_handle_and_lookat >= 135:
                self._dominant_arm = 'right'
            else:
                self._dominant_arm = 'left'

            # for debugging... self._dominant_arm_for_ep_reset contains the ground truth dominant arm
            # self._dominant_arm = self._dominant_arm_for_ep_reset

            print(f'Chosen dominant arm is {self._dominant_arm}')

            ##### ChatGPT for determining acting arm
            # self.rlbench_env._highres_front_cam.handle_explicitly()
            # highres_front_rgb = self.rlbench_env._highres_front_cam.capture_rgb()
            # highres_front_rgb = np.clip((highres_front_rgb * 255.).astype(np.uint8), 0, 255)
            # # highres_front_rgb = Image.fromarray(highres_front_rgb) # for debugging
            # # highres_front_rgb.show() # for debugging

            # prompt = "This picture shows a simulation environment with two robotic arms and a drawer in a tabletop environment. The image is taken by a front camera looking at the two robotic arms, a drawer with three handles (top, middle, and bottom), and a table. One robotic arm is positioned on the left side of the table. The other robotic arm is placed on the right side of the table. The drawer is positioned in between the two robotic arms. The robotic arms are fixed onto the table, but the drawer can randomly spawn on the table between the two robotic arms. The drawer could spawn in a different location with a different orientation. Ignore the background walls and floor. Pay attention to the orientation of the drawer with respect to the two robotic arms and the table. Describe what's in the image in detail. Then, tell me in a new sentence that which robotic arm is the front of the drawer (with top, middle, and bottom drawer handles) facing without other texts."
            # # obs = self.rlbench_env._scene.get_observation()
            # # dominant_arm = self._determine_dominant_hand_LLM_helper(obs.front_rgb, prompt) # front rgb image (128 x 128)
            # dominant_arm = self._determine_dominant_hand_LLM_helper(highres_front_rgb, prompt) # high-res front rgb image (512 x 512)
            # self._dominant_arm = dominant_arm
            # print(f'\n\n !!!!!! ChatGPT dominant arm prediction: {dominant_arm}')
        elif self.task_name == 'PutItemInDrawer':
            raise NotImplementedError(f"determine_dominant_hand not made for  “{self.task_name}” ")
            ##### VLM for determining acting arm
            # compute the angle between the look-at vector and the top drawer handle (normals vector)
            top_drawer_handle_normals = self.get_3d_normals_by_name('top drawer handle')
            top_drawer_handle_normals_avg = np.mean(top_drawer_handle_normals, axis=0)
            self.angle_top_drawer_handle_and_lookat = math.degrees(np.arccos(self.lookat_vectors['front'][0] * top_drawer_handle_normals_avg[0] + self.lookat_vectors['front'][1] * top_drawer_handle_normals_avg[1] + self.lookat_vectors['front'][2] * top_drawer_handle_normals_avg[2]))

            # 134 is the threshold angle used to determine which arm to use
            if self.angle_top_drawer_handle_and_lookat >= 134:
                self._dominant_arm = 'left'
            else:
                self._dominant_arm = 'right'
            # print('\n\n\n\n\n!!!!!!!!!! self.angle_top_drawer_handle_and_lookat: ', self.angle_top_drawer_handle_and_lookat)

            # for debugging... self._dominant_arm_for_ep_reset contains the ground truth dominant arm
            # self._dominant_arm = self._dominant_arm_for_ep_reset

            print(f'Chosen dominant arm is {self._dominant_arm}')
        elif self.task_name == 'OpenJar':
            raise NotImplementedError(f"determine_dominant_hand not made for  “{self.task_name}” ")
            jar_points = self.get_3d_points_by_name('jar')
            jar_points_avg = np.mean(jar_points, axis=0)
            robot_right_position, robot_left_position = self.get_robot_arms_position()

            jar_to_right_arm_dist = math.dist(jar_points_avg, robot_right_position)
            jar_to_left_arm_dist = math.dist(jar_points_avg, robot_left_position)

            if jar_to_right_arm_dist < jar_to_left_arm_dist:
                # jar is closer to the robot arm on the right
                self._dominant_arm = 'right'
            else:
                # jar is closer to the robot arm on the left
                self._dominant_arm = 'left'
            print('jar_to_right_arm_dist in determine_dominant_hand: ', jar_to_right_arm_dist)
            print('jar_to_left_arm_dist in determine_dominant_hand: ', jar_to_left_arm_dist)
            print('determine_dominant_hand: ', self._dominant_arm)
        elif self.task_name == 'HandOverItem':
            raise NotImplementedError(f"determine_dominant_hand not made for  “{self.task_name}” ")
            cube_points = self.get_3d_points_by_name('cube')
            cube_points_avg = np.mean(cube_points, axis=0)
            robot_right_position, robot_left_position = self.get_robot_arms_position()

            cube_to_right_arm_dist = math.dist(cube_points_avg, robot_right_position)
            cube_to_left_arm_dist = math.dist(cube_points_avg, robot_left_position)

            if cube_to_right_arm_dist < cube_to_left_arm_dist:
                # cube is closer to the robot arm on the right
                self._dominant_arm = 'left'
            else:
                # cube is closer to the robot arm on the left
                self._dominant_arm = 'right'
            print('cube_to_right_arm_dist in determine_dominant_hand: ', cube_to_right_arm_dist)
            print('cube_to_left_arm_dist in determine_dominant_hand: ', cube_to_left_arm_dist)
            print('determine_dominant_hand: ', self._dominant_arm)
        else:
            raise NotImplementedError
        

    def _determine_dominant_hand_LLM_helper(self, front_rgb_numpy, prompt):
        print("Now we are in the voxposer/rlbench_env.py function _determine_dominant_hand_LLM_helper, which is not working now for evarything have been under the comments")
        # openAI API Key
        api_key = "REPLACE-ME"

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string
        front_rgb = Image.fromarray(front_rgb_numpy)
        front_rgb.save(self._env_image_saved_path)
        base64_image = encode_image(self._env_image_saved_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{prompt}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        if payload in self._image_cache:
            print('(using image cache in _determine_dominant_hand_LLM_helper)')
            return self._image_cache[payload]
        else:
            try:
                content = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content']
            except:
                print('Retry OpenAI API in 5 seconds...')
                time.sleep(5) # sleep for 5 seconds
                content = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()['choices'][0]['message']['content']
            print('\n\nChatGPT full response: ', content)

            # extract the last two sentences from the response
            acting_arm = '.'.join(content.split('.')[-2:])
            if 'left' in acting_arm:
                acting_arm = 'left'
            elif 'right' in acting_arm:
                acting_arm = 'right'
            else:
                print('!!!!!!!!!!!!!!! Incorrect response from ChatGPT in _determine_dominant_hand_LLM_helper: ', content)
                # randomly assign acting arm
                rand = np.random.randint(2)
                if rand == 0:
                    acting_arm = 'right'
                else:
                    acting_arm = 'left'
                print(f'Randomly assign {content} as the acting arm')
            self._image_cache[payload] = acting_arm

        time.sleep(2) # for debugging... easier to read the prints from the terminal
        return acting_arm
    
    def get_target_object_world_coords(self, gt_target_object_world_coords=False, auto_crop=False):
        """
        NOTE: target_object_world_coords should be close to or the same as target_object_pos in scene_two_robots.py
        """
        if gt_target_object_world_coords:
            if self.task_name in ['OpenDrawer', 'PutItemInDrawer']:
                object_handle = Shape('drawer_middle').get_handle()
            elif self.task_name == 'OpenJar':
                object_handle = [Shape('jar_lid0').get_handle(), Shape('jar0').get_handle()]
            else:
                raise NotImplementedError
            target_object_world_coords = self.get_target_object_pos_by_obj_handle_front_camera(object_handle)
            return target_object_world_coords

        # option 1: get image from the front camera
        obs = self.rlbench_env._scene.get_observation()
        # ToDo：  recheck the name for the name might be different! 
        front_rgb = Image.fromarray(obs.front_rgb)
        front_rgb = front_rgb.resize((1024, 1024))
        points = self.rlbench_env._scene._cam_front.capture_pointcloud()

        # option 2: get image from a higher-resolution front camera
        # front_rgb = self.rlbench_env._scene.get_highres_front_image_in_pil()
        # get object points
        # points = self.rlbench_env._scene._highres_front_cam.capture_pointcloud()

        # target_object_world_coords, auto_crop_radius = self.rlbench_env._scene.vlm.get_target_object_world_coords(front_rgb, points, self.task_name, debug=True, auto_crop=auto_crop)
        target_object_world_coords, auto_crop_radius = self.rlbench_env._scene.vlm.get_target_object_world_coords(front_rgb, points, self.task_name, debug=False, auto_crop=auto_crop)

        return target_object_world_coords, auto_crop_radius
    
    def get_object_names(self):
        """
        Returns the names of all objects in the current task environment.

        Returns:
            list: A list of object names.
        """
        name_mapping = self.task_object_names[self.task.get_name()]
        exposed_names = [names[0] for names in name_mapping]
        return exposed_names
    