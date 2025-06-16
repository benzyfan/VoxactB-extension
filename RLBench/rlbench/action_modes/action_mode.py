from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode
from rlbench.action_modes.arm_action_modes import BimanualJointPosition, JointPosition
from rlbench.action_modes.gripper_action_modes import GripperActionMode
from rlbench.action_modes.gripper_action_modes import BimanualGripperJointPosition, GripperJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.backend.scene import Scene
from rlbench.action_modes.gripper_action_modes import UnimanualDiscrete,SelectableUnimanualDiscrete
from rlbench.action_modes.arm_action_modes import UnimanualEndEffectorPoseViaPlanning,SelectableUnimanualArmEndEffectorPoseViaPlanning


class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise NotImplementedError('You must define your own action bounds.')


class MoveArmThenGripper(ActionMode):
    """A customizable action mode.

    The arm action is first applied, followed by the gripper action.
    """

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        ignore_collisions = bool(action[arm_act_size+1:arm_act_size+2])
        self.arm_action_mode.action(scene, arm_action, ignore_collisions)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))


class BimanualMoveArmThenGripper(MoveArmThenGripper):
    """The arm action is first applied, followed by the gripper action. """

    def action(self, scene: Scene, action: np.ndarray):

        assert(len(action) == 18)

        arm_action_size = np.prod(self.arm_action_mode.unimanual_action_shape(scene))
        ee_action_size = np.prod(self.gripper_action_mode.unimanual_action_shape(scene))
        ignore_collisions_size = 1

        action_size = arm_action_size + ee_action_size + ignore_collisions_size

        assert(action_size == 9)

        right_action = action[:action_size]
        left_action = action[action_size:]

        right_arm_action = np.array(right_action[:arm_action_size])
        left_arm_action = np.array(left_action[:arm_action_size])

        arm_action = np.concatenate([right_arm_action, left_arm_action], axis=0)        

        right_ee_action = np.array(right_action[arm_action_size:arm_action_size+ee_action_size])
        left_ee_action = np.array(left_action[arm_action_size:arm_action_size+ee_action_size])
        ee_action = np.concatenate([right_ee_action, left_ee_action], axis=0)

        right_ignore_collisions = bool(right_action[arm_action_size+ee_action_size:arm_action_size+ee_action_size+1])
        left_ignore_collisions = bool(left_action[arm_action_size+ee_action_size:arm_action_size+ee_action_size+1])
        ignore_collisions = [right_ignore_collisions, left_ignore_collisions]

        self.arm_action_mode.action(scene, arm_action, ignore_collisions)
        self.gripper_action_mode.action(scene, ee_action)


    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene)) + 2

# RLBench is highly customizable, in both observations and action modes.
# This can be a little daunting, so below we have defined some
# common action modes for you to choose from.

class JointPositionActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionActionMode, self).__init__(
            JointPosition(False), GripperJointPosition(True))

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        return np.array(7 * [-0.1] + [0.0]), np.array(7 * [0.1] + [0.04])


class BimanualJointPositionActionMode(ActionMode):

    def __init__(self, arm_action_mode=None, gripper_action_mode=None):
        arm_action_mode = arm_action_mode or BimanualJointPosition()
        gripper_action_mode = gripper_action_mode or BimanualDiscrete()

        super(BimanualJointPositionActionMode, self).__init__(arm_action_mode, gripper_action_mode)

    def action(self, scene: Scene, action: np.ndarray):

        assert(action.shape == (16,))

        
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        assert(arm_act_size == 14)

        arm_action = np.concatenate([action[0:7], action[8:15]], axis=0 )
        ee_action = np.array([action[7], action[15]])


        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)

        self.arm_action_mode.action_step(scene)

        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

    def action_bounds(self):
        """Returns the min and max of the action mode."""
        raise Exception("Not implemented yet.")

# Use for only control one arm in the bianual scene
class SelectableUnimanualMoveArmThenGripper(ActionMode):
    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False,
                 attach_grasped_objects: bool = True,
                 detach_before_open: bool = True):
        self.left_arm_mode = UnimanualEndEffectorPoseViaPlanning(
            absolute_mode=absolute_mode,
            frame=frame,
            collision_checking=collision_checking,
            robot_name='left'
        )
        self.left_gripper_mode = UnimanualDiscrete(
            attach_grasped_objects=attach_grasped_objects,
            detach_before_open=detach_before_open,
            robot_name='left'
        )
        self.right_arm_mode = UnimanualEndEffectorPoseViaPlanning(
            absolute_mode=absolute_mode,
            frame=frame,
            collision_checking=collision_checking,
            robot_name='right'
        )
        self.right_gripper_mode = UnimanualDiscrete(
            attach_grasped_objects=attach_grasped_objects,
            detach_before_open=detach_before_open,
            robot_name='right'
        )

    def action(self, scene: Scene, action: np.ndarray, which_arm: str):
        expected_shape = self.action_shape(scene)
        if action.shape != (expected_shape,):
             raise ValueError(f"Action shape mismatch. Expected {expected_shape}, got {action.shape}")

        arm_act_size = 7 
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        ignore_collisions = bool(action[arm_act_size+1:arm_act_size+2])

        if which_arm == 'left':
            print(f"Executing action on LEFT arm.")
            self.left_arm_mode.action(scene, arm_action, ignore_collisions)
            self.left_gripper_mode.action(scene, ee_action)
        elif which_arm == 'right':
            print(f"Executing action on RIGHT arm.")
            self.right_arm_mode.action(scene, arm_action, ignore_collisions)
            self.right_gripper_mode.action(scene, ee_action)
        else:
            raise ValueError(f"Invalid 'which_arm' specified: {which_arm}. Must be 'left' or 'right'.")

    def action_shape(self, scene: Scene) -> int:

        arm_shape = np.prod(self.left_arm_mode.action_shape(scene))
        gripper_shape = np.prod(self.left_gripper_mode.action_shape(scene))
        return arm_shape + gripper_shape + 1



class SelectableMoveArmThenGripper(ActionMode):
    def __init__(self,
                 arm_action_mode: SelectableUnimanualArmEndEffectorPoseViaPlanning,
                 gripper_action_mode: SelectableUnimanualDiscrete):
        assert isinstance(arm_action_mode, SelectableUnimanualArmEndEffectorPoseViaPlanning)
        assert isinstance(gripper_action_mode, SelectableUnimanualDiscrete)
        super().__init__(arm_action_mode, gripper_action_mode)

    def action(self, scene: Scene, action: np.ndarray, which_arm: str):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene)) # 7
        ee_act_size = np.prod(self.gripper_action_mode.action_shape(scene)) # 1

        if len(action) != arm_act_size + ee_act_size + 1:
             raise ValueError(f"Expected action length {arm_act_size + ee_act_size + 1}, got {len(action)}")

        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+ee_act_size])
        ignore_collisions = bool(action[arm_act_size+ee_act_size])

        self.arm_action_mode.action(
            scene, arm_action, ignore_collisions, which_arm=which_arm
        )
        self.gripper_action_mode.action(
            scene, ee_action, which_arm=which_arm
        )

    def action_shape(self, scene: Scene) -> int:
        return (np.prod(self.arm_action_mode.action_shape(scene)) +
                np.prod(self.gripper_action_mode.action_shape(scene)) + 1)