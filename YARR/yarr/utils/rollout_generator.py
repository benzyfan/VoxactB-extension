from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from voxposer.utils import set_lmp_objects


class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False, 
                  left_arm_agent = None,
                  which_arm = None, 
                  crop_target_obj_voxel = None, 
                  crop_radius = None,
                  # voxposer_only_eval always false 
                  voxposer_only_eval = False, 
                  ep_number: int = -1, 
                  no_voxposer: bool = False, 
                  no_acting_stabilizing: bool = False, 
                  baseline_name: str = '',
                  gt_target_object_world_coords: bool = False, 
                  kwargs = None):


        task_name=env._task_class.__name__
        if kwargs is not None:
            if kwargs['eval_cfg'].method.diff_appearance and task_name in ['OpenJar', 'OpenDrawer']:
                print("We are now at yarr.rolloutgenerator in openjar/opendrawer, this acturally will never happen ")
                raise NotImplementedError
            if voxposer_only_eval:
                print("We are now at yarr.rolloutgenerator, in voxposer_only_eval, this is not being transfered!")
                raise NotImplementedError
            else:
                if crop_target_obj_voxel and not no_voxposer:
                    print("We are now at yarr.rolloutgenerator, in crop_target_obj_voxel and not no_voxposer")
                elif which_arm == 'dominant_assistive' or no_voxposer:
                    # this method is not using VoxPoser; we're just using VoxPoser's RLBench environment
                    print("We are now at yarr.rolloutgenerator, in which_arm == 'dominant_assistive' and  no_voxposer")
                    description, obs = env.reset_to_demo_voxposer(eval_demo_seed, ep_number)
                    agent.reset()
                    if left_arm_agent:
                        left_arm_agent.reset()
                    set_lmp_objects(env.lmps, env._rlbench_env.get_object_names())  # set the object names to be used by voxposer
                else:
                    #origin pipeline
                    if eval:
                        obs = env.reset_to_demo(eval_demo_seed)
                    else:
                        obs = env.reset()


                if crop_radius == 'auto':
                    auto_crop = True
                    crop_radius_local = None
                else:
                    auto_crop = False
                    crop_radius_local = crop_radius

                if crop_target_obj_voxel and which_arm == 'both':
                    print("We are now at yarr.rolloutgenerator, in crop_target_obj_voxel and which_arm == 'both', this should not happen")
                    raise NotImplementedError
                    # obs = env.get_observation()
                    # obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

                    # # get target object's coordinates and new scene bounds based on this target object
                    # target_object_world_coords, auto_crop_radius = env._rlbench_env.get_target_object_world_coords(gt_target_object_world_coords, auto_crop)
                    # if auto_crop:
                    #     crop_radius_local = auto_crop_radius
                    # new_scene_bounds = get_new_scene_bounds_based_on_crop(crop_radius_local, target_object_world_coords)

                    # # left PerAct policy + right PerAct policy
                    # for step in range(episode_length):
                    #     if step % 2 == 0:
                    #         curr_agent = left_arm_agent
                    #         which_arm = 'left'
                    #     else:
                    #         curr_agent = agent
                    #         which_arm = 'right'

                    #     prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

                    #     act_result = curr_agent.act(step_signal.value, prepped_data,
                    #                         deterministic=eval, which_arm=which_arm, new_scene_bounds=new_scene_bounds)

                    #     # Convert to np if not already
                    #     agent_obs_elems = {k: np.array(v) for k, v in
                    #                     act_result.observation_elements.items()}
                    #     extra_replay_elements = {k: np.array(v) for k, v in
                    #                             act_result.replay_elements.items()}
                    #     transition = env.step_custom_action_mode(act_result, which_arm=which_arm)
                    #     obs_tp1 = dict(transition.observation)
                    #     timeout = False
                    #     if step == episode_length - 1:
                    #         # If last transition, and not terminal, then we timed out
                    #         timeout = not transition.terminal
                    #         if timeout:
                    #             transition.terminal = True
                    #             if "needs_reset" in transition.info:
                    #                 transition.info["needs_reset"] = True

                    #     obs_and_replay_elems = {}
                    #     obs_and_replay_elems.update(obs)
                    #     obs_and_replay_elems.update(agent_obs_elems)
                    #     obs_and_replay_elems.update(extra_replay_elements)

                    #     for k in obs_history.keys():
                    #         obs_history[k].append(transition.observation[k])
                    #         obs_history[k].pop(0)

                    #     transition.info["active_task_id"] = env.active_task_id

                    #     replay_transition = ReplayTransition(
                    #         obs_and_replay_elems, act_result.action, transition.reward,
                    #         transition.terminal, timeout, summaries=transition.summaries,
                    #         info=transition.info)

                    #     if transition.terminal or timeout:
                    #         # If the agent gives us observations then we need to call act
                    #         # one last time (i.e. acting in the terminal state).
                    #         if len(act_result.observation_elements) > 0:
                    #             prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    #             act_result = curr_agent.act(step_signal.value, prepped_data,
                    #                                 deterministic=eval, which_arm=which_arm)
                    #             agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                    #                                 act_result.observation_elements.items()}
                    #             obs_tp1.update(agent_obs_elems_tp1)
                    #         replay_transition.final_observation = obs_tp1

                    #     # if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                    #     #     env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                    #     #                                                     steps=60, step_scene=True)

                    #     obs = dict(transition.observation)
                    #     yield replay_transition

                    #     if transition.info.get("needs_reset", transition.terminal):
                    #         return
                elif crop_target_obj_voxel and which_arm == 'dominant_assistive':
                    print("We are now at yarr.rolloutgenerator, in crop_target_obj_voxel and which_arm == 'dominant_assistive' ")
                    raise NotImplementedError
                elif crop_target_obj_voxel and which_arm == 'multiarm':
                    print("We are now at yarr.rolloutgenerator, in crop_target_obj_voxel and which_arm == 'multiarm', this should not happen")
                elif which_arm == 'dominant_assistive':
                    # methods using dominant_assistive
                    obs = env.get_observation()
                    obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

                    dominant_arm_agent = agent
                    assistive_arm_agent = left_arm_agent

                    dominant_arm = env.get_dominant_arm()
                    if dominant_arm == 'right':
                        assistive_arm = 'left'
                    else:
                        assistive_arm = 'right'

                    if baseline_name == 'baseline1':
                        dominant_assitive_policy = False
                    else:
                        dominant_assitive_policy = True

                    for step in range(episode_length):
                        if step % 2 == 0:
                            curr_agent = assistive_arm_agent
                            act_which_arm = assistive_arm
                        else:
                            curr_agent = dominant_arm_agent
                            act_which_arm = dominant_arm

                        prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

                        act_result = curr_agent.act(step_signal.value, prepped_data,
                                            deterministic=eval, which_arm=act_which_arm, dominant_assitive_policy=dominant_assitive_policy)

                        # Convert to np if not already
                        agent_obs_elems = {k: np.array(v) for k, v in
                                        act_result.observation_elements.items()}
                        extra_replay_elements = {k: np.array(v) for k, v in
                                                act_result.replay_elements.items()}
                        transition = env.step_custom_action_mode(act_result, which_arm=act_which_arm)
                        obs_tp1 = dict(transition.observation)
                        timeout = False
                        if step == episode_length - 1:
                            # If last transition, and not terminal, then we timed out
                            timeout = not transition.terminal
                            if timeout:
                                transition.terminal = True
                                if "needs_reset" in transition.info:
                                    transition.info["needs_reset"] = True

                        obs_and_replay_elems = {}
                        obs_and_replay_elems.update(obs)
                        obs_and_replay_elems.update(agent_obs_elems)
                        obs_and_replay_elems.update(extra_replay_elements)

                        for k in obs_history.keys():
                            obs_history[k].append(transition.observation[k])
                            obs_history[k].pop(0)

                        transition.info["active_task_id"] = env.active_task_id

                        replay_transition = ReplayTransition(
                            obs_and_replay_elems, act_result.action, transition.reward,
                            transition.terminal, timeout, summaries=transition.summaries,
                            info=transition.info)

                        if transition.terminal or timeout:
                            # If the agent gives us observations then we need to call act
                            # one last time (i.e. acting in the terminal state).
                            if len(act_result.observation_elements) > 0:
                                prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                                act_result = curr_agent.act(step_signal.value, prepped_data,
                                                    deterministic=eval, which_arm=act_which_arm, dominant_assitive_policy=dominant_assitive_policy)
                                agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                                    act_result.observation_elements.items()}
                                obs_tp1.update(agent_obs_elems_tp1)
                            replay_transition.final_observation = obs_tp1

                        # if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                        #     env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                        #                                                     steps=60, step_scene=True)

                        obs = dict(transition.observation)
                        yield replay_transition

                        if transition.info.get("needs_reset", transition.terminal):
                            return
                else:
                    if which_arm == 'both' and left_arm_agent is not None:
                        print("We are now at yarr.rolloutgenerator, in which_arm='both', this should not happen")
                        raise NotImplementedError
                        # baseline #1
                        agent.reset()
                        left_arm_agent.reset()
                        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

                        for step in range(episode_length):
                            if step % 2 == 0:
                                curr_agent = left_arm_agent
                                which_arm = 'left'
                            else:
                                curr_agent = agent
                                which_arm = 'right'

                            prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

                            act_result = curr_agent.act(step_signal.value, prepped_data,
                                                deterministic=eval, which_arm=which_arm)

                            # Convert to np if not already
                            agent_obs_elems = {k: np.array(v) for k, v in
                                            act_result.observation_elements.items()}
                            extra_replay_elements = {k: np.array(v) for k, v in
                                                    act_result.replay_elements.items()}
                            transition = env.step(act_result, which_arm=which_arm)
                            obs_tp1 = dict(transition.observation)
                            timeout = False
                            if step == episode_length - 1:
                                # If last transition, and not terminal, then we timed out
                                timeout = not transition.terminal
                                if timeout:
                                    transition.terminal = True
                                    if "needs_reset" in transition.info:
                                        transition.info["needs_reset"] = True

                            obs_and_replay_elems = {}
                            obs_and_replay_elems.update(obs)
                            obs_and_replay_elems.update(agent_obs_elems)
                            obs_and_replay_elems.update(extra_replay_elements)

                            for k in obs_history.keys():
                                obs_history[k].append(transition.observation[k])
                                obs_history[k].pop(0)

                            transition.info["active_task_id"] = env.active_task_id

                            replay_transition = ReplayTransition(
                                obs_and_replay_elems, act_result.action, transition.reward,
                                transition.terminal, timeout, summaries=transition.summaries,
                                info=transition.info)

                            if transition.terminal or timeout:
                                # If the agent gives us observations then we need to call act
                                # one last time (i.e. acting in the terminal state).
                                if len(act_result.observation_elements) > 0:
                                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                                    act_result = curr_agent.act(step_signal.value, prepped_data,
                                                        deterministic=eval, which_arm=which_arm)
                                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                                        act_result.observation_elements.items()}
                                    obs_tp1.update(agent_obs_elems_tp1)
                                replay_transition.final_observation = obs_tp1

                            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                                steps=60, step_scene=True)

                            obs = dict(transition.observation)
                            yield replay_transition

                            if transition.info.get("needs_reset", transition.terminal):
                                return
                    elif which_arm == 'both' and left_arm_agent is None:
                        # baseline #2
                        agent.reset()
                        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}

                        # implementation 1: take an action in the environment at each timestep.
                        # This only works for OpenJar because we assume, when left gripper closes, right arm can start executing, and left arm can be disabled. This works better than implementation #2.
                        if env._task._task.__class__.__name__ == 'OpenJar':
                            prev_gripper = None
                            which_arm = 'left'
                            # print('episode_length: ', episode_length) # for debugging
                            for step in range(episode_length):
                                # print('step: ', step) # for debugging
                                prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

                                act_result = agent.act(step_signal.value, prepped_data,
                                                    deterministic=eval, which_arm=which_arm)

                                # Convert to np if not already
                                agent_obs_elems = {k: np.array(v) for k, v in
                                                act_result.observation_elements.items()}
                                extra_replay_elements = {k: np.array(v) for k, v in
                                                        act_result.replay_elements.items()}
                                transition = env.step(act_result, which_arm=which_arm)
                                obs_tp1 = dict(transition.observation)
                                timeout = False
                                if step == episode_length - 1:
                                    # If last transition, and not terminal, then we timed out
                                    timeout = not transition.terminal
                                    if timeout:
                                        transition.terminal = True
                                        if "needs_reset" in transition.info:
                                            transition.info["needs_reset"] = True

                                obs_and_replay_elems = {}
                                obs_and_replay_elems.update(obs)
                                obs_and_replay_elems.update(agent_obs_elems)
                                obs_and_replay_elems.update(extra_replay_elements)

                                for k in obs_history.keys():
                                    obs_history[k].append(transition.observation[k])
                                    obs_history[k].pop(0)

                                transition.info["active_task_id"] = env.active_task_id

                                replay_transition = ReplayTransition(
                                    obs_and_replay_elems, act_result.action, transition.reward,
                                    transition.terminal, timeout, summaries=transition.summaries,
                                    info=transition.info)

                                if transition.terminal or timeout:
                                    # If the agent gives us observations then we need to call act
                                    # one last time (i.e. acting in the terminal state).
                                    if len(act_result.observation_elements) > 0:
                                        prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                                        act_result = agent.act(step_signal.value, prepped_data,
                                                            deterministic=eval, which_arm=which_arm)
                                        agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                                            act_result.observation_elements.items()}
                                        obs_tp1.update(agent_obs_elems_tp1)
                                    replay_transition.final_observation = obs_tp1

                                if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                                    env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                                    steps=60, step_scene=True)

                                obs = dict(transition.observation)
                                yield replay_transition

                                if transition.info.get("needs_reset", transition.terminal):
                                    return

                                if prev_gripper is not None and prev_gripper != act_result.action[7]:
                                    which_arm = 'right'
                                    print('Switch to right arm!!!!!!!!!!!!')
                                prev_gripper = act_result.action[7]
                        else:
                            # implementation 2: same as baseline #1 execution strategy
                            agent.reset()
                            obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
                            curr_agent = agent
                            for step in range(episode_length):
                                if step % 2 == 0:
                                    which_arm = 'left'
                                else:
                                    which_arm = 'right'

                                prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

                                act_result = curr_agent.act(step_signal.value, prepped_data,
                                                    deterministic=eval, which_arm=which_arm)

                                # Convert to np if not already
                                agent_obs_elems = {k: np.array(v) for k, v in
                                                act_result.observation_elements.items()}
                                extra_replay_elements = {k: np.array(v) for k, v in
                                                        act_result.replay_elements.items()}
                                transition = env.step(act_result, which_arm=which_arm)
                                obs_tp1 = dict(transition.observation)
                                timeout = False
                                if step == episode_length - 1:
                                    # If last transition, and not terminal, then we timed out
                                    timeout = not transition.terminal
                                    if timeout:
                                        transition.terminal = True
                                        if "needs_reset" in transition.info:
                                            transition.info["needs_reset"] = True

                                obs_and_replay_elems = {}
                                obs_and_replay_elems.update(obs)
                                obs_and_replay_elems.update(agent_obs_elems)
                                obs_and_replay_elems.update(extra_replay_elements)

                                for k in obs_history.keys():
                                    obs_history[k].append(transition.observation[k])
                                    obs_history[k].pop(0)

                                transition.info["active_task_id"] = env.active_task_id

                                replay_transition = ReplayTransition(
                                    obs_and_replay_elems, act_result.action, transition.reward,
                                    transition.terminal, timeout, summaries=transition.summaries,
                                    info=transition.info)

                                if transition.terminal or timeout:
                                    # If the agent gives us observations then we need to call act
                                    # one last time (i.e. acting in the terminal state).
                                    if len(act_result.observation_elements) > 0:
                                        prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                                        act_result = curr_agent.act(step_signal.value, prepped_data,
                                                            deterministic=eval, which_arm=which_arm)
                                        agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                                            act_result.observation_elements.items()}
                                        obs_tp1.update(agent_obs_elems_tp1)
                                    replay_transition.final_observation = obs_tp1

                                if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                                    env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                                    steps=60, step_scene=True)

                                obs = dict(transition.observation)
                                yield replay_transition

                                if transition.info.get("needs_reset", transition.terminal):
                                    return
                    else:
                        agent.reset()
                        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
                        for step in range(episode_length):

                            prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

                            act_result = agent.act(step_signal.value, prepped_data,
                                                deterministic=eval)

                            # Convert to np if not already
                            agent_obs_elems = {k: np.array(v) for k, v in
                                            act_result.observation_elements.items()}
                            extra_replay_elements = {k: np.array(v) for k, v in
                                                    act_result.replay_elements.items()}

                            transition = env.step(act_result)
                            obs_tp1 = dict(transition.observation)
                            timeout = False
                            if step == episode_length - 1:
                                # If last transition, and not terminal, then we timed out
                                timeout = not transition.terminal
                                if timeout:
                                    transition.terminal = True
                                    if "needs_reset" in transition.info:
                                        transition.info["needs_reset"] = True

                            obs_and_replay_elems = {}
                            obs_and_replay_elems.update(obs)
                            obs_and_replay_elems.update(agent_obs_elems)
                            obs_and_replay_elems.update(extra_replay_elements)

                            for k in obs_history.keys():
                                obs_history[k].append(transition.observation[k])
                                obs_history[k].pop(0)

                            transition.info["active_task_id"] = env.active_task_id

                            replay_transition = ReplayTransition(
                                obs_and_replay_elems, act_result.action, transition.reward,
                                transition.terminal, timeout, summaries=transition.summaries,
                                info=transition.info)

                            if transition.terminal or timeout:
                                # If the agent gives us observations then we need to call act
                                # one last time (i.e. acting in the terminal state).
                                if len(act_result.observation_elements) > 0:
                                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                                    act_result = agent.act(step_signal.value, prepped_data,
                                                        deterministic=eval)
                                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                                        act_result.observation_elements.items()}
                                    obs_tp1.update(agent_obs_elems_tp1)
                                replay_transition.final_observation = obs_tp1

                            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                                steps=60, step_scene=True)

                            obs = dict(transition.observation)
                            yield replay_transition

                            if transition.info.get("needs_reset", transition.terminal):
                                return
        else:
            "Kwarg is none, which means that it is for the origional pipeline"
            print("Now we are in the origional pipeline of the peract_bimanual")
            if eval:
                obs = env.reset_to_demo(eval_demo_seed)
            else:
                obs = env.reset()

            agent.reset()
            obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
            for step in range(episode_length):

                prepped_data = {k:torch.tensor(np.array(v)[None], device=self._env_device) for k, v in obs_history.items()}

                act_result = agent.act(step_signal.value, prepped_data,
                                    deterministic=eval)

                # Convert to np if not already
                agent_obs_elems = {k: np.array(v) for k, v in
                                act_result.observation_elements.items()}
                extra_replay_elements = {k: np.array(v) for k, v in
                                        act_result.replay_elements.items()}

                transition = env.step(act_result)
                obs_tp1 = dict(transition.observation)
                timeout = False
                if step == episode_length - 1:
                    # If last transition, and not terminal, then we timed out
                    timeout = not transition.terminal
                    if timeout:
                        transition.terminal = True
                        if "needs_reset" in transition.info:
                            transition.info["needs_reset"] = True

                obs_and_replay_elems = {}
                obs_and_replay_elems.update(obs)
                obs_and_replay_elems.update(agent_obs_elems)
                obs_and_replay_elems.update(extra_replay_elements)

                for k in obs_history.keys():
                    obs_history[k].append(transition.observation[k])
                    obs_history[k].pop(0)

                transition.info["active_task_id"] = env.active_task_id

                replay_transition = ReplayTransition(
                    obs_and_replay_elems, act_result.action, transition.reward,
                    transition.terminal, timeout, summaries=transition.summaries,
                    info=transition.info)

                if transition.terminal or timeout:
                    # If the agent gives us observations then we need to call act
                    # one last time (i.e. acting in the terminal state).
                    if len(act_result.observation_elements) > 0:
                        prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                        act_result = agent.act(step_signal.value, prepped_data,
                                            deterministic=eval)
                        agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                            act_result.observation_elements.items()}
                        obs_tp1.update(agent_obs_elems_tp1)
                    replay_transition.final_observation = obs_tp1

                if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                    env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                    steps=60, step_scene=True)

                obs = dict(transition.observation)
                yield replay_transition

                if transition.info.get("needs_reset", transition.terminal):
                    return