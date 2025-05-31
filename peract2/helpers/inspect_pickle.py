import pickle
import pprint # 导入 pprint 模块，用于更美观地打印复杂数据结构
import os 

DATA_ROOT = "/home/hanwen/"
TASK_NAME = "bimanual_transfer_item"


# 替换成你的 pickle 文件的实际路径
pickle_file_name = 'low_dim_obs.pkl' 

full_pickle_path = os.path.join(
    DATA_ROOT, 
    "data", 
    "train", 
    TASK_NAME, 
    "all_variations", 
    "episodes", 
    "episode0",
    pickle_file_name
)

# --- 安全警告 ---
# 确保你完全信任这个 pickle 文件的来源。
# 加载不可信的 pickle 文件可能执行恶意代码。
# 因为这是你自己创建的，所以这里是安全的。

try:
    from rlbench.demo import Demo
    from rlbench.observation import BimanualObservation, UnimanualObservation, Observation
except ImportError:
    print("Waring: Can not inplement RLBench! ")
    # 定义占位符以便 isinstance 检查不会报错 (这不是最佳实践，但有助于演示)
    class BimanualObservation: pass
    class UnimanualObservation: pass
    class Observation: pass

try:
    with open(full_pickle_path, 'rb') as f:
        demo_data = pickle.load(f)

    print(f"--- successful loading pickle file  ---")
    print(f"Object Type : {type(demo_data)}")
    
    # 检查 Demo 对象的长度 (多少个时间步)
    demo_length = len(demo_data)
    print(f"Demo includes  {demo_length} step of  observations")

    if demo_length > 0:
        # 获取第一个观测步骤
        first_obs = demo_data[0]
        print(f"\n--- First Observation ---")
        print(f"Frist Observation Type : {type(first_obs)}")

        # 检查是否是 BimanualObservation (根据任务名称推断)
        # 注意：如果上面导入失败，isinstance 可能无法正常工作，
        # 我们更多地依赖于直接访问属性并处理可能的 AttributeError。
        
        print("\n--- Check basic status ---")
        if hasattr(first_obs, 'task_low_dim_state'):
            print("task_low_dim_state:")
            pprint.pprint(first_obs.task_low_dim_state)
        else:
            print("No 'task_low_dim_state' ")

        if hasattr(first_obs, 'perception_data'):
            print("\nperception_data (Keys):")
            pprint.pprint(first_obs.perception_data.keys())
        else:
            print("No 'perception_data'")

        # 检查双臂属性 (Bimanual)
        if hasattr(first_obs, 'left') and first_obs.left is not None:
            print("\n--- (Left Arm) ---")
            left_arm = first_obs.left
            print(f"  Left Arm Type : {type(left_arm)}")
            if hasattr(left_arm, 'joint_positions'):
                print("  joint_positions:")
                pprint.pprint(left_arm.joint_positions)
            if hasattr(left_arm, 'gripper_pose'):
                print("  gripper_pose:")
                pprint.pprint(left_arm.gripper_pose)
            if hasattr(left_arm, 'gripper_open'):
                print(f"  gripper_open: {left_arm.gripper_open}")
        else:
            print("\n No  'left' attribute or  None。")

        if hasattr(first_obs, 'right') and first_obs.right is not None:
            print("\n--- (Right Arm) ---")
            right_arm = first_obs.right
            print(f"  Right Arm type : {type(right_arm)}")
            if hasattr(right_arm, 'joint_positions'):
                print("  joint_positions:")
                pprint.pprint(right_arm.joint_positions)
            if hasattr(right_arm, 'gripper_pose'):
                print("  gripper_pose:")
                pprint.pprint(right_arm.gripper_pose)
            if hasattr(right_arm, 'gripper_open'):
                print(f"  gripper_open: {right_arm.gripper_open}")
        else:
            print("\n No  'right' attrtibute or  None。")
            
        # 尝试调用 get_low_dim_data (如果适用)
        if hasattr(first_obs, 'get_low_dim_data'):
            print("\n--- get_low_dim_data ---")
            try:
                # 注意 BimanualObservation 的 get_low_dim_data 需要一个参数
                if hasattr(first_obs, 'right'): # 假设我们看右臂的
                    print("----    Right low_dim_data:    ----")
                    pprint.pprint(first_obs.get_low_dim_data(first_obs.right))
                    if hasattr(first_obs, 'left'): # 假设我们看左臂的
                        print("----    Left low_dim_data:    ----")
                        pprint.pprint(first_obs.get_low_dim_data(first_obs.left))
                elif not first_obs.is_bimanual: # 如果是 Unimanual
                    print("\n ----    low_dim_data:    ----")
                    pprint.pprint(first_obs.get_low_dim_data())
            except Exception as e:
                print(f"  Get get_low_dim_data 出错: {e}")
        
        if hasattr(first_obs, 'target_object_pos'):
            print("\n ----    target_object_pos:    ----")
            # 它是一个 NumPy 数组，用 pprint 打印
            pprint.pprint(first_obs.target_object_pos) 
        else:
            print("No 'target_object_pos' ")

        if hasattr(first_obs, 'auto_crop_radius'):
            # 它是一个 float，直接打印即可
            print(f"\n ----    auto_crop_radius: {first_obs.auto_crop_radius}    ----")
        else:
            print("No 'auto_crop_radius' 属性。")


except FileNotFoundError:
    print(f"错误：找不到文件 '{full_pickle_path}'")
except ImportError as e:
    print(f"错误：加载 Pickle 文件失败，可能是因为找不到 RLBench 的类定义。请确保 RLBench 已正确安装在你的 Python 环境中。 {e}")
except Exception as e:
    print(f"加载或检查 Pickle 文件时发生错误：{e}")