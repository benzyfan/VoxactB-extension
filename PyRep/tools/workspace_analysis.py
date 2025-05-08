from pyrep import PyRep
from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight
import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import csv
import os

class WorkspaceAnalyzer:
    def __init__(self, scene_path, num_samples=1000000, target_reachable_points=10000):
        self.scene_path = scene_path
        self.num_samples = num_samples
        self.target_reachable_points = target_reachable_points
        self.pr = None
        self.left_arm = None
        self.right_arm = None

    def initialize(self):
        try:
            self.pr = PyRep()
            print(f"Attempting to launch scene: {self.scene_path}")
            self.pr.launch(self.scene_path)
            self.pr.start()
            self.left_arm = PandaLeft(0)
            self.right_arm = PandaRight(0)
            print("Initialization successful.")
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            raise

    def shutdown(self):
        if self.pr:
            try:
                self.pr.stop()
                import time
                time.sleep(1)
                self.pr.shutdown()
                print("Shutdown successful.")
            except Exception as e:
                print(f"Shutdown failed: {str(e)}")

    def test_point_reachability(self, arm, point):
        try:
            joint_config = arm.solve_ik_via_jacobian(point)
            arm.set_joint_positions(joint_config)
            if arm.check_arm_collision():
                # print(f"Point collides with environment: {point}")  # 注释掉
                return False
            return True
        except Exception as e:
            # print(f"Point not reachable: {point}, Error: {str(e)}")  # 注释掉
            return False

    def sample_arm_workspace(self, arm, intervals):
        points = []
        total_attempts = 0
        success_count = 0
        while total_attempts < self.num_samples and len(points) < self.target_reachable_points:
            joint_angles = [
                np.random.uniform(intervals[i][0], intervals[i][1])
                for i in range(arm.get_joint_count())
            ]
            arm.set_joint_positions(joint_angles)
            tip = arm.get_tip()
            pos = tip.get_position()
            if self.test_point_reachability(arm, pos):
                points.append(pos)
                success_count += 1
            total_attempts += 1
            # 每10,000个点输出一次
            if total_attempts % 10000 == 0:
                print(f"Tested {total_attempts} points, {success_count} successed")
        print(f"Total attempts: {total_attempts}, Reachable points: {len(points)}")
        return np.array(points)

    def analyze_and_visualize(self, output_dir=None):
        if not output_dir:
            output_dir = os.path.dirname(os.path.abspath(__file__))

        cyclic_left, intervals_left = self.left_arm.get_joint_intervals()
        cyclic_right, intervals_right = self.right_arm.get_joint_intervals()

        # 采样工作空间
        points_left = self.sample_arm_workspace(self.left_arm, intervals_left)
        if len(points_left) > 3:
            hull_left = ConvexHull(points_left)
            vertices_left = points_left[hull_left.vertices]
        else:
            print("Warning: Not enough valid points for ConvexHull on Left Arm.")
            vertices_left = points_left

        points_right = self.sample_arm_workspace(self.right_arm, intervals_right)
        if len(points_right) > 3:
            hull_right = ConvexHull(points_right)
            vertices_right = points_right[hull_right.vertices]
        else:
            print("Warning: Not enough valid points for ConvexHull on Right Arm.")
            vertices_right = points_right

        # 保存所有可达点
        with open(os.path.join(output_dir, 'left_arm_reachable_points.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z'])
            for point in points_left:
                writer.writerow(point)

        with open(os.path.join(output_dir, 'right_arm_reachable_points.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z'])
            for point in points_right:
                writer.writerow(point)

        # 保存边界点
        with open(os.path.join(output_dir, 'left_arm_boundary.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z'])
            for point in vertices_left:
                writer.writerow(point)

        with open(os.path.join(output_dir, 'right_arm_boundary.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Z'])
            for point in vertices_right:
                writer.writerow(point)

        # test my points
        # test_points = [
        #     [0.0, 0.0, 0.5],
        #     [0.1, 0.1, 0.5],
        #     [0.2, -0.2, 0.5]
        # ]
        # for point in test_points:
        #     # print(f"Testing Left Arm reachability for point {point}: {self.test_point_reachability(self.left_arm, point)}")  # 注释掉
        #     # print(f"Testing Right Arm reachability for point {point}: {self.test_point_reachability(self.right_arm, point)}")  # 注释掉
        #     pass

        # 可视化
        fig_left = go.Figure()
        fig_left.add_trace(go.Scatter3d(
            x=points_left[:, 0], y=points_left[:, 1], z=points_left[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.5),
            name='Left Arm Workspace Points'
        ))
        if len(vertices_left) > 3:
            fig_left.add_trace(go.Mesh3d(
                x=vertices_left[:, 0], y=vertices_left[:, 1], z=vertices_left[:, 2],
                i=hull_left.simplices[:, 0], j=hull_left.simplices[:, 1], k=hull_left.simplices[:, 2],
                color='red',
                opacity=0.3,
                name='Left Arm Workspace Boundary'
            ))
        fig_left.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            title='Left Arm Workspace (Panda_leftArm)'
        )
        fig_left.write_html(os.path.join(output_dir, 'left_arm_workspace.html'))

        fig_right = go.Figure()
        fig_right.add_trace(go.Scatter3d(
            x=points_right[:, 0], y=points_right[:, 1], z=points_right[:, 2],
            mode='markers',
            marker=dict(size=2, color='green', opacity=0.5),
            name='Right Arm Workspace Points'
        ))
        if len(vertices_right) > 3:
            fig_right.add_trace(go.Mesh3d(
                x=vertices_right[:, 0], y=vertices_right[:, 1], z=vertices_right[:, 2],
                i=hull_right.simplices[:, 0], j=hull_right.simplices[:, 1], k=hull_right.simplices[:, 2],
                color='orange',
                opacity=0.3,
                name='Right Arm Workspace Boundary'
            ))
        fig_right.update_layout(
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            title='Right Arm Workspace (Panda_rightArm)'
        )
        fig_right.write_html(os.path.join(output_dir, 'right_arm_workspace.html'))

        print("Left Arm Boundary Points:")
        for i, point in enumerate(vertices_left):
            print(f"Point {i+1}: {point}")

        print("\nRight Arm Boundary Points:")
        for i, point in enumerate(vertices_right):
            print(f"Point {i+1}: {point}")

        return vertices_left, vertices_right

def analyze_workspace(scene_path=None, output_dir=None, num_samples=1000000):
    if scene_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '../../'))
        scene_path = os.path.join(project_root, 'RLBench', 'rlbench', 'task_design_bimanual.ttt')
        print(f"Debug: Resolved scene path: {scene_path}")

    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    analyzer = WorkspaceAnalyzer(scene_path, num_samples)
    try:
        analyzer.initialize()
        vertices_left, vertices_right = analyzer.analyze_and_visualize(output_dir)
        return vertices_left, vertices_right
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise
    finally:
        analyzer.shutdown()

if __name__ == "__main__":
    analyze_workspace()