import numpy as np
import pyrender
import os
from utils import visualise_voxel
from click_prompt import filepath_option
import click

# Save the voxel in npz and visualize on local


@click.command()
@click.option("--file-path", default="/home/hanwen/Downloads/left_voxel_step0.npz", help="Path to the voxel .npz file.")
def main(file_path):
    file_path = os.path.expanduser(file_path)  # Ensure file path is absolute and valid

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    image = visualize_voxel_from_file(file_path, show=True)

def load_voxel_data(file_path: str):
    data = np.load(file_path)

    voxel_data = {
        "voxel_grid": data["voxel_grid"] if "voxel_grid" in data else None,
        "q_attention": data["q_attention"] if "q_attention" in data else None,
        "highlight_coordinate": data["highlight_coordinate"] if "highlight_coordinate" in data else None,
        "highlight_gt_coordinate": data["highlight_gt_coordinate"] if "highlight_gt_coordinate" in data else None,
    }

    if voxel_data["voxel_grid"] is None:
        raise ValueError("voxel_grid data not found in the .npz file!")

    return voxel_data


def visualize_voxel_from_file(file_path: str, show: bool = True, voxel_size: float = 0.1):
    voxel_data = load_voxel_data(file_path)

    # Create an offscreen renderer for visualization
    offscreen_renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)

    # Generate visualization
    image = visualise_voxel(
        voxel_grid=voxel_data["voxel_grid"],
        q_attention=voxel_data["q_attention"],
        highlight_coordinate=voxel_data["highlight_coordinate"],
        highlight_gt_coordinate=voxel_data["highlight_gt_coordinate"],
        show=show,
    )

    print("Voxel visualization completed successfully!")


if __name__ == "__main__":
    main()