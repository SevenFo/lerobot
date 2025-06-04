#!/usr/bin/env python3
"""
Convert HDF5 robot assembly dataset to LeRobotDataset format
"""

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Note: calculate_episode_data_index not needed for this simplified conversion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_features_config():
    """Create features configuration for the robot assembly dataset"""
    return {
        "observation.joint_pos": {
            "dtype": "float32",
            "shape": (9,),
            "names": [f"joint_{i}" for i in range(9)],
        },
        "observation.joint_vel": {
            "dtype": "float32",
            "shape": (9,),
            "names": [f"joint_vel_{i}" for i in range(9)],
        },
        "observation.eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.eef_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["qx", "qy", "qz", "qw"],
        },
        "observation.gripper_pos": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_finger", "right_finger"],
        },
        "observation.assemble_inner_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.assemble_inner_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["qx", "qy", "qz", "qw"],
        },
        "observation.assemble_outer_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.assemble_outer_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["qx", "qy", "qz", "qw"],
        },
        "observation.object_state": {
            "dtype": "float32",
            "shape": (20,),
            "names": [f"object_feature_{i}" for i in range(20)],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
        },
    }


def convert_hdf5_to_lerobot(input_file, output_dir, fps=30):
    """Convert HDF5 dataset to LeRobot format"""

    logger.info(f"Converting {input_file} to LeRobot format")

    # Create features configuration
    features = create_features_config()

    with h5py.File(input_file, "r") as f:
        # Get all demos
        demos = [key for key in f["data"] if key.startswith("demo_")]
        logger.info(f"Found {len(demos)} demonstrations")

        # Get dataset dimensions
        first_demo = f["data"][demos[0]]
        timesteps_per_demo = first_demo["actions"].shape[0]
        total_frames = len(demos) * timesteps_per_demo

        logger.info(f"Timesteps per demo: {timesteps_per_demo}")
        logger.info(f"Total frames: {total_frames}")

        # Create LeRobot dataset first
        dataset = LeRobotDataset.create(
            root=Path("/home/ps/Projects/lerobot/examples/").joinpath(str(output_dir)),
            repo_id="robot_assembly_dataset",
            fps=fps,
            features=features,
            robot_type="robot_assembly",
        )

        # Convert data
        frame_idx = 0
        for demo_idx, demo_key in enumerate(demos):
            logger.info(f"Processing {demo_key} ({demo_idx + 1}/{len(demos)})")

            demo_data = f["data"][demo_key]

            # Get all data for this demo
            actions = demo_data["actions"][:]
            obs = demo_data["obs"]

            for t in range(timesteps_per_demo):
                # Prepare data dict
                data_dict = {
                    "action": actions[t].astype(np.float32),
                    "observation.joint_pos": obs["joint_pos"][t].astype(np.float32),
                    "observation.joint_vel": obs["joint_vel"][t].astype(np.float32),
                    "observation.eef_pos": obs["eef_pos"][t].astype(np.float32),
                    "observation.eef_quat": obs["eef_quat"][t].astype(np.float32),
                    "observation.gripper_pos": obs["gripper_pos"][t].astype(np.float32),
                    "observation.assemble_inner_pos": obs["assemble_inner_positions"][
                        t
                    ].astype(np.float32),
                    "observation.assemble_inner_quat": obs[
                        "assemble_inner_orientations"
                    ][t].astype(np.float32),
                    "observation.assemble_outer_pos": obs["assemble_outer_positions"][
                        t
                    ].astype(np.float32),
                    "observation.assemble_outer_quat": obs[
                        "assemble_outer_orientations"
                    ][t].astype(np.float32),
                    "observation.object_state": obs["object"][t].astype(np.float32),
                    "timestamp": np.array(
                        [t / fps], dtype=np.float32
                    ),  # 如果设置了这个参数，将会采用这里的timestampe，否则使用默认值（利用fps计算）
                    "task": "robot_assembly",
                }

                # Add to dataset
                dataset.add_frame(data_dict)
                frame_idx += 1

            dataset.save_episode()

            if (demo_idx + 1) % 10 == 0:
                logger.info(f"Processed {demo_idx + 1}/{len(demos)} demos")

        # Save dataset
        logger.info(f"Dataset saved to {output_dir}")

        return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 to LeRobot format")
    parser.add_argument("input_file", type=str, help="Path to input HDF5 file")
    parser.add_argument(
        "output_dir", type=str, help="Output directory for LeRobot dataset"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    convert_hdf5_to_lerobot(input_path, output_path, args.fps)


if __name__ == "__main__":
    main()
