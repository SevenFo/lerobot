# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Dataset conversion utility for Isaac Lab to LeRobot format.
This tool uses unified configuration to convert Isaac Lab HDF5 datasets to LeRobot format.
"""

import argparse
import logging
import os
import shutil
from typing import Optional

import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.config_loader import find_config_file, load_task_config

logger = logging.getLogger(__name__)


def create_lerobot_dataset_from_config(
    config_path: str,
    hdf5_path: Optional[str] = None,
    output_repo_id: Optional[str] = None,
    max_episodes: Optional[int] = None,
) -> LeRobotDataset:
    """
    Create LeRobot dataset from Isaac Lab HDF5 using unified configuration.

    Args:
        config_path: Path to the task configuration file
        hdf5_path: Path to HDF5 file (overrides config if provided)
        output_repo_id: Output repository ID (overrides config if provided)
        max_episodes: Maximum episodes to convert (overrides config if provided)

    Returns:
        Created LeRobot dataset
    """
    # Load configuration
    config = load_task_config(config_path)

    # Override config values if provided
    if hdf5_path is not None:
        config.dataset_config.source_hdf5 = hdf5_path
    if output_repo_id is not None:
        config.dataset_config.name = output_repo_id
    if max_episodes is not None:
        config.dataset_config.max_episodes = max_episodes

    dataset_cfg = config.dataset_config
    observation_mapping = config.get_observation_mapping_dict()

    logger.info(f"=== Creating LeRobot dataset: {dataset_cfg.name} ===")
    logger.info(f"Source HDF5: {dataset_cfg.source_hdf5}")
    logger.info(f"Task: {dataset_cfg.task}")
    logger.info(f"FPS: {dataset_cfg.fps}")

    # Build features dictionary
    features = {
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    logger.info("\n--- Analyzing source data dimensions ---")
    if not os.path.exists(dataset_cfg.source_hdf5):
        raise FileNotFoundError(f"HDF5 file not found: {dataset_cfg.source_hdf5}")

    with h5py.File(dataset_cfg.source_hdf5, "r") as f:
        demo_1 = f["data/demo_1"]
        obs_data = demo_1["obs"]
        actions_data = demo_1["actions"][()]

        logger.info(f"Available environment observation keys: {list(obs_data.keys())}")

        state_feature_dims = {}
        available_env_keys = set(obs_data.keys())

        for env_key, mapping in config.observation_mapping.items():
            if env_key not in available_env_keys:
                logger.warning(
                    f"Environment key '{env_key}' not found in HDF5, skipping."
                )
                continue

            policy_key = mapping.policy_key
            data_slice = mapping.slice
            data_type = mapping.data_type

            if data_type == "image":
                img_shape_hwc = obs_data[env_key].shape[1:]
                features[policy_key] = {
                    "dtype": "image",
                    "shape": img_shape_hwc,
                    "names": ["height", "width", "channels"],
                }
                logger.info(
                    f"📷 Image: {env_key} -> {policy_key} | Shape(H,W,C): {img_shape_hwc}"
                )

            elif data_type == "state":
                if policy_key not in state_feature_dims:
                    state_feature_dims[policy_key] = 0

                if data_slice:
                    dim = data_slice[1] - data_slice[0]
                else:
                    data_shape = obs_data[env_key].shape
                    dim = data_shape[1] if len(data_shape) > 1 else 1

                state_feature_dims[policy_key] += dim
                logger.info(f"📊 State: {env_key} -> {policy_key} | Dim: {dim}")

        # Add action feature
        action_dim = actions_data.shape[1]
        features["action"] = {"dtype": "float32", "shape": (action_dim,), "names": None}

        # Add state features
        for policy_key, dim in state_feature_dims.items():
            features[policy_key] = {"dtype": "float32", "shape": (dim,), "names": None}

        # Apply feature overrides
        for feature_key, override in config.feature_overrides.items():
            if feature_key in features:
                features[feature_key].update(override)
                logger.info(f"🔧 Override applied to {feature_key}: {override}")

        logger.info("\n📐 Final policy features:")
        for key, value in features.items():
            logger.info(f"  - {key}: {value}")

    # Create dataset
    logger.info("\n--- Creating LeRobot dataset ---")
    dataset_root = os.path.join(dataset_cfg.output_root, dataset_cfg.name)
    if os.path.exists(dataset_root):
        logger.warning(f"Target path {dataset_root} exists, removing...")
        shutil.rmtree(dataset_root)

    dataset = LeRobotDataset.create(
        repo_id=dataset_cfg.name,
        root=dataset_root,
        features=features,
        fps=dataset_cfg.fps,
        use_videos=False,
    )
    logger.info(f"✅ LeRobotDataset created at {dataset_root}")

    # Convert data
    logger.info("\n--- Converting and filling data ---")
    converted_episodes = 0

    with h5py.File(dataset_cfg.source_hdf5, "r") as f:
        data_group = f["data"]
        demo_names = sorted(
            [name for name in data_group if name.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )

        if dataset_cfg.max_episodes:
            demo_names = demo_names[: dataset_cfg.max_episodes]

        for demo_name in demo_names:
            logger.info(f"  Converting {demo_name}...")
            demo = data_group[demo_name]

            if "obs" not in demo or "actions" not in demo:
                logger.warning(f"  Skipping {demo_name}: missing obs or actions")
                continue

            obs_data_h5 = demo["obs"]
            actions_data = np.array(demo["actions"])
            timesteps = actions_data.shape[0]

            # Organize data by policy feature
            state_data_to_concat = {}
            image_data = {}

            for env_key, mapping in config.observation_mapping.items():
                if env_key not in obs_data_h5:
                    continue

                policy_key = mapping.policy_key
                data_slice = mapping.slice
                data_type = mapping.data_type

                if data_type == "image":
                    img_array = np.array(obs_data_h5[env_key])
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                    if img_array.shape[-1] not in [1, 3, 4]:
                        img_array = np.transpose(img_array, (0, 2, 3, 1))
                    image_data[policy_key] = img_array

                elif data_type == "state":
                    data = np.array(obs_data_h5[env_key])
                    if data_slice:
                        data = data[:, data_slice[0] : data_slice[1]]
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)

                    if policy_key not in state_data_to_concat:
                        state_data_to_concat[policy_key] = []
                    state_data_to_concat[policy_key].append(data)

            # Concatenate state data
            concatenated_state_data = {}
            for policy_key, data_list in state_data_to_concat.items():
                if data_list:
                    concatenated_state_data[policy_key] = np.concatenate(
                        data_list, axis=1
                    )

            # Add frames to dataset
            for t in range(timesteps):
                frame_data = {
                    "action": actions_data[t],
                    "next.done": np.array([t == timesteps - 1], dtype=bool),
                }

                # Add state data
                for key, data in concatenated_state_data.items():
                    frame_data[key] = data[t]

                # Add image data
                for key, data in image_data.items():
                    frame_data[key] = data[t]

                dataset.add_frame(frame_data, dataset_cfg.task)

            dataset.save_episode()
            converted_episodes += 1

            if (
                dataset_cfg.max_episodes
                and converted_episodes >= dataset_cfg.max_episodes
            ):
                break

    logger.info(f"\n✅ Successfully converted {converted_episodes} episodes")
    logger.info(f"💾 Dataset saved at: {dataset.root}")

    return dataset


def main():
    """Main function for dataset conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 to LeRobot dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="lerobot_task_config.yaml",
        help="Path to task configuration file",
    )
    parser.add_argument(
        "--hdf5_path", type=str, help="Override HDF5 file path from config"
    )
    parser.add_argument(
        "--output_repo_id", type=str, help="Override output repository ID from config"
    )
    parser.add_argument(
        "--max_episodes", type=int, help="Override maximum episodes from config"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    try:
        # Find config file
        if not os.path.exists(args.config):
            config_path = find_config_file(args.config)
        else:
            config_path = args.config

        # Convert dataset
        dataset = create_lerobot_dataset_from_config(
            config_path=config_path,
            hdf5_path=args.hdf5_path,
            output_repo_id=args.output_repo_id,
            max_episodes=args.max_episodes,
        )

        logger.info("🎉 Dataset conversion completed successfully!")

    except Exception as e:
        logger.error(f"❌ Dataset conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
