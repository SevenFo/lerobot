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
    resume: bool = False,
    force_resume: bool = False,
    tolerance: Optional[float] = None,
) -> LeRobotDataset:
    """
    Create LeRobot dataset from Isaac Lab HDF5 using unified configuration.

    Args:
        config_path: Path to the task configuration file
        hdf5_path: Path to HDF5 file (overrides config if provided)
        output_repo_id: Output repository ID (overrides config if provided)
        max_episodes: Maximum episodes to convert (overrides config if provided)
        resume: Resume conversion from existing dataset if found
        force_resume: Force resume by skipping timestamp validation checks
        tolerance: Override timestamp tolerance in seconds

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
    # observation_mapping = config.get_observation_mapping_dict()  # 暂不使用

    logger.info(f"=== Creating LeRobot dataset: {dataset_cfg.name} ===")
    logger.info(f"Source HDF5: {dataset_cfg.source_hdf5}")
    logger.info(f"Task: {dataset_cfg.task}")
    logger.info(f"FPS: {dataset_cfg.fps}")
    logger.info(f"Resume mode: {resume}")
    if force_resume:
        logger.info(f"Force resume mode: {force_resume} (will fix corrupted episodes)")
    if tolerance:
        logger.info(f"Custom tolerance: {tolerance}s")

    # 检查现有数据集
    dataset_root = os.path.join(dataset_cfg.output_root, dataset_cfg.name)
    existing_episodes = set()
    corrupted_episodes = set()

    if resume and os.path.exists(dataset_root):
        try:
            # 首先尝试正常加载现有数据集
            existing_dataset = LeRobotDataset(
                repo_id=dataset_cfg.name, root=dataset_root
            )
            existing_episodes = set(
                existing_dataset.episode_data_index["episode_index"].unique()
            )
            logger.info(
                f"🔄 Resume mode: Found existing dataset with {len(existing_episodes)} episodes"
            )
            logger.info(f"   Existing episodes: {sorted(existing_episodes)}")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Failed to load existing dataset for resume: {e}")

            # 如果启用了force_resume，尝试修复
            if force_resume:
                logger.info("🔧 Force resume mode: attempting to fix corrupted dataset")

                # 解析错误信息，提取有问题的episode
                if "episode_index" in error_msg:
                    import re

                    matches = re.findall(r"'episode_index': (\d+)", error_msg)
                    if matches:
                        corrupted_episodes = {int(ep) for ep in matches}
                        logger.info(
                            f"🔍 Detected corrupted episodes: {sorted(corrupted_episodes)}"
                        )

                        # 删除有问题的episode文件
                        logger.info(
                            f"🗑️  Deleting corrupted episode files: {sorted(corrupted_episodes)}"
                        )
                        for ep_idx in corrupted_episodes:
                            _delete_episode_files(dataset_root, ep_idx)

                        # 尝试重新加载数据集
                        try:
                            existing_dataset = LeRobotDataset(
                                repo_id=dataset_cfg.name, root=dataset_root
                            )
                            existing_episodes = set(
                                existing_dataset.episode_data_index[
                                    "episode_index"
                                ].unique()
                            )
                            logger.info(
                                "✅ Successfully loaded dataset after removing corrupted episodes"
                            )
                            logger.info(
                                f"📝 Found {len(existing_episodes)} valid episodes, will reconvert {len(corrupted_episodes)} corrupted ones"
                            )
                        except Exception as e2:
                            logger.warning(
                                f"Still failed to load dataset even after removing corrupted episodes: {e2}"
                            )
                            logger.info("   Will create completely new dataset")
                            existing_episodes = set()
                            corrupted_episodes = set()
                    else:
                        logger.warning(
                            "Could not parse episode indices from error message"
                        )
                        existing_episodes = set()
                else:
                    logger.warning(
                        "Error doesn't seem to be related to episode timestamps"
                    )
                    existing_episodes = set()
            else:
                logger.info(
                    "   Use --force_resume to automatically attempt fixing corrupted episodes"
                )
                existing_episodes = set()

    # Build features dictionary
    features = {
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    logger.info("\n--- Analyzing source data dimensions ---")
    if not os.path.exists(dataset_cfg.source_hdf5):
        raise FileNotFoundError(f"HDF5 file not found: {dataset_cfg.source_hdf5}")

    with h5py.File(dataset_cfg.source_hdf5, "r") as f:
        demo_1 = f["data/demo_1"]
        obs_data = demo_1["obs"]  # type: ignore
        actions_data = demo_1["actions"][()]  # type: ignore

        logger.info(f"Available environment observation keys: {list(obs_data.keys())}")  # type: ignore

        state_feature_dims = {}
        available_env_keys = set(obs_data.keys())  # type: ignore

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
                img_shape_hwc = obs_data[env_key].shape[1:]  # type: ignore
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
                    data_shape = obs_data[env_key].shape  # type: ignore
                    dim = data_shape[1] if len(data_shape) > 1 else 1

                state_feature_dims[policy_key] += dim
                logger.info(f"📊 State: {env_key} -> {policy_key} | Dim: {dim}")

        # Add action feature
        action_dim = actions_data.shape[1]  # type: ignore
        features["action"] = {"dtype": "float32", "shape": (action_dim,), "names": None}

        # Add state features
        for policy_key, dim in state_feature_dims.items():
            features[policy_key] = {"dtype": "float32", "shape": (dim,), "names": None}

        # Apply feature overrides
        for feature_key, override in config.feature_overrides.items():
            if feature_key in features:
                # Convert list to tuple for shape if needed
                if "shape" in override and isinstance(override["shape"], list):
                    override = override.copy()  # Don't modify original
                    override["shape"] = tuple(override["shape"])
                features[feature_key].update(override)
                logger.info(f"🔧 Override applied to {feature_key}: {override}")

        logger.info("\n📐 Final policy features:")
        for key, value in features.items():
            logger.info(f"  - {key}: {value}")

    # Create or load dataset
    logger.info("\n--- Creating/Loading LeRobot dataset ---")

    if resume and existing_episodes:
        # 加载现有数据集（可能需要跳过时间戳检查）
        if corrupted_episodes:
            logger.info(
                "🔧 Loading dataset with relaxed validation due to corrupted episodes"
            )
            dataset = LeRobotDataset(
                repo_id=dataset_cfg.name,
                root=dataset_root,
                delta_timestamps=None,  # 跳过时间戳检查
            )
        else:
            dataset = LeRobotDataset(repo_id=dataset_cfg.name, root=dataset_root)
        logger.info(f"✅ Loaded existing LeRobotDataset from {dataset_root}")
    else:
        # 创建新数据集（如果不是 resume 模式，先删除现有目录）
        if os.path.exists(dataset_root) and not resume:
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
    skipped_episodes = 0

    with h5py.File(dataset_cfg.source_hdf5, "r") as f:
        data_group = f["data"]
        demo_names = sorted(
            [name for name in data_group if name.startswith("demo_")],
            key=lambda x: int(x.split("_")[1]),
        )

        if dataset_cfg.max_episodes:
            demo_names = demo_names[: dataset_cfg.max_episodes]

        for demo_idx, demo_name in enumerate(demo_names):
            # 检查是否需要转换这个episode
            should_reconvert = demo_idx in corrupted_episodes
            already_exists = demo_idx in existing_episodes

            if resume and already_exists and not should_reconvert:
                logger.info(f"  ⏭️  Skipping {demo_name} (already exists and valid)")
                skipped_episodes += 1
                continue

            if should_reconvert:
                logger.info(f"  🔄 Reconverting {demo_name} (was corrupted)")
            else:
                logger.info(f"  Converting {demo_name}...")

            demo = data_group[demo_name]  # type: ignore

            if "obs" not in demo or "actions" not in demo:  # type: ignore
                logger.warning(f"  Skipping {demo_name}: missing obs or actions")
                continue

            obs_data_h5 = demo["obs"]  # type: ignore
            actions_data = np.array(demo["actions"])  # type: ignore
            timesteps = actions_data.shape[0]

            # Organize data by policy feature
            state_data_to_concat = {}
            image_data = {}

            for env_key, mapping in config.observation_mapping.items():
                if env_key not in obs_data_h5:  # type: ignore
                    continue

                policy_key = mapping.policy_key
                data_slice = mapping.slice
                data_type = mapping.data_type

                if data_type == "image":
                    img_array = np.array(obs_data_h5[env_key])  # type: ignore
                    if img_array.dtype != np.uint8:
                        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
                    if img_array.shape[-1] not in [1, 3, 4]:
                        img_array = np.transpose(img_array, (0, 2, 3, 1))
                    image_data[policy_key] = img_array

                elif data_type == "state":
                    data = np.array(obs_data_h5[env_key])  # type: ignore
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

    logger.info(f"\n✅ Successfully converted {converted_episodes} new episodes")
    if skipped_episodes > 0:
        logger.info(f"⏭️  Skipped {skipped_episodes} existing episodes")
    if corrupted_episodes:
        logger.info(
            f"🔄 Reconverted {len(corrupted_episodes)} corrupted episodes: {sorted(corrupted_episodes)}"
        )
    logger.info(f"💾 Dataset saved at: {dataset.root}")

    return dataset


def _delete_episode_files(dataset_root: str, episode_idx: int) -> None:
    """删除指定episode的所有相关文件"""
    from pathlib import Path

    dataset_path = Path(dataset_root)

    # 计算episode的chunk (假设chunks_size=1000，这是LeRobot的默认值)
    chunks_size = 1000
    chunk = episode_idx // chunks_size

    # 删除数据文件: data/chunk-XXX/episode_XXXXXX.parquet
    data_pattern = f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
    data_files = list(dataset_path.glob(data_pattern))
    for file_path in data_files:
        logger.info(f"  🗑️  Deleting data file: {file_path}")
        file_path.unlink(missing_ok=True)

    # 删除视频文件: videos/chunk-XXX/*/episode_XXXXXX.mp4
    video_pattern = f"videos/chunk-{chunk:03d}/*/episode_{episode_idx:06d}.mp4"
    video_files = list(dataset_path.glob(video_pattern))
    for file_path in video_files:
        logger.info(f"  🗑️  Deleting video file: {file_path}")
        file_path.unlink(missing_ok=True)

    # 删除图像文件夹: images/episode_XXXXXX/
    image_pattern = f"images/episode_{episode_idx:06d}"
    image_dirs = list(dataset_path.glob(image_pattern))
    for dir_path in image_dirs:
        if dir_path.is_dir():
            logger.info(f"  🗑️  Deleting image directory: {dir_path}")
            shutil.rmtree(dir_path)

    # 更新metadata
    _update_metadata_after_episode_deletion(dataset_root, {episode_idx})


def _update_metadata_after_episode_deletion(
    dataset_root: str, deleted_episodes: set
) -> None:
    """更新metadata文件，移除被删除episode的记录"""
    import json
    from pathlib import Path

    meta_dir = Path(dataset_root) / "meta"

    # 更新 episodes.jsonl - 移除被删除的episode记录
    episodes_file = meta_dir / "episodes.jsonl"
    if episodes_file.exists():
        logger.info("📝 Updating episodes.jsonl metadata")
        lines = []
        with open(episodes_file, "r") as f:
            for line in f:
                episode_data = json.loads(line.strip())
                if episode_data.get("episode_index") not in deleted_episodes:
                    lines.append(line)

        with open(episodes_file, "w") as f:
            f.writelines(lines)

    # 更新 info.json - 减少总episode和frame数量
    info_file = meta_dir / "info.json"
    if info_file.exists():
        logger.info("📝 Updating info.json metadata")
        with open(info_file, "r") as f:
            info = json.load(f)

        # 简单地减少episode数量，frame数量需要重新计算或者让系统自动更新
        info["total_episodes"] = max(
            0, info.get("total_episodes", 0) - len(deleted_episodes)
        )

        with open(info_file, "w") as f:
            json.dump(info, f, indent=2)


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
        "--resume",
        action="store_true",
        help="Resume conversion from existing dataset if found",
    )
    parser.add_argument(
        "--force_resume",
        action="store_true",
        help="Force resume by skipping timestamp validation checks",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Override timestamp tolerance in seconds",
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
        create_lerobot_dataset_from_config(
            config_path=config_path,
            hdf5_path=args.hdf5_path,
            output_repo_id=args.output_repo_id,
            max_episodes=args.max_episodes,
            resume=args.resume,
            force_resume=args.force_resume,
            tolerance=args.tolerance,
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
