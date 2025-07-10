#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Universal dataset conversion tool for Isaac Lab to LeRobot format.
This script uses the unified configuration system to convert datasets.
"""

import argparse
import logging
import os
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.config_loader import find_config_file, load_task_config
from lerobot.utils.dataset_converter import create_lerobot_dataset_from_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main conversion function."""
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 datasets to LeRobot format using unified configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to unified task configuration file (YAML). Example: 'configs/lerobot_task_config.yaml'",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default=None,
        help="Path to source HDF5 file (overrides config). Example: 'assets/pressed_ori_20250708_rgb.hdf5'",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output dataset name (overrides config). Example: 'my_custom_dataset'",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert (overrides config). Example: 100",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be converted without actually converting.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="visualize the dataset after conversion ",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing dataset to continue conversion.",
    )
    parser.add_argument(
        "--force_resume",
        action="store_true",
        help="Force resume by skipping timestamp validation checks. Use this when resume fails due to synchronization issues.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Override timestamp tolerance in seconds. Default uses dataset FPS-based tolerance.",
    )

    args = parser.parse_args()

    # Setup logging level
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    try:
        # Load configuration
        if not os.path.exists(args.config):
            config_path = find_config_file(args.config)
        else:
            config_path = args.config

        logger.info(f"Loading configuration from: {config_path}")
        task_config = load_task_config(config_path)

        # Override config values from command line
        source_hdf5 = args.hdf5_path or task_config.dataset_config.source_hdf5
        output_name = args.output_name or task_config.dataset_config.name
        max_episodes = args.max_episodes or task_config.dataset_config.max_episodes

        if not args.vis:
            logger.info("=== Dataset Conversion Configuration ===")
            logger.info(f"Source HDF5: {source_hdf5}")
            logger.info(f"Output name: {output_name}")
            logger.info(f"Max episodes: {max_episodes}")
            logger.info(f"Task: {task_config.dataset_config.task}")
            logger.info(f"FPS: {task_config.dataset_config.fps}")
            logger.info(f"Output root: {task_config.dataset_config.output_root}")

            # Show observation mapping
            logger.info("\n=== Observation Mapping ===")
            for isaac_key, mapping in task_config.observation_mapping.items():
                logger.info(
                    f"  {isaac_key} -> {mapping.policy_key} ({mapping.data_type})"
                )

            if args.dry_run:
                logger.info(
                    "\n=== DRY RUN MODE - No actual conversion will be performed ==="
                )
                return

            # Check if source file exists
            if not os.path.exists(source_hdf5):
                logger.error(f"Source HDF5 file not found: {source_hdf5}")
                return

            # Create dataset
            logger.info("\n=== Starting Dataset Conversion ===")
            dataset = create_lerobot_dataset_from_config(
                config_path=config_path,
                hdf5_path=source_hdf5,
                output_repo_id=output_name,
                max_episodes=max_episodes,
                resume=args.resume,
                force_resume=args.force_resume,
                tolerance=args.tolerance,
            )

            logger.info("✅ Dataset conversion completed successfully!")
            logger.info(f"Dataset location: {dataset.root}")
            logger.info(f"Total episodes: {len(dataset)}")

            # Show dataset info
            logger.info("\n=== Dataset Information ===")
            logger.info(f"Dataset features: {list(dataset.features.keys())}")

        else:
            dataset = LeRobotDataset(
                repo_id=output_name,
                root=os.path.join(task_config.dataset_config.output_root, output_name),
                episodes=[0, 1, 2],
            )
        # Show first few samples
        logger.info("\n=== Sample Data ===")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            logger.info(f"Sample {i}:")
            for key, value in sample.items():
                if hasattr(value, "shape"):
                    logger.info(f"  {key}: {value.shape} {value.dtype} {value}")
                else:
                    logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Dataset conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
