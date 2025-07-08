# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Auto-generate EnvConfig from LeRobot dataset metainfo.
This utility helps convert LeRobot dataset metadata into Isaac Lab compatible EnvConfig.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Define local versions to avoid dependency issues
class FeatureType(Enum):
    """Types of features in the dataset."""

    VISUAL = "visual"
    STATE = "state"
    ENV = "env"
    ACTION = "action"


@dataclass
class PolicyFeature:
    """Policy feature configuration."""

    type: FeatureType
    shape: tuple


@dataclass
class DatasetMetaInfo:
    """Container for dataset metadata information."""

    codebase_version: str
    robot_type: Optional[str]
    total_episodes: int
    total_frames: int
    total_tasks: int
    fps: int
    features: Dict[str, dict]
    tasks: List[Dict[str, str]]

    @classmethod
    def from_meta_dir(cls, meta_dir: str) -> "DatasetMetaInfo":
        """Load dataset metadata from meta directory."""
        meta_path = Path(meta_dir)

        # Load info.json
        with open(meta_path / "info.json", "r") as f:
            info_data = json.load(f)

        # Load tasks.jsonl
        tasks = []
        tasks_file = meta_path / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))

        return cls(
            codebase_version=info_data["codebase_version"],
            robot_type=info_data.get("robot_type"),
            total_episodes=info_data["total_episodes"],
            total_frames=info_data["total_frames"],
            total_tasks=info_data["total_tasks"],
            fps=info_data["fps"],
            features=info_data["features"],
            tasks=tasks,
        )


def infer_feature_type(feature_name: str, feature_info: dict) -> FeatureType:
    """
    Infer the PolicyFeature type from feature name and info.

    Args:
        feature_name: Name of the feature
        feature_info: Feature metadata from dataset

    Returns:
        Inferred FeatureType
    """
    # Check if it's an image feature
    if feature_info.get("dtype") == "image":
        return FeatureType.VISUAL

    # Check if it's action
    if feature_name == "action":
        return FeatureType.ACTION

    # Check if it's observation state
    if feature_name.startswith("observation.state"):
        return FeatureType.STATE

    # Check if it's environment state/objects
    if feature_name.startswith("observation.") and any(
        keyword in feature_name.lower()
        for keyword in ["env", "object", "box", "spanner", "environment"]
    ):
        return FeatureType.ENV

    # Check if it's visual observation
    if feature_name.startswith("observation.images"):
        return FeatureType.VISUAL

    # Default to state for other observation features
    if feature_name.startswith("observation."):
        return FeatureType.STATE

    # Default fallback
    return FeatureType.STATE


def create_policy_features(dataset_meta: DatasetMetaInfo) -> Dict[str, PolicyFeature]:
    """
    Create PolicyFeature dictionary from dataset metadata.

    Args:
        dataset_meta: Dataset metadata

    Returns:
        Dictionary of PolicyFeatures
    """
    policy_features = {}

    for feature_name, feature_info in dataset_meta.features.items():
        # Skip internal features
        if feature_name in [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "next.done",
        ]:
            continue

        # Get feature type
        feature_type = infer_feature_type(feature_name, feature_info)

        # Get feature shape
        shape = tuple(feature_info["shape"])

        # Create PolicyFeature
        policy_features[feature_name] = PolicyFeature(type=feature_type, shape=shape)

        logger.info(
            f"Created feature: {feature_name} -> {feature_type}, shape: {shape}"
        )

    return policy_features


def create_features_map(policy_features: Dict[str, PolicyFeature]) -> Dict[str, str]:
    """
    Create features mapping from policy features.

    Args:
        policy_features: Dictionary of PolicyFeatures

    Returns:
        Dictionary mapping feature names to policy keys
    """
    features_map = {}

    for feature_name, feature in policy_features.items():
        if feature.type == FeatureType.ACTION:
            features_map[feature_name] = "action"
        elif feature.type == FeatureType.VISUAL:
            # Map image features to observation keys
            if "top" in feature_name:
                features_map[feature_name] = "observation.images.top"
            elif "side" in feature_name:
                features_map[feature_name] = "observation.images.side"
            elif "wrist" in feature_name:
                features_map[feature_name] = "observation.images.wrist"
            else:
                features_map[feature_name] = "observation.images"
        elif feature.type == FeatureType.STATE:
            features_map[feature_name] = "observation.state"
        elif feature.type == FeatureType.ENV:
            features_map[feature_name] = "observation.environment_state"
        else:
            # Default mapping
            features_map[feature_name] = feature_name

    return features_map


@dataclass
class GeneratedEnvConfig:
    """Simple container for generated environment configuration."""

    task: str
    fps: int
    episode_length: int
    obs_type: str
    render_mode: str
    features: Dict[str, PolicyFeature]
    features_map: Dict[str, str]

    @property
    def gym_kwargs(self) -> Dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


def generate_envconfig_from_meta(
    meta_dir: str, config_name: str = "auto_generated"
) -> GeneratedEnvConfig:
    """
    Generate EnvConfig from dataset metadata directory.

    Args:
        meta_dir: Path to dataset meta directory
        config_name: Name for the generated config class

    Returns:
        Generated EnvConfig instance
    """
    # Load dataset metadata
    dataset_meta = DatasetMetaInfo.from_meta_dir(meta_dir)

    # Create policy features
    policy_features = create_policy_features(dataset_meta)

    # Create features map
    features_map = create_features_map(policy_features)

    # Get task information
    task_name = "Isaac-Manipulation-v0"  # Default task
    if dataset_meta.tasks:
        task_name = dataset_meta.tasks[0].get("task", task_name)

    # Check if we have image features
    has_images = any(f.type == FeatureType.VISUAL for f in policy_features.values())
    obs_type = "pixels" if has_images else "state_environment_state"

    # Create config instance
    config_instance = GeneratedEnvConfig(
        task=task_name,
        fps=dataset_meta.fps,
        episode_length=max(
            1000, dataset_meta.total_frames // dataset_meta.total_episodes
        ),
        obs_type=obs_type,
        render_mode="rgb_array",
        features=policy_features,
        features_map=features_map,
    )

    logger.info(f"Generated EnvConfig with {len(policy_features)} features")
    logger.info(f"Task: {config_instance.task}")
    logger.info(f"FPS: {config_instance.fps}")
    logger.info(f"Episode length: {config_instance.episode_length}")
    logger.info(f"Observation type: {config_instance.obs_type}")
    logger.info(f"Features: {list(policy_features.keys())}")

    return config_instance


def save_envconfig_code(envconfig: GeneratedEnvConfig, output_file: str):
    """
    Save the generated EnvConfig as Python code.

    Args:
        envconfig: EnvConfig instance
        output_file: Path to save the code
    """
    code_template = '''# Auto-generated EnvConfig from dataset metadata
# Generated by envconfig_generator.py

from dataclasses import dataclass, field
from typing import Dict
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig


@EnvConfig.register_subclass("auto_generated_isaac")
@dataclass
class AutoGeneratedEnvConfig(EnvConfig):
    """Auto-generated configuration for Isaac Lab environments."""
    
    task: str = "{task}"
    fps: int = {fps}
    episode_length: int = {episode_length}
    obs_type: str = "{obs_type}"
    render_mode: str = "{render_mode}"
    
    # Auto-generated features
    features: Dict[str, PolicyFeature] = field(
        default_factory=lambda: {{
{features_code}
        }}
    )
    
    # Auto-generated features map
    features_map: Dict[str, str] = field(
        default_factory=lambda: {{
{features_map_code}
        }}
    )
    
    def __post_init__(self):
        # Check if we have image features
        has_images = any(f.type == FeatureType.VISUAL for f in self.features.values())
        if has_images:
            self.obs_type = "pixels"
            
    @property
    def gym_kwargs(self) -> Dict:
        return {{
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }}
'''

    # Generate features code
    features_code = []
    for name, feature in envconfig.features.items():
        features_code.append(
            f'            "{name}": PolicyFeature(type=FeatureType.{feature.type.name}, shape={feature.shape}),'
        )

    # Generate features map code
    features_map_code = []
    for env_key, policy_key in envconfig.features_map.items():
        features_map_code.append(f'            "{env_key}": "{policy_key}",')

    # Fill template
    code = code_template.format(
        task=envconfig.task,
        fps=envconfig.fps,
        episode_length=envconfig.episode_length,
        obs_type=envconfig.obs_type,
        render_mode=envconfig.render_mode,
        features_code="\n".join(features_code),
        features_map_code="\n".join(features_map_code),
    )

    # Save to file
    with open(output_file, "w") as f:
        f.write(code)

    logger.info(f"Saved EnvConfig code to {output_file}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate EnvConfig from dataset metadata"
    )
    parser.add_argument(
        "--meta_dir", required=True, help="Path to dataset meta directory"
    )
    parser.add_argument("--output", help="Output file for generated code")
    parser.add_argument(
        "--config_name", default="auto_generated", help="Name for the config class"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Generate config
    envconfig = generate_envconfig_from_meta(args.meta_dir, args.config_name)

    # Save code if requested
    if args.output:
        save_envconfig_code(envconfig, args.output)

    print("EnvConfig generated successfully!")
    print(f"Features: {list(envconfig.features.keys())}")
    print(f"Features map: {envconfig.features_map}")
