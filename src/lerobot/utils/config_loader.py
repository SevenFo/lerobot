# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unified configuration system for Isaac Lab to LeRobot pipeline.
This module handles loading and processing configuration files that are shared
between dataset conversion, training, and inference.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str
    task: str
    fps: int
    max_episodes: Optional[int]
    source_hdf5: str
    output_root: str


@dataclass
class ObservationMapping:
    """Single observation mapping configuration."""

    policy_key: str
    slice: Optional[List[int]]
    data_type: str  # "state", "image", "action"


@dataclass
class InferenceMapping:
    """Inference mapping for finding Isaac observations."""

    isaac_candidates: List[str]
    fallback_patterns: List[str]


@dataclass
class TaskConfig:
    """Unified task configuration for Isaac Lab to LeRobot pipeline."""

    dataset_config: DatasetConfig
    observation_mapping: Dict[str, ObservationMapping]
    inference_mapping: Dict[str, InferenceMapping]
    feature_overrides: Dict[str, Dict]
    training: Dict[str, Any]
    inference: Dict[str, Any]

    @classmethod
    def load_from_file(cls, config_path: str) -> "TaskConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Parse dataset config
        dataset_data = config_data["dataset_config"]
        dataset_config = DatasetConfig(**dataset_data)

        # Parse observation mapping
        observation_mapping = {}
        for isaac_key, mapping_data in config_data["observation_mapping"].items():
            observation_mapping[isaac_key] = ObservationMapping(**mapping_data)

        # Parse inference mapping
        inference_mapping = {}
        for policy_key, mapping_data in config_data["inference_mapping"].items():
            inference_mapping[policy_key] = InferenceMapping(**mapping_data)

        # Get other sections
        feature_overrides = config_data.get("feature_overrides", {})
        training = config_data.get("training", {})
        inference = config_data.get("inference", {})

        return cls(
            dataset_config=dataset_config,
            observation_mapping=observation_mapping,
            inference_mapping=inference_mapping,
            feature_overrides=feature_overrides,
            training=training,
            inference=inference,
        )

    def get_observation_mapping_dict(self) -> Dict[str, Union[str, Dict]]:
        """Convert observation mapping to dictionary format for dataset conversion."""
        mapping_dict = {}
        for isaac_key, mapping in self.observation_mapping.items():
            if mapping.slice is not None:
                mapping_dict[isaac_key] = {
                    "policy_key": mapping.policy_key,
                    "slice": tuple(mapping.slice),
                }
            else:
                mapping_dict[isaac_key] = mapping.policy_key
        return mapping_dict

    def find_isaac_observation_key(
        self, policy_key: str, available_isaac_keys: List[str]
    ) -> Optional[str]:
        """
        Find the best matching Isaac observation key for a given policy key.

        Args:
            policy_key: The LeRobot policy feature key
            available_isaac_keys: List of available Isaac observation keys

        Returns:
            Best matching Isaac key or None if no match found
        """
        if policy_key not in self.inference_mapping:
            return None

        mapping = self.inference_mapping[policy_key]

        # First try exact matches from candidates
        for candidate in mapping.isaac_candidates:
            if candidate in available_isaac_keys:
                return candidate

        # Then try pattern matching
        for isaac_key in available_isaac_keys:
            for pattern in mapping.fallback_patterns:
                if pattern.lower() in isaac_key.lower():
                    return isaac_key

        return None

    def get_image_feature_keys(self, policy_type: str = "diffusion") -> List[str]:
        """Get list of image feature keys for the specified policy type."""
        if policy_type in self.training:
            return self.training[policy_type].get("image_features", [])
        return []

    def get_state_feature_keys(self, policy_type: str = "diffusion") -> List[str]:
        """Get list of state feature keys for the specified policy type."""
        if policy_type in self.training:
            return self.training[policy_type].get("state_features", [])
        return []

    def get_training_config(self, policy_type: str = "shared") -> Dict[str, Any]:
        """
        Get training configuration for specified policy type.

        Args:
            policy_type: Policy type ('act', 'diffusion', 'shared')

        Returns:
            Training configuration dictionary
        """
        if policy_type == "shared":
            return self.training.get("shared", {})
        elif policy_type in self.training:
            # Merge policy-specific config with shared config
            shared_config = self.training.get("shared", {})
            policy_config = self.training.get(policy_type, {})

            # Deep merge dictionaries
            merged_config = shared_config.copy()
            for key, value in policy_config.items():
                if (
                    key in merged_config
                    and isinstance(merged_config[key], dict)
                    and isinstance(value, dict)
                ):
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value

            return merged_config
        else:
            logging.warning(
                f"Policy type '{policy_type}' not found in training config, returning shared config"
            )
            return self.training.get("shared", {})

    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self.inference

    def get_dataset_output_path(self) -> str:
        """Get the full path to the converted dataset."""
        return os.path.join(self.dataset_config.output_root, self.dataset_config.name)

    def get_dataset_meta_path(self) -> str:
        """Get the path to the dataset metadata directory."""
        return os.path.join(self.get_dataset_output_path(), "meta")

    def validate_config(self) -> List[str]:
        """
        Validate the configuration and return list of warnings/errors.

        Returns:
            List of validation messages
        """
        warnings = []

        # Check if source HDF5 exists
        if not os.path.exists(self.dataset_config.source_hdf5):
            warnings.append(
                f"Source HDF5 file not found: {self.dataset_config.source_hdf5}"
            )

        # Check observation mapping consistency
        policy_keys = set()
        for mapping in self.observation_mapping.values():
            policy_keys.add(mapping.policy_key)

        inference_keys = set(self.inference_mapping.keys())

        # Check if all policy keys have inference mapping
        for policy_key in policy_keys:
            if policy_key not in inference_keys:
                warnings.append(f"Policy key '{policy_key}' has no inference mapping")

        # Check if all inference keys have observation mapping
        for inference_key in inference_keys:
            if inference_key not in policy_keys:
                warnings.append(
                    f"Inference key '{inference_key}' has no observation mapping"
                )

        return warnings


def load_task_config(config_path: str) -> TaskConfig:
    """
    Load task configuration from file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Loaded TaskConfig instance
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading task configuration from: {config_path}")
    return TaskConfig.load_from_file(config_path)


def find_config_file(config_name: str, search_dirs: Optional[List[str]] = None) -> str:
    """
    Find configuration file by name in search directories.

    Args:
        config_name: Name of the config file (with or without .yaml extension)
        search_dirs: List of directories to search in. If None, uses default locations.

    Returns:
        Path to the found configuration file

    Raises:
        FileNotFoundError: If configuration file is not found
    """
    if not config_name.endswith(".yaml") and not config_name.endswith(".yml"):
        config_name += ".yaml"

    if search_dirs is None:
        # Default search directories
        search_dirs = [
            "configs",
            "configs/lerobot",
            "/home/ps/Projects/isaac-lab-workspace/IsaacLabLatest/IsaacLab/configs",
            os.getcwd(),
        ]

    for search_dir in search_dirs:
        config_path = os.path.join(search_dir, config_name)
        if os.path.exists(config_path):
            return config_path

    raise FileNotFoundError(
        f"Configuration file '{config_name}' not found in directories: {search_dirs}"
    )


# Convenience function for loading default config
def load_default_task_config() -> TaskConfig:
    """Load the default task configuration."""
    config_path = find_config_file("lerobot_task_config")
    return load_task_config(config_path)
