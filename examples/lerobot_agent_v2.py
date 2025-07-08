# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with LeRobot policy agent (v2 with auto-generated EnvConfig)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import logging
import os

# init logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# For IsaacSim, we should import other packages after launching simulation
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="LeRobot agent for Isaac Lab environments (v2 with auto EnvConfig)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--env_config_file",
    type=str,
    default=None,
    help="Env Config YAML file Path, use to update default env config",
)
# LeRobot specific arguments
parser.add_argument(
    "--policy_path",
    type=str,
    required=True,
    help="Path to LeRobot policy (Hub ID or local path). Example: 'lerobot/diffusion_pusht' or 'outputs/train/model/checkpoints/005000/pretrained_model'",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to LeRobot dataset (Hub ID or local path). Example: 'lerobot/diffusion_pusht_dataset' or 'outputs/train/dataset'",
)
parser.add_argument(
    "--config",
    type=str,
    default="lerobot_task_config.yaml",
    help="Path to unified task configuration file. This replaces separate dataset_meta_dir argument.",
)
parser.add_argument(
    "--dataset_meta_dir",
    type=str,
    default=None,
    help="Path to dataset meta directory (legacy, use --config instead). Example: 'assets/converted_dataset/pressed_ori_20250708_rgb/meta'",
)
parser.add_argument(
    "--policy_device",
    type=str,
    default="cuda",
    help="Device to run the policy on (cuda, cpu, auto).",
)
parser.add_argument(
    "--use_amp",
    action="store_true",
    default=False,
    help="Use automatic mixed precision for policy inference.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=1000,
    help="Maximum steps per episode.",
)
parser.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    help="Logging level.",
)
parser.add_argument(
    "--save_envconfig",
    type=str,
    default=None,
    help="Save the auto-generated EnvConfig as Python code to the specified file.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Environment Configuration imports
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
import yaml
from isaaclab_tasks.utils import parse_env_cfg

# LeRobot imports
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.utils import get_device_from_parameters

# Add the lerobot utils to Python path
from lerobot.utils.envconfig_generator import (
    FeatureType,
    generate_envconfig_from_meta,
    save_envconfig_code,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


class DynamicEnvConfig:
    """
    A wrapper class to make the GeneratedEnvConfig compatible with LeRobot's EnvConfig interface.
    """

    def __init__(self, generated_config):
        self.task = generated_config.task
        self.fps = generated_config.fps
        self.episode_length = generated_config.episode_length
        self.obs_type = generated_config.obs_type
        self.render_mode = generated_config.render_mode
        self.features = generated_config.features
        self.features_map = generated_config.features_map

    @property
    def gym_kwargs(self):
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


def dynamic_preprocess_observation(
    observations: dict[str, np.ndarray], env_cfg, device: torch.device, task_config=None
) -> dict[str, torch.Tensor]:
    """
    Convert environment observation to LeRobot format observation based on EnvConfig.
    Now uses unified task configuration for observation key mapping.

    Args:
        observations: Dictionary of observation batches from a Gym environment.
        env_cfg: Environment configuration containing features and feature mapping.
        device: Target device for tensors
        task_config: Optional TaskConfig for observation key mapping

    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    import einops

    # Get available Isaac observation keys
    available_obs_keys = list(observations.keys())
    logging.debug(f"Available Isaac observation keys: {available_obs_keys}")

    # Initialize policy features dictionary
    policy_features = {}

    # Use unified task configuration for observation mapping if available
    if task_config is not None:
        # Use task config for mapping
        for policy_feature_name, _ in env_cfg.features.items():
            if policy_feature_name == "action":  # Skip action feature
                continue

            # Find Isaac observation key using task config
            isaac_key = task_config.find_isaac_observation_key(
                policy_feature_name, available_obs_keys
            )
            if isaac_key:
                policy_features[policy_feature_name] = [isaac_key]
                logging.debug(
                    f"Config-based mapping: {policy_feature_name} -> {isaac_key}"
                )
            else:
                logging.warning(
                    f"No Isaac observation key found for policy feature: {policy_feature_name}"
                )
    else:
        # Fallback to hardcoded mapping for backward compatibility
        for policy_feature_name, _ in env_cfg.features.items():
            if policy_feature_name == "action":  # Skip action feature
                continue

            # Find matching Isaac observation keys using hardcoded patterns
            matched_isaac_keys = []

            if policy_feature_name.startswith("observation.images"):
                # Handle image features
                if policy_feature_name == "observation.images.top":
                    for obs_key in available_obs_keys:
                        if any(
                            pattern in obs_key.lower()
                            for pattern in ["top", "camera", "rgb"]
                        ):
                            matched_isaac_keys.append(obs_key)
                            break
                elif policy_feature_name == "observation.images.side":
                    for obs_key in available_obs_keys:
                        if "side" in obs_key.lower():
                            matched_isaac_keys.append(obs_key)
                            break
                elif policy_feature_name == "observation.images.wrist":
                    for obs_key in available_obs_keys:
                        if "wrist" in obs_key.lower():
                            matched_isaac_keys.append(obs_key)
                            break
            elif policy_feature_name == "observation.state":
                # Handle state features - collect all state-related keys
                for obs_key in available_obs_keys:
                    if any(
                        pattern in obs_key.lower()
                        for pattern in ["joint", "pos", "state", "robot"]
                    ):
                        matched_isaac_keys.append(obs_key)

            # If no matches found, try direct mapping
            if not matched_isaac_keys and policy_feature_name in available_obs_keys:
                matched_isaac_keys.append(policy_feature_name)

            if matched_isaac_keys:
                policy_features[policy_feature_name] = matched_isaac_keys
                logging.debug(
                    f"Hardcoded mapping: {policy_feature_name} -> {matched_isaac_keys}"
                )
            else:
                logging.warning(
                    f"No Isaac observation keys found for policy feature: {policy_feature_name}"
                )

    return_observations = {}

    # 处理每个策略特征
    for policy_feature_name, isaac_keys in policy_features.items():
        tensors_to_concat = []

        for isaac_key in isaac_keys:
            if isaac_key not in observations:
                logging.warning(
                    f"Isaac observation key '{isaac_key}' not found in observations"
                )
                continue

            value = observations[isaac_key]
            feature = env_cfg.features.get(policy_feature_name)

            if feature is None:
                logging.warning(
                    f"Feature {policy_feature_name} not found in env_cfg.features"
                )
                continue

            # Convert to tensor
            if isinstance(value, np.ndarray):
                tensor_value = torch.from_numpy(value)
            elif torch.is_tensor(value):
                tensor_value = value.clone()
            else:
                try:
                    tensor_value = torch.tensor(value, device=device)
                except Exception as e:
                    logging.warning(
                        f"Could not convert observation key '{isaac_key}' to tensor: {e}"
                    )
                    continue

            # Handle different feature types
            if feature.type == FeatureType.VISUAL or feature.type.name == "VISUAL":
                # Handle image observations
                if tensor_value.ndim == 3:
                    tensor_value = tensor_value.unsqueeze(0)  # Add batch dimension

                # Sanity check that images are channel last
                _, h, w, c = tensor_value.shape
                assert c < h and c < w, (
                    f"expect channel last images, but instead got {tensor_value.shape=}"
                )

                # Sanity check that images are uint8
                if tensor_value.dtype == torch.uint8:
                    # Convert to channel first of type float32 in range [0,1]
                    tensor_value = einops.rearrange(
                        tensor_value, "b h w c -> b c h w"
                    ).contiguous()
                    tensor_value = tensor_value.type(torch.float32)
                    tensor_value /= 255
                elif tensor_value.dtype == torch.float32:
                    # Already float, just rearrange if needed
                    if (
                        tensor_value.shape[-1] < tensor_value.shape[-2]
                    ):  # likely channel last
                        tensor_value = einops.rearrange(
                            tensor_value, "b h w c -> b c h w"
                        ).contiguous()

            elif (
                feature.type == FeatureType.STATE
                or feature.type.name == "STATE"
                or feature.type == FeatureType.ENV
                or feature.type.name == "ENV"
                or feature.type == FeatureType.ACTION
                or feature.type.name == "ACTION"
            ):
                # Handle state/env/action observations
                tensor_value = tensor_value.float()
                if tensor_value.dim() == 1:
                    tensor_value = tensor_value.unsqueeze(0)  # Add batch dimension
                # 对于多批次数据，保持batch维度，只拍平特征维度
                elif tensor_value.dim() > 2:
                    # 将除了batch维度外的所有维度拍平
                    batch_size = tensor_value.shape[0]
                    tensor_value = tensor_value.view(batch_size, -1)

            # Move to device
            tensor_value = tensor_value.to(device, non_blocking=device.type == "cuda")
            tensors_to_concat.append(tensor_value)

        # 如果有多个tensor需要拼接
        if len(tensors_to_concat) > 1:
            # 确保所有tensor有相同的batch size
            batch_sizes = [t.shape[0] for t in tensors_to_concat]
            if len(set(batch_sizes)) > 1:
                logging.warning(
                    f"Inconsistent batch sizes for policy feature '{policy_feature_name}': {batch_sizes}"
                )
                # 使用最小的batch size
                min_batch_size = min(batch_sizes)
                tensors_to_concat = [t[:min_batch_size] for t in tensors_to_concat]

            # 在最后一个维度拼接
            concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
            return_observations[policy_feature_name] = concatenated_tensor
        elif len(tensors_to_concat) == 1:
            return_observations[policy_feature_name] = tensors_to_concat[0]
        else:
            logging.warning(
                f"No valid tensors found for policy feature '{policy_feature_name}'"
            )

    return return_observations


def load_lerobot_policy(policy_path, dataset_meta, policy_device="cuda", use_amp=False):
    """
    Load LeRobot policy from path.

    Args:
        policy_path: Path to policy (Hub ID or local path)
        dataset_meta: Dataset metadata
        policy_device: Device to load policy on
        use_amp: Whether to use automatic mixed precision

    Returns:
        Loaded policy instance
    """
    try:
        # Check if policy_path contains pretrained_model subdirectory
        pretrained_model_path = os.path.join(policy_path, "pretrained_model")
        if os.path.exists(pretrained_model_path):
            logging.info(
                f"Found pretrained_model directory, using: {pretrained_model_path}"
            )
            policy_path = pretrained_model_path

        # Check if config.json exists
        config_path = os.path.join(policy_path, "config.json")
        if not os.path.exists(config_path):
            logging.error(f"config.json not found in {policy_path}")
            raise FileNotFoundError(f"config.json not found in {policy_path}")

        # Create a minimal policy config
        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)

        if policy_device:
            logging.warning(
                f"over write policy device (from checkpoint) with {policy_device}"
            )
            policy_cfg.device = policy_device

        # Log the configuration
        logging.info(f"Policy config type: {type(policy_cfg)}")
        logging.info(f"Policy config: {policy_cfg}")

        # Load the policy
        logging.info(f"Loading LeRobot policy from: {policy_path}")
        policy = make_policy(cfg=policy_cfg, ds_meta=dataset_meta, env_cfg=None)

        # Move policy to the specified device
        if hasattr(policy, "to"):
            policy = policy.to(policy_device)

        policy.eval()

        logging.info("LeRobot policy loaded successfully!")
        return policy

    except Exception as e:
        logging.error(f"Failed to load LeRobot policy: {e}")
        raise


def single_step_inference(
    policy, observation, device, env_cfg=None, use_amp=False, task_config=None
):
    """
    Perform single step inference with LeRobot policy.

    Args:
        policy: LeRobot policy instance
        observation: Isaac Lab observation
        device: Computing device
        env_cfg: Environment configuration for dynamic preprocessing
        use_amp: Whether to use automatic mixed precision
        task_config: Optional TaskConfig for observation key mapping

    Returns:
        Action tensor ready for Isaac Lab environment
    """
    try:
        # Convert Isaac Lab observation to LeRobot format
        if env_cfg is not None:
            # Use dynamic preprocessing with task config
            processed_obs = dynamic_preprocess_observation(
                observation, env_cfg, device, task_config
            )
        else:
            # Fallback to legacy function
            raise ValueError(
                "env_cfg is None, please provide a valid EnvConfig for dynamic preprocessing."
            )

        # Perform inference
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type=device.type):
                    action = policy.select_action(processed_obs)
            else:
                action = policy.select_action(processed_obs)

        logging.info(f"Action from policy: {action}")

        # Ensure action is on the correct device for Isaac Lab
        action = action.to(device)

        # Fix action shape for Isaac Lab environment
        # Isaac Lab expects action shape [batch_size, action_dim] but we might get [action_dim] or [1, action_dim]
        if action.dim() == 1:
            # [action_dim] -> [1, action_dim]
            action = action.unsqueeze(0)
        elif action.dim() > 2:
            # [batch, 1, action_dim] or similar -> [batch, action_dim]
            action = action.view(action.shape[0], -1)

        logging.debug(f"Action shape after processing: {action.shape}")

        return action

    except Exception as e:
        logging.error(f"Policy inference failed: {type(e).__name__} {e}")
        # Return zero actions as fallback
        import traceback

        logging.error(traceback.format_exc())
        logging.warning("Falling back to zero actions")
        return torch.zeros(
            policy.action_dim if hasattr(policy, "action_dim") else 7, device=device
        )


def main():
    """LeRobot policy agent with Isaac Lab environment (v2 with unified config)."""

    # Setup logging
    logging.basicConfig(level=getattr(logging, args_cli.log_level.upper()))
    init_logging()

    # Set random seed
    set_seed(args_cli.seed)

    # Check and setup device
    device = get_safe_torch_device(args_cli.policy_device, log=True)
    logging.info(f"Using device: {device}")

    # Load unified task configuration if provided
    task_config = None
    if args_cli.config:
        try:
            from lerobot.utils.config_loader import find_config_file, load_task_config

            if not os.path.exists(args_cli.config):
                config_path = find_config_file(args_cli.config)
            else:
                config_path = args_cli.config

            task_config = load_task_config(config_path)
            logging.info(f"Loaded unified task configuration from: {config_path}")
            logging.info(f"Dataset: {task_config.dataset_config.name}")
            logging.info(f"Task: {task_config.dataset_config.task}")
        except Exception as e:
            logging.warning(f"Failed to load unified task config: {e}")
            logging.warning("Falling back to legacy metadata-based configuration")

    # Generate/load EnvConfig
    if task_config is not None:
        # TODO: Generate EnvConfig from task config instead of metadata
        # For now, still use metadata but deprecate the dataset_meta_dir argument
        if args_cli.dataset_meta_dir:
            logging.warning(
                "Using --dataset_meta_dir is deprecated. The unified config will be used in the future."
            )

        # Use the output directory from task config as meta directory
        meta_dir = os.path.join(
            task_config.dataset_config.output_root,
            task_config.dataset_config.name,
            "meta",
        )
        if not os.path.exists(meta_dir):
            logging.error(f"Meta directory not found: {meta_dir}")
            logging.error(
                "Please run dataset conversion first or use --dataset_meta_dir for legacy support"
            )
            simulation_app.close()
            return

        logging.info(f"Using meta directory from task config: {meta_dir}")
        generated_config = generate_envconfig_from_meta(
            meta_dir, config_name="unified_config"
        )
    else:
        # Legacy: Generate EnvConfig from dataset metadata
        if not args_cli.dataset_meta_dir:
            logging.error("Either --config or --dataset_meta_dir must be provided")
            simulation_app.close()
            return

        logging.info(
            f"Generating EnvConfig from meta directory: {args_cli.dataset_meta_dir}"
        )
        generated_config = generate_envconfig_from_meta(
            args_cli.dataset_meta_dir, config_name="auto_generated_v2"
        )

    # Wrap generated config for compatibility
    isaac_env_cfg = DynamicEnvConfig(generated_config)

    logging.info("EnvConfig generated successfully!")
    logging.info(f"Generated features: {list(isaac_env_cfg.features.keys())}")
    logging.info(f"Features map: {isaac_env_cfg.features_map}")

    # Save EnvConfig code if requested
    if args_cli.save_envconfig:
        save_envconfig_code(generated_config, args_cli.save_envconfig)
        logging.info(f"Saved EnvConfig code to: {args_cli.save_envconfig}")

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    if args_cli.env_config_file:
        with open(args_cli.env_config_file, "r") as f:
            env_new_cfg = yaml.safe_load(f)

        def dynamic_set_attr(object: object, kwargs: dict, path: list[str]):
            for k, v in kwargs.items():
                if k in object.__dict__:
                    if isinstance(v, dict):
                        next_path = path.copy()
                        next_path.append(k)
                        dynamic_set_attr(object.__getattribute__(k), v, next_path)
                    else:
                        print(
                            f"set {'.'.join(path + [k])} from {object.__getattribute__(k)} to {v}"
                        )
                        object.__setattr__(k, v)

        dynamic_set_attr(env_cfg, env_new_cfg, path=["env_cfg"])

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Print environment info
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Update isaac_env_cfg with actual environment information
    try:
        # Get a sample observation to determine dimensions
        sample_obs, _ = env.reset()
        if "policy" in sample_obs:
            sample_obs = sample_obs["policy"]

        logging.info(f"Available Isaac observation keys: {list(sample_obs.keys())}")

        # 打印每个观测的形状以便调试
        for key, value in sample_obs.items():
            if hasattr(value, "shape"):
                logging.info(f"  {key}: {value.shape} {type(value)}")

        # Update feature shapes based on actual observations
        for env_key, value in sample_obs.items():
            if env_key in isaac_env_cfg.features and hasattr(value, "shape"):
                if len(value.shape) > 1:
                    # Multi-dimensional tensor, use the last dimension as feature dim
                    feature_dim = value.shape[-1]
                else:
                    # 1D tensor
                    feature_dim = value.shape[0] if len(value.shape) > 0 else 1

                # Update the feature shape
                current_feature = isaac_env_cfg.features[env_key]
                from lerobot.utils.envconfig_generator import FeatureType, PolicyFeature

                isaac_env_cfg.features[env_key] = PolicyFeature(
                    type=current_feature.type, shape=(feature_dim,)
                )

        # Update action dimension from environment
        if hasattr(env.action_space, "shape") and env.action_space.shape:
            action_dim = env.action_space.shape[-1]  # Use the last dimension
            from lerobot.utils.envconfig_generator import FeatureType, PolicyFeature

            isaac_env_cfg.features["action"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(action_dim,)
            )
            logging.info(f"Updated action dimension to: {action_dim}")

        logging.info("Updated Isaac config based on actual observations")

    except Exception as e:
        logging.warning(
            f"Could not infer dimensions from environment: {e}, using auto-generated config"
        )

    # Load LeRobot policy
    try:
        # Check if dataset path exists and is valid
        if not os.path.exists(args_cli.dataset_path):
            logging.warning(f"Dataset path does not exist: {args_cli.dataset_path}")
            # Try to create a minimal dataset metadata
            dataset_meta = LeRobotDatasetMetadata(repo_id="", root="")
        else:
            dataset_meta = LeRobotDatasetMetadata(
                repo_id="", root=args_cli.dataset_path
            )

        policy = load_lerobot_policy(
            args_cli.policy_path, dataset_meta, args_cli.policy_device, args_cli.use_amp
        )
        policy_device = get_device_from_parameters(policy)
        logging.info(f"Policy device: {policy_device}")

        # Reset policy
        if hasattr(policy, "reset"):
            policy.reset()

    except Exception as e:
        logging.error(f"Failed to load policy, exiting: {e}")
        import traceback

        traceback.print_exc()
        env.close()
        simulation_app.close()
        return

    # Reset environment
    obs, info = env.reset()
    step_count = 0

    logging.info("Starting policy rollout...")

    # Main simulation loop
    while simulation_app.is_running():
        try:
            # Run policy inference
            with torch.inference_mode():
                obs = obs["policy"]
                action = single_step_inference(
                    policy,
                    obs,
                    policy_device,
                    isaac_env_cfg,
                    args_cli.use_amp,
                    task_config,
                )

                # Apply actions to environment
                obs, reward, terminated, truncated, info = env.step(action)

                step_count += 1

                # Reset if episode ends or max steps reached
                should_reset = False
                if step_count >= args_cli.max_episode_steps:
                    should_reset = True
                elif torch.is_tensor(terminated):
                    should_reset = terminated.any().item()
                elif isinstance(terminated, (bool, np.bool_)):
                    should_reset = bool(terminated)
                elif torch.is_tensor(truncated):
                    should_reset = should_reset or truncated.any().item()
                elif isinstance(truncated, (bool, np.bool_)):
                    should_reset = should_reset or bool(truncated)

                if should_reset:
                    logging.info(
                        f"Episode ended at step {step_count}. Resetting environment..."
                    )
                    obs, info = env.reset()
                    if hasattr(policy, "reset"):
                        policy.reset()  # Reset policy state
                    step_count = 0

        except Exception as e:
            logging.error(f"Error during simulation step: {type(e).__name__}: {e}")
            import traceback

            logging.error(traceback.format_exc())
            # Try to continue with zero actions
            if hasattr(env.action_space, "shape") and env.action_space.shape:
                action_shape = env.action_space.shape
                device = (
                    policy_device
                    if "policy_device" in locals()
                    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                # Create zero action with correct shape [batch_size, action_dim]
                if len(action_shape) == 2:  # [batch_size, action_dim]
                    action = torch.zeros(action_shape, device=device)
                else:  # [action_dim] -> [1, action_dim]
                    action = torch.zeros((1, action_shape[0]), device=device)
            else:
                # Fallback action dimension
                action_shape = isaac_env_cfg.features["action"].shape
                action = torch.zeros((1, action_shape[0]), device=policy_device)
            obs, reward, terminated, truncated, info = env.step(action)

    # Close the environment and simulator
    env.close()
    logging.info("Simulation ended.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
