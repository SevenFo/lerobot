# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with LeRobot policy agent."""

"""Launch Isaac Sim Simulator first."""

import argparse  # noqa: E402
import logging  # noqa: E402

# init logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# For IsaacSim, we should import other packages after launching simulation
from isaaclab.app import AppLauncher  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(
    description="LeRobot agent for Isaac Lab environments."
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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Isaac Environment Configuration imports
from dataclasses import dataclass, field  # noqa: E402

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.envs.configs import EnvConfig  # noqa: E402

# LeRobot imports
from lerobot.common.policies.factory import make_policy  # noqa: E402
from lerobot.common.policies.utils import get_device_from_parameters  # noqa: E402
from lerobot.common.utils.random_utils import set_seed  # noqa: E402
from lerobot.common.utils.utils import get_safe_torch_device, init_logging  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402


# Isaac Environment Configuration
@EnvConfig.register_subclass("isaac")
@dataclass
class IsaacEnv(EnvConfig):
    """Configuration for Isaac Lab environments."""

    task: str = "Isaac-Manipulation-v0"
    fps: int = 100
    episode_length: int = 1000
    obs_type: str = "state_environment_state"
    render_mode: str = "rgb_array"

    # features的key是env observation keys
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            # 机器人状态相关的env observation keys
            "joint_pos": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "joint_vel": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
            "eef_pos": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "eef_quat": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            "gripper_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            # 环境状态相关的env observation keys
            "box_positions": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
            "box_orientations": PolicyFeature(type=FeatureType.ENV, shape=(4,)),
            "spanner_positions": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
            "spanner_orientations": PolicyFeature(type=FeatureType.ENV, shape=(4,)),
            "object": PolicyFeature(type=FeatureType.ENV, shape=(26,)),
            # 动作和任务
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )

    # features_map: env observation keys -> policy feature keys
    # 支持多个env observation拼接成一个policy feature
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            # 将多个机器人状态拼接成observation.state
            "joint_pos": "observation.state",
            "joint_vel": "observation.state",
            "eef_pos": "observation.state",
            "eef_quat": "observation.state",
            "gripper_pos": "observation.state",
            # 将多个环境状态拼接成observation.environment_state
            "box_positions": "observation.environment_state",
            "box_orientations": "observation.environment_state",
            "spanner_positions": "observation.environment_state",
            "spanner_orientations": "observation.environment_state",
            # object已经是完整的环境状态，也可以映射到environment_state或单独映射
            "object": "observation.environment_state",
            # 动作和任务
            "action": "action",
            "task": "task",
        }
    )

    def __post_init__(self):
        # 如果需要图像观测，可以添加像素特征
        if "pixels" in self.obs_type:
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 224, 224)
            )
            self.features_map["pixels"] = "observation.image"

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


def dynamic_preprocess_observation(
    observations: dict[str, np.ndarray], env_cfg: EnvConfig, device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Convert environment observation to LeRobot format observation based on EnvConfig.
    支持多个env observation keys拼接成一个policy feature。

    Args:
        observations: Dictionary of observation batches from a Gym environment.
        env_cfg: Environment configuration containing features and feature mapping.
        device: Target device for tensors

    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    import einops

    # 首先按policy feature key分组env observation keys
    policy_groups = {}
    for env_key, policy_key in env_cfg.features_map.items():
        if env_key == "task":  # task特殊处理
            continue
        if policy_key not in policy_groups:
            policy_groups[policy_key] = []
        policy_groups[policy_key].append(env_key)

    return_observations = {}

    # 处理每个policy feature
    for policy_key, env_keys in policy_groups.items():
        tensors_to_concat = []

        for env_key in env_keys:
            if env_key not in observations:
                logging.warning(
                    f"Env observation key '{env_key}' not found in observations"
                )
                continue

            value = observations[env_key]
            feature = env_cfg.features.get(env_key)

            if feature is None:
                logging.warning(f"Feature {env_key} not found in env_cfg.features")
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
                        f"Could not convert observation key '{env_key}' to tensor: {e}"
                    )
                    continue

            # Handle different feature types
            if feature.type == FeatureType.VISUAL:
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

            elif feature.type in [
                FeatureType.STATE,
                FeatureType.ENV,
                FeatureType.ACTION,
            ]:
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
                    f"Inconsistent batch sizes for policy key '{policy_key}': {batch_sizes}"
                )
                # 使用最小的batch size
                min_batch_size = min(batch_sizes)
                tensors_to_concat = [t[:min_batch_size] for t in tensors_to_concat]

            # 在最后一个维度拼接
            concatenated_tensor = torch.cat(tensors_to_concat, dim=-1)
            return_observations[policy_key] = concatenated_tensor
        elif len(tensors_to_concat) == 1:
            return_observations[policy_key] = tensors_to_concat[0]
        else:
            logging.warning(f"No valid tensors found for policy key '{policy_key}'")

    return return_observations


def isaaclab_obs_to_lerobot_format(obs_dict, device):
    """
    Convert Isaac Lab observation format to LeRobot expected format.
    This is a legacy function - prefer using dynamic_preprocess_observation with EnvConfig.

    Args:
        obs_dict: Isaac Lab observation dictionary
        device: Target device for tensors

    Returns:
        Processed observation dictionary compatible with LeRobot
    """
    processed_obs = {}

    for key, value in obs_dict.items():
        if torch.is_tensor(value):
            # Ensure tensor is on correct device and has batch dimension
            if value.dim() == 1:
                value = value.unsqueeze(0)  # Add batch dimension if missing
            processed_obs[key] = value.to(device, non_blocking=device.type == "cuda")
        elif isinstance(value, np.ndarray):
            # Convert numpy to tensor and add batch dimension if needed
            tensor_value = torch.from_numpy(value)
            if tensor_value.dim() == 1:
                tensor_value = tensor_value.unsqueeze(0)
            processed_obs[key] = tensor_value.to(
                device, non_blocking=device.type == "cuda"
            )
        else:
            # Handle other data types
            try:
                tensor_value = torch.tensor(value, device=device)
                if tensor_value.dim() == 0:
                    tensor_value = tensor_value.unsqueeze(
                        0
                    )  # Add batch dimension for scalars
                processed_obs[key] = tensor_value
            except Exception as e:
                logging.warning(
                    f"Could not convert observation key '{key}' to tensor: {e}"
                )
                continue

    return processed_obs


def load_lerobot_policy(policy_path, dataset_meta, policy_device="cuda", use_amp=False):
    """
    Load LeRobot policy from path.

    Args:
        policy_path: Path to policy (Hub ID or local path)
        policy_device: Device to load policy on
        use_amp: Whether to use automatic mixed precision

    Returns:
        Loaded policy instance
    """
    try:
        # Create a minimal policy config
        # Note: In practice, you might want to load this from a config file

        policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cfg.pretrained_path = policy_path
        policy_cfg.device = policy_device
        policy_cfg.use_amp = use_amp

        # Create a minimal env config (might need adjustment based on your specific setup)
        # env_cfg = SimpleNamespace()
        # env_cfg.type = "isaaclab"  # This might need to be adjusted

        # Load the policy
        logging.info(f"Loading LeRobot policy from: {policy_path}")
        policy = make_policy(cfg=policy_cfg, ds_meta=dataset_meta, env_cfg=None)
        policy.eval()

        logging.info("LeRobot policy loaded successfully!")
        return policy

    except Exception as e:
        logging.error(f"Failed to load LeRobot policy: {e}")
        raise


def single_step_inference(policy, observation, device, env_cfg=None, use_amp=False):
    """
    Perform single step inference with LeRobot policy.

    Args:
        policy: LeRobot policy instance
        observation: Isaac Lab observation
        device: Computing device
        env_cfg: Environment configuration for dynamic preprocessing
        use_amp: Whether to use automatic mixed precision

    Returns:
        Action tensor ready for Isaac Lab environment
    """
    try:
        # Convert Isaac Lab observation to LeRobot format
        if env_cfg is not None:
            # Use dynamic preprocessing based on env config
            processed_obs = dynamic_preprocess_observation(observation, env_cfg, device)
        else:
            # Fallback to legacy function
            processed_obs = isaaclab_obs_to_lerobot_format(observation, device)

        # Apply LeRobot preprocessing (if needed)
        # Note: For Isaac environments, we may need additional preprocessing
        # try:
        #     # Check if we need to use the original preprocess_observation function
        #     # Only use it if our observation format matches what it expects
        #     if "pixels" in observation or "agent_pos" in observation:
        #         from lerobot.common.envs.utils import (
        #             preprocess_observation as lerobot_preprocess,  # noqa: E402
        #         )

        #         additional_processed = lerobot_preprocess(observation)
        #         # Merge with our processed observations, giving priority to our processing
        #         for key, value in additional_processed.items():
        #             if key not in processed_obs:
        #                 processed_obs[key] = value.to(device)
        # except Exception as e:
        #     logging.debug(
        #         f"LeRobot preprocess_observation failed, using dynamic preprocessing: {type(e).__name__} {e}"
        #     )
        #     import traceback

        #     logging.debug(traceback.format_exc())

        # Perform inference
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type=device.type):
                    action = policy.select_action(processed_obs)
            else:
                action = policy.select_action(processed_obs)

        logging.debug(f"Action from policy: {action}")

        # Ensure action is on the correct device for Isaac Lab
        action = action.to(device)

        # Remove batch dimension if present and Isaac Lab expects flat actions
        if action.dim() > 1 and action.shape[0] == 1:
            action = action.squeeze(0)

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
    """LeRobot policy agent with Isaac Lab environment."""

    # Setup logging
    logging.basicConfig(level=getattr(logging, args_cli.log_level.upper()))
    init_logging()

    # Set random seed
    set_seed(args_cli.seed)

    # Check and setup device
    device = get_safe_torch_device(args_cli.policy_device, log=True)
    logging.info(f"Using device: {device}")

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

    # Create Isaac environment config for dynamic preprocessing
    isaac_env_cfg = IsaacEnv()
    # Try to infer dimensions from the actual environment
    try:
        # Get a sample observation to determine dimensions
        sample_obs, _ = env.reset()
        if "policy" in sample_obs:
            sample_obs = sample_obs["policy"]

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
                isaac_env_cfg.features[env_key] = PolicyFeature(
                    type=current_feature.type, shape=(feature_dim,)
                )

        # Update action dimension from environment
        if hasattr(env.action_space, "shape") and env.action_space.shape:
            action_dim = env.action_space.shape[0]
            isaac_env_cfg.features["action"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(action_dim,)
            )

        logging.info("Updated Isaac config based on actual observations")

    except Exception as e:
        logging.warning(
            f"Could not infer dimensions from environment: {e}, using defaults"
        )

    # Load LeRobot policy
    try:
        dataset_meta = LeRobotDatasetMetadata(repo_id="", root=args_cli.dataset_path)
        policy = load_lerobot_policy(
            args_cli.policy_path, dataset_meta, args_cli.policy_device, args_cli.use_amp
        )
        policy_device = get_device_from_parameters(policy)
        logging.info(f"Policy device: {policy_device}")

        # Reset policy
        policy.reset()

    except Exception as e:
        logging.error(f"Failed to load policy, exiting: {e}")
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
                    policy, obs, policy_device, isaac_env_cfg, args_cli.use_amp
                )

                # Apply actions to environment
                obs, reward, terminated, truncated, info = env.step(action)

                step_count += 1

                # Reset if episode ends or max steps reached
                # Handle both tensor and scalar cases for terminated/truncated
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
                action = torch.zeros(action_shape, device=device)
            else:
                # Fallback action dimension
                action_shape = isaac_env_cfg.features["action"].shape
                action = torch.zeros(action_shape, device=policy_device)
            obs, reward, terminated, truncated, info = env.step(action)

    # Close the environment and simulator
    env.close()
    logging.info("Simulation ended.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
