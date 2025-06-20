# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with LeRobot policy agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import logging

from isaaclab.app import AppLauncher

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

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from lerobot.common.envs.utils import preprocess_observation  # noqa: E402

# LeRobot imports
from lerobot.common.policies.factory import make_policy  # noqa: E402
from lerobot.common.policies.utils import get_device_from_parameters  # noqa: E402
from lerobot.common.utils.random_utils import set_seed  # noqa: E402
from lerobot.common.utils.utils import get_safe_torch_device, init_logging  # noqa: E402


def isaaclab_obs_to_lerobot_format(obs_dict, device):
    """
    Convert Isaac Lab observation format to LeRobot expected format.

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


def load_lerobot_policy(policy_path, policy_device="cuda", use_amp=False):
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
        from types import SimpleNamespace

        policy_cfg = SimpleNamespace()
        policy_cfg.path = policy_path
        policy_cfg.device = policy_device
        policy_cfg.use_amp = use_amp

        # Create a minimal env config (might need adjustment based on your specific setup)
        env_cfg = SimpleNamespace()
        env_cfg.type = "isaaclab"  # This might need to be adjusted

        # Load the policy
        logging.info(f"Loading LeRobot policy from: {policy_path}")
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()

        logging.info("LeRobot policy loaded successfully!")
        return policy

    except Exception as e:
        logging.error(f"Failed to load LeRobot policy: {e}")
        raise


def single_step_inference(policy, observation, device, use_amp=False):
    """
    Perform single step inference with LeRobot policy.

    Args:
        policy: LeRobot policy instance
        observation: Isaac Lab observation
        device: Computing device
        use_amp: Whether to use automatic mixed precision

    Returns:
        Action tensor ready for Isaac Lab environment
    """
    try:
        # Convert Isaac Lab observation to LeRobot format
        processed_obs = isaaclab_obs_to_lerobot_format(observation, device)

        # Apply LeRobot preprocessing (if needed)
        # Note: preprocess_observation might need adaptation for Isaac Lab
        try:
            processed_obs = preprocess_observation(processed_obs)
        except Exception as e:
            logging.debug(
                f"LeRobot preprocess_observation failed, using raw observation: {e}"
            )

        # Perform inference
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type=device.type):
                    action = policy.select_action(processed_obs)
            else:
                action = policy.select_action(processed_obs)

        # Ensure action is on the correct device for Isaac Lab
        action = action.to(device)

        # Remove batch dimension if present and Isaac Lab expects flat actions
        if action.dim() > 1 and action.shape[0] == 1:
            action = action.squeeze(0)

        return action

    except Exception as e:
        logging.error(f"Policy inference failed: {e}")
        # Return zero actions as fallback
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

    # Load LeRobot policy
    try:
        policy = load_lerobot_policy(
            args_cli.policy_path, args_cli.policy_device, args_cli.use_amp
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
                action = single_step_inference(
                    policy, obs, policy_device, args_cli.use_amp
                )

                # Apply actions to environment
                obs, reward, terminated, truncated, info = env.step(action)

                step_count += 1

                # Reset if episode ends or max steps reached
                if (
                    step_count >= args_cli.max_episode_steps
                    or terminated.any()
                    or truncated.any()
                ):
                    logging.info(
                        f"Episode ended at step {step_count}. Resetting environment..."
                    )
                    obs, info = env.reset()
                    policy.reset()  # Reset policy state
                    step_count = 0

        except Exception as e:
            logging.error(f"Error during simulation step: {e}")
            # Try to continue with zero actions
            action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)

    # Close the environment and simulator
    env.close()
    logging.info("Simulation ended.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
