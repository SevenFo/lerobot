# LeRobot Agent for Isaac Lab

This script (`lerobot_agent.py`) integrates LeRobot policies with Isaac Lab environments, allowing you to deploy trained LeRobot models in Isaac Lab simulations.

## Prerequisites

1. **Isaac Lab**: Make sure Isaac Lab is properly installed and configured
2. **LeRobot**: Ensure LeRobot is installed in your environment
3. **Trained Policy**: Have a trained LeRobot policy (either from Hub or local checkpoint)

## Usage

### Basic Usage

```bash
# Run with a policy from HuggingFace Hub
python examples/lerobot_agent.py \
    --task Isaac-Cartpole-v0 \
    --policy_path lerobot/diffusion_pusht \
    --num_envs 1

# Run with a local policy checkpoint
python examples/lerobot_agent.py \
    --task Isaac-Reach-Franka-v0 \
    --policy_path outputs/train/diffusion_reach/checkpoints/005000/pretrained_model \
    --num_envs 4 \
    --policy_device cuda
```

### Advanced Usage

```bash
# With custom configuration and AMP
python examples/lerobot_agent.py \
    --task Isaac-Lift-Franka-v0 \
    --policy_path path/to/your/policy \
    --env_config_file config/custom_env.yaml \
    --num_envs 8 \
    --policy_device cuda \
    --use_amp \
    --max_episode_steps 500 \
    --seed 123 \
    --log_level DEBUG
```

## Command Line Arguments

### Required Arguments
- `--task`: Isaac Lab task name (e.g., "Isaac-Reach-Franka-v0")
- `--policy_path`: Path to LeRobot policy (Hub ID or local path)

### Optional Arguments
- `--num_envs`: Number of parallel environments (default: based on task)
- `--policy_device`: Device for policy inference ("cuda", "cpu", "auto")
- `--use_amp`: Enable automatic mixed precision for faster inference
- `--seed`: Random seed for reproducibility (default: 42)
- `--max_episode_steps`: Maximum steps per episode (default: 1000)
- `--env_config_file`: YAML file to override environment configuration
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Isaac Lab Specific Arguments
- `--disable_fabric`: Disable fabric and use USD I/O operations
- Standard Isaac Lab AppLauncher arguments (--headless, --enable_cameras, etc.)

## Key Features

### 1. Automatic Format Conversion
The script automatically converts between Isaac Lab observation format and LeRobot expected format:
- Handles tensor/numpy conversions
- Manages batch dimensions
- Ensures proper device placement

### 2. Robust Error Handling
- Graceful fallback to zero actions if policy inference fails
- Automatic episode reset on completion or maximum steps
- Comprehensive logging for debugging

### 3. Policy State Management
- Proper policy reset between episodes
- Device-aware tensor operations
- Memory-efficient inference mode

## Architecture Overview

```
Isaac Lab Environment → Observation → Format Conversion → LeRobot Policy → Action → Isaac Lab Environment
                   ↑                                                                      ↓
                   └────────────── Episode Reset & State Management ←────────────────────┘
```

## Key Functions

### `isaaclab_obs_to_lerobot_format(obs_dict, device)`
Converts Isaac Lab observations to LeRobot-compatible format.

### `load_lerobot_policy(policy_path, policy_device, use_amp)`
Loads a LeRobot policy from Hub ID or local path.

### `single_step_inference(policy, observation, device, use_amp)`
Performs single-step policy inference with error handling.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure both Isaac Lab and LeRobot are in your Python path
2. **Device Mismatch**: Check that policy_device matches your available hardware
3. **Observation Shape Mismatch**: The script includes automatic shape handling, but complex observations might need custom preprocessing
4. **Policy Loading Fails**: Verify the policy path and ensure all required files (config.json, model.safetensors) are present

### Debug Mode
Run with `--log_level DEBUG` to get detailed information about:
- Observation preprocessing steps
- Policy inference details
- Error traces and fallback actions

## Example Workflows

### 1. Testing a Hub Policy
```bash
# Quick test with a pre-trained policy
python examples/lerobot_agent.py \
    --task Isaac-Cartpole-v0 \
    --policy_path lerobot/diffusion_pusht \
    --max_episode_steps 100 \
    --log_level INFO
```

### 2. Evaluating Your Trained Model
```bash
# Test your own trained model
python examples/lerobot_agent.py \
    --task Isaac-Reach-Franka-v0 \
    --policy_path outputs/train/my_policy/checkpoints/010000/pretrained_model \
    --num_envs 4 \
    --seed 42
```

### 3. Production Deployment
```bash
# High-performance deployment
python examples/lerobot_agent.py \
    --task Isaac-Assembly-Franka-v0 \
    --policy_path path/to/production/policy \
    --num_envs 16 \
    --use_amp \
    --policy_device cuda \
    --headless \
    --max_episode_steps 1000
```

## Notes

- The script follows Isaac Lab's import structure requirements (imports after AppLauncher)
- Automatic mixed precision (AMP) can significantly speed up inference on modern GPUs
- The script is designed to be robust and continue running even if individual policy inferences fail
- Episode resets are handled automatically based on environment termination/truncation or max steps
