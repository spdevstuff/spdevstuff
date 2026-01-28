
import os
import torch
import numpy as np
import cv2
import math

# 2026 Standardized Imports
from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero.libero import benchmark
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

from lerobot.envs.utils import preprocess_observation, add_envs_task
os.environ["MUJOCO_GL"] = "glfw"

def process_image(img_data, rotate=True):
    """Converts LeRobot pixel data to OpenCV BGR and corrects 180-degree inversion."""
    if hasattr(img_data, "permute"):
        # Image is typically [Batch, C, H, W] -> index 0 -> [C, H, W] -> [H, W, C]
        img_data = img_data[0].permute(1, 2, 0).cpu().numpy()
    elif isinstance(img_data, np.ndarray):
        if img_data.ndim == 4: img_data = img_data[0]
        if img_data.shape[0] == 3: img_data = np.transpose(img_data, (1, 2, 0))

    if img_data.dtype != np.uint8:
        img_data = (img_data * 255).astype(np.uint8)
    
    frame_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    if rotate:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
    return frame_bgr
def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = "HuggingFaceVLA/smolvla_libero"

    print(f"Loading SmolVLA from {repo_id}...")
    policy = SmolVLAPolicy.from_pretrained(repo_id)
    policy.to(device)
    policy.eval()

    # Setup LIBERO-Spatial Environment
    env_cfg = make_env_config("libero", task="libero_spatial", render_mode="rgb_array")
    envs_dict = make_env(env_cfg)
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_instance = benchmark_dict[suite_name]()
    libero_task = suite_instance.get_task(task_id)
    task_instruction = libero_task.language
    print(f"Instruction found: {task_instruction}")
    preprocessor, postprocessor = make_pre_post_processors(policy.config, repo_id)
    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy.config)



    env = envs_dict[suite_name][task_id]

    # Resolve Horizon
    inner_env = env.unwrapped.envs[0]
    max_horizon = getattr(inner_env._env, "horizon", getattr(getattr(inner_env._env, "env", {}), "horizon", 500))
    
    observation, info = env.reset()
    cnt = 0
    
    print(f"SmolVLA evaluating {suite_name}. Press 'q' to stop.")

    try:
        while True:
            observation = preprocess_observation(observation)
            
            observation = add_envs_task(env, observation)
            observation = env_preprocessor(observation)
            observation = preprocessor(observation)

            # 2. Model Inference
            with torch.no_grad():
                action = policy.select_action(observation)
            action = postprocessor(action)
            

            action_transition = {"action": action}
            action_transition = env_postprocessor(action_transition)
            action = action_transition["action"]

            # Convert to CPU / numpy.
            action_numpy: np.ndarray = action.to("cpu").numpy()
            assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

            observation, reward, terminated, truncated, info = env.step(action_numpy)
            cnt += 1

            # --- Multi-View Rendering ---
            raw_render = env.render()
            if raw_render is not None:
                global_frame = np.array(raw_render[0]) if isinstance(raw_render, (list, tuple)) else raw_render
                cv2.imshow("1. Global View", cv2.cvtColor(global_frame, cv2.COLOR_RGB2BGR))
            
            if "pixels" in observation:
                cv2.imshow("2. Agent View", process_image(observation["pixels"]["image"]))
                cv2.imshow("3. Wrist Camera", process_image(observation["pixels"]["image2"]))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 4. Reset Logic
            is_success = False
            if isinstance(info, dict) and "success" in info:
                is_success = any(info["success"])

            if any(terminated) or any(truncated) or cnt >= max_horizon or is_success:
                print(f"Resetting Env. Step: {cnt}/{max_horizon} | Success: {is_success}")
                observation, info = env.reset()
                cnt = 0
                
    finally:
        cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    main()
