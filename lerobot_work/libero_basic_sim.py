
import numpy as np
import cv2
from lerobot.envs.factory import make_env, make_env_config

def process_image(img_data, rotate=True):
    """Converts LeRobot pixel data to OpenCV BGR and corrects 180-degree inversion."""
    #Handle Torch Tensors or NumPy Arrays
    if hasattr(img_data, "permute"):
        # [Batch, C, H, W] -> index 0 -> [C, H, W] -> [H, W, C]
        img_data = img_data[0].permute(1, 2, 0).cpu().numpy()
    elif isinstance(img_data, np.ndarray):
        # [Batch, C, H, W] -> index 0 -> [C, H, W]
        if img_data.ndim == 4:
            img_data = img_data[0]
        # [C, H, W] -> [H, W, C]
        if img_data.shape[0] == 3:
            img_data = np.transpose(img_data, (1, 2, 0))

    #Scale 0.0-1.0 float to 0-255 uint8
    if img_data.dtype != np.uint8:
        img_data = (img_data * 255).astype(np.uint8)
    
    #Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
    
    if rotate:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)
        
    return frame_bgr

def main():

    cfg = make_env_config(
        "libero",
        task="libero_spatial", 
        render_mode="rgb_array", 
        fps=20,
    )

    envs_dict = make_env(cfg)
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    observation, info = env.reset()
    
    le_robot_env = env.unwrapped.envs[0]
    if hasattr(le_robot_env._env, "horizon"):
        max_horizon = le_robot_env._env.horizon
    elif hasattr(le_robot_env._env, "env") and hasattr(le_robot_env._env.env, "horizon"):
        max_horizon = le_robot_env._env.env.horizon
    else:
        max_horizon = 500 # Safe fallback for LIBERO
    print(f"Detected Robosuite Horizon: {max_horizon}")
    
    print(f"Simulation started. Press 'q' in any window to exit.")
    cnt = 0
    try:
        while True:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            # --- VIEW 1: Robosuite Default ---
            # env.render() returns the global scene view
            raw_render = env.render()
            if raw_render is not None:
                # SyncVectorEnv returns a list of frames; take the first
                global_frame = np.array(raw_render[0]) if isinstance(raw_render, (list, tuple)) else raw_render
                cv2.imshow("1. Robosuite Global View", cv2.cvtColor(global_frame, cv2.COLOR_RGB2BGR))
            
            # --- VIEW 2: Agent View (image) ---
            if "image" in observation["pixels"]:
                agent_view = process_image(observation["pixels"]["image"], rotate=True)
                cv2.imshow("2. Agent View (image)", agent_view)

            # --- VIEW 3: Wrist Camera ---
            if "image2" in observation["pixels"]:
                wrist_view =  process_image(observation["pixels"]["image2"],rotate=True)
                cv2.imshow("3. Wrist Camera", wrist_view)



            # Refresh all windows; 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cnt+=1
            is_success = info[0].get("success", False) if isinstance(info, (list, tuple)) else info.get("success", False)
            # Handle end-of-episode to avoid robosuite ValueError
            if any(terminated) or any(truncated) or cnt==max_horizon-10 or is_success:
                print(f"Resetting. Reason: {'Success' if is_success else 'Timeout'}")
                observation, info = env.reset()
                cnt=0
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()

if __name__ == "__main__":
    main()
