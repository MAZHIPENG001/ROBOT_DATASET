import dataclasses
import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import pathlib
import tqdm
import collections
import math
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import imageio
import logging
import tyro
import datetime
from openpi_client import image_tools
from openpi_client import websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
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

@dataclasses.dataclass
class Args:
    resize_size: int = 224
    replan_steps: int = 5# 重规划步数(replan_steps),模型每预测一串动作，执行其中的前replan_steps个动作，然后再请求新的预测。

    # LIBERO environment-specific parameters
    # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_suite_name: str = ("libero_spatial")
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 2  # Number of rollouts per task

    # Utils
    video_out_path: str = "result/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)

    # model
    model="pi05_libero"
    """
    Only '''pi05_libero''' was downloaded to the local device
    model could choose:
    pi0_aloha pi05_aloha pi0_aloha_towel pi0_aloha_tupperware pi0_droid pi0_fast_droid pi05_droid pi0_libero
    pi0_libero_low_mem_finetune pi0_fast_libero pi0_fast_libero_low_mem_finetune pi05_libero
    pi0_aloha_pen_uncap pi05_aloha_pen_uncap pi0_fast_full_droid_finetune pi05_full_droid_finetune
    pi05_droid_finetune pi0_aloha_sim debug debug_restore debug_pi05
    """
    checkpoint_dir = "/root/autodl-tmp/openpi/pi05_libero"
    remote_host='172.17.0.11'

def eval_libero(args: Args) -> None:
    # # model config
    # config = _config.get_config(args.model)
    # checkpoint_dir=args.checkpoint_dir
    # # checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")
    # # Create a trained policy.
    # policy = policy_config.create_trained_policy(config, checkpoint_dir)
    client = websocket_client_policy.WebsocketClientPolicy(host='127.0.0.1',port=8000,)
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # 创建视频保存文件夹
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # 不同任务设置最大步数
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Start evaluation 任务数量：num_tasks_in_suite(10)个
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        print(f"\33[92mEvaluating task {task_id+1}:{task_description}\33[0m")
        logging.info(f"\33[1,92mEvaluating task {task_id+1}:{task_description}\33[0m")
        # Start episodes 每个任务测试num_trials_per_task(50)次
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # # Finished executing previous action chunk -- compute new chunk
                        # # Prepare observations dict
                        # element = {
                        #     "observation/image": img,
                        #     "observation/wrist_image": wrist_img,
                        #     "observation/state": np.concatenate(
                        #         (
                        #             obs["robot0_eef_pos"],
                        #             _quat2axisangle(obs["robot0_eef_quat"]),
                        #             obs["robot0_gripper_qpos"],
                        #         )
                        #     ),
                        #     "prompt": str(task_description),
                        # }
                        observation = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        #inference
                        # action_chunk = policy.infer(element)["actions"]
                        action_chunk = client.infer(observation)["actions"]
                        assert (len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])
                        print(f"\33[93m当前任务:{task_id+1}/{num_tasks_in_suite};当前次数:{episode_idx+1}/{args.num_trials_per_task};当前步数:{t}/{max_steps + args.num_steps_wait}\33[0m")
                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            # imageio.mimwrite(
            #     pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
            #     [np.asarray(x) for x in replay_images],
            #     fps=10,
            # )
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rollout_{task_segment}_task{task_id}_ep{episode_idx}_{timestamp}_{suffix}.mp4"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / filename,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            print(f"\33[95msave video:{filename}\33[0m")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("result/libero_eval.log"),  # 保存到文件
                            logging.StreamHandler()  # 输出到屏幕
                        ],
                        filemode='w'  # 'a' 为追加模式，'w' 为覆盖模式
                        )
    # eval_libero(Args())
    tyro.cli(eval_libero)
