from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import shutil
import numpy as np
np.random.seed(42)
class lerobotdata():
    def __init__(self,resume=True):
        super().__init__()
        self.resume=resume
        self.REPO_ID="mazhipeng/piper_dataset"
        self.output_path = HF_LEROBOT_HOME / self.REPO_ID
        self.num_episodes=50
        # 是否创建新数据集
        if self.resume:
            self.dataset =LeRobotDataset(self.REPO_ID,root=self.output_path,)
            print(f"\33[92m加载数据集:{self.output_path},当前episodes:{self.dataset.num_episodes}\33[0m")
        else:
            print(f"\33[92m创建新数据集:{self.output_path}\33[0m")
            # Clean up any existing dataset in the output directory
            if self.output_path.exists():
                shutil.rmtree(self.output_path)
            # Create empty dataset
            # Create LeRobot dataset, define features to store
            self.dataset = LeRobotDataset.create(
                repo_id=self.REPO_ID,
                robot_type="piper",
                fps=10,
                features={
                    "image": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "state": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (7,),
                        "names": ["actions"],
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,)
        # 当前数据条数
        self.recorded_episodes=self.dataset.num_episodes
    def keep_record(self,image_head,image_wrist,state,actions,task):
        self.dataset.add_frame(
            {
                "image": image_head,  # 外部相机
                "wrist_image": image_wrist,  # 手上相机
                "state": state,  # j1、j2、j3、j4、j5、j6、夹爪
                "actions": actions,
                "task": task,  # .decode(),
            }
        )

    def keeping_record(self):
        step=self.getstep()
        self.dataset.add_frame(
            {
                "image": step["observation"]["image"],#外部相机
                "wrist_image": step["observation"]["wrist_image"],#手上相机
                "state": step["observation"]["state"],  # j1、j2、j3、j4、j5、j6、夹爪
                "actions": step["action"],
                "task": step["language_instruction"],  # .decode(),
            }
        )
    def save_record(self):
        self.dataset.save_episode()
        self.recorded_episodes = self.dataset.num_episodes
        print(f"\33[92m数据保存完成,当前数据:{self.recorded_episodes}条\33[0m")
    def del_record(self):
        self.dataset.clear_episode_buffer()
        print(f"\33[93m当前episode缓存已清空\33[0m")
    def getstep(self):
        # tasks=["pick up the red block",]
        # language_instruction = np.random.choice(tasks)
        language_instruction = "pick up the blue block"
        step = {
            "observation": {
                "image": np.random.randint(1, 255, size=(480, 640, 3), dtype=np.uint8),  # 主相机图像
                "wrist_image": np.random.randint(1, 255, size=(480, 640, 3), dtype=np.uint8),  # 手腕相机图像
                "state": np.random.randn(7).astype(np.float32),  # 状态向量
            },
            "action": np.random.randn(7).astype(np.float32),  # 动作向量
            "language_instruction": language_instruction,  # 任务指令的字节串
        }
        return step

class lerobotdata_dual():
    def __init__(self,resume=True):
        super().__init__()
        self.resume=resume
        self.REPO_ID="mazhipeng/piper_dataset_dual"
        self.output_path = HF_LEROBOT_HOME / self.REPO_ID
        self.num_episodes=50
        # 是否创建新数据集
        if self.resume:
            self.dataset =LeRobotDataset(self.REPO_ID,root=self.output_path,)
            print(f"\33[92m加载数据集:{self.output_path},当前episodes:{self.dataset.num_episodes}\33[0m")
        else:
            print(f"\33[92m创建新数据集:{self.output_path}\33[0m")
            # Clean up any existing dataset in the output directory
            if self.output_path.exists():
                shutil.rmtree(self.output_path)
            # Create empty dataset
            # Create LeRobot dataset, define features to store
            self.dataset = LeRobotDataset.create(
                repo_id=self.REPO_ID,
                robot_type="piper",
                fps=10,
                features={
                    "head_image": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "left_wrist_image": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "right_wrist_image": {
                        "dtype": "image",
                        "shape": (480, 640, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "state": {
                        "dtype": "float32",
                        "shape": (14,),
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": (14,),
                        "names": ["actions"],
                    },
                },
                image_writer_threads=10,
                image_writer_processes=5,)
        # 当前数据条数
        self.recorded_episodes=self.dataset.num_episodes

    def keep_record(self,head_image_head,left_wrist_image,right_wrist_image,state,actions,task):
        self.dataset.add_frame(
            {
                "head_image": head_image_head,  # 外部相机
                "left_wrist_image": left_wrist_image,  # 左手上相机
                "right_wrist_image": right_wrist_image,  # 右手上相机
                "state": state,  # j1、j2、j3、j4、j5、j6、j1、j2、j3、j4、j5、j6、夹爪、夹爪
                "actions": actions,
                "task": task,  # .decode(),
            }
        )

    def keeping_record(self):
        step=self.getstep()
        self.dataset.add_frame(
            {
                "head_image": step["observation"]["head_image"],#外部相机
                "left_wrist_image": step["observation"]["left_wrist_image"],#左手上相机
                "right_wrist_image": step["observation"]["right_wrist_image"],  # 右手上相机
                "state": step["observation"]["state"],  # j1、j2、j3、j4、j5、j6、j1、j2、j3、j4、j5、j6、夹爪、夹爪
                "actions": step["action"],
                "task": step["language_instruction"],  # .decode(),
            }
        )
    def save_record(self):
        self.dataset.save_episode()
        self.recorded_episodes = self.dataset.num_episodes
        print(f"\33[92m数据保存完成,当前数据:{self.recorded_episodes}条\33[0m")
    def del_record(self):
        self.dataset.clear_episode_buffer()
        print(f"\33[93m当前episode缓存已清空\33[0m")
    def getstep(self):
        # tasks=["pick up the red block",]
        # language_instruction = np.random.choice(tasks)
        language_instruction = "pick up the blue block"
        step = {
            "observation": {
                "head_image": np.random.randint(1, 255, size=(480, 640, 3), dtype=np.uint8),  # 主相机图像
                "left_wrist_image": np.random.randint(1, 255, size=(480, 640, 3), dtype=np.uint8),  # 左手腕相机图像
                "right_wrist_image": np.random.randint(1, 255, size=(480, 640, 3), dtype=np.uint8),  # 右手腕相机图像
                "state": np.random.randn(14).astype(np.float32),  # 状态向量
            },
            "action": np.random.randn(14).astype(np.float32),  # 动作向量
            "language_instruction": language_instruction,  # 任务指令的字节串
        }
        return step

if __name__ == "__main__":
    import time
    import click
    import sys
    sys.path.append('/home/mzp/ROBOT_DATASET')
    from device.keyboard import KeystrokeCounter,KeyCode,Key

    ld=lerobotdata_dual(resume=False)
    print(f'\33[93m{ld.dataset}\33[0m')

    stop = False
    is_recording = False
    with KeystrokeCounter() as key_counter:
        while not stop:
            # 缓存数据
            if is_recording:
                ld.keeping_record()
            # 按键判断
            press_events = key_counter.get_press_events()
            # Q: 退出程序
            # C: 开始录制
            # S: 保存当前数据
            # num: order
            # Backspace: 删除最近录制的episode
            for key_stroke in press_events:
                if key_stroke == KeyCode(char='q'):
                    # Exit program
                    print(f'\33[91mExit program!\33[0m')
                    stop = True
                elif key_stroke == KeyCode(char='c'):
                    print(f'\33[92mStart Recording!\33[0m')
                    key_counter.clear()
                    is_recording = True
                elif key_stroke == KeyCode(char='s'):
                    # Stop recording
                    key_counter.clear()
                    is_recording = False
                    ld.save_record()
                    print(ld.dataset)
                    print(f'\33[91mStopped.\33[0m')

                elif key_stroke == Key.backspace:
                    # Delete the most recent recorded episode
                    is_recording = False
                    if click.confirm('Are you sure to drop an episode?'):
                        key_counter.clear()
                        ld.del_record()
                        prev_actions = None
                        idx = 0

            time.sleep(0.1)
