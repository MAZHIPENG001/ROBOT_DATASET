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
                "state": state,  # 末端位置x、y、z、rx、ry、rz、夹爪
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
                "state": step["observation"]["state"],  # 末端位置x、y、z、rx、ry、rz、夹爪
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