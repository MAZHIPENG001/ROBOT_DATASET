# ROBOT_DATASET
真神：https://gemini.google.com/app

## 效果：
### 数据收集
https://www.bilibili.com/video/BV1ZXBTBCEEs/?vd_source=21305f4d66e9c0e234a3094ab6e4e0e6
### openpi05
https://www.bilibili.com/video/BV1vzBMBGEzu/?vd_source=21305f4d66e9c0e234a3094ab6e4e0e6
松灵机械臂遥操作数据采集与openpi微调

运行robot_dataset.py按键控制数据记录：
```bash
c:开始记录;
s:保存当前数据;
q:退出程序;
Backspace:删除当前缓存数据
```
## Github仓库
```bash
cd ~
mkdir GithubDoc
cd GithubDoc
# 官方仓库(由于部分python库版本不匹配，需要修改)
git clone https://github.com/Physical-Intelligence/openpi.git
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout
git switch origin/user/azouitine/2025-04-24-hot-fix-ci
```
修改(如需要)
```python
'''
******************************************
/lerobot/lerobot/common/datasets/lerobot_dataset.py
******************************************
'''
# Check timestamps
# timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
timestamps = torch.stack(list(self.hf_dataset["timestamp"])).numpy()
# episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
episode_indices = torch.stack(list(self.hf_dataset["episode_index"])).numpy()
```
### 环境创建：参考：https://github.com/Physical-Intelligence/openpi
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/GithubDoc/openpi
# 配置基本环境
GIT_LFS_SKIP_SMUDGE=1 uv sync
# 安装openpi
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
# 安装lerobot
cd ..
cd lerobot
uv pip install -e .
```


## 准备
### huggingface登录
涉及将数据上传到huggingface
```bash
huggingface-cli login
```
token获取：https://huggingface.co/settings/tokens 选择Creat new token。在终端输入不会显示，注意为token开启权限！！！

## robot_dataset.py部分信息介绍
### 1.相机：
相机使用/device/realsense_camera.py查看设备号
```python
# 加载相机
camera1 = RealSenseCamera(width=640, height=480,serial_number="233622070932")
camera2 = RealSenseCamera(width=640, height=480,serial_number="938422074612")
```
### 2.机械臂：
具体参照piper_sdk说明文档，本处简要列举命令
```bash
#查找端口
bash find_all_can_port.sh
#选择端口连接
bash can_activate.sh can0 1000000 "1-1.2:1.0"
bash can_activate.sh can1 1000000 "1-1.3:1.0"
```
### 3.数据集：
1.是否复用数据集，尽量一次性采集数据，设置resume=False，否则机械臂抖动较大（可能由于网络延迟问题）
```python
# 加载已存在数据集
ld = lerobotdata(resume=True)
# 创建新数据集
ld = lerobotdata(resume=False)
```
2.上传到huggingface，对网络有要求，采集完数据后取消注释再次运行代码可上传，可手动在网页端上传
```python
ld.dataset.push_to_hub(
    repo_id="mazhipeng/piper_dataset",  # 替换为你的仓库名
    private=False,  # 可选：设为True保持私有
    commit_message="Initial dataset upload"
)
```
3.数据归一化:processer/compute_norm_stats.py

相对于openpi文件修改_config
```python
# import openpi.training.config as _config
import unit.dataset_config as _config
```
4.对新数据集第一次计算需要从huggingface下载文件

网络配置
```python
import os
PROXY_URL = "http://127.0.0.1:7897"
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL
os.environ['all_proxy']   = PROXY_URL
```
5.数据集检查
确保数据集版本为2.1(3.0结构有区别：/meta对数据集的说明文件缺失，模型训练需要episodes.jsonl  episodes_stats.jsonl  info.json  tasks.jsonl)

数据集保存位置：~/.cache/huggingface/lerobot

进入/meta，info.json文件内容应包括："codebase_version": "v2.1",
```
#目录结构：
mzp@MA:~/.cache/huggingface/lerobot/mazhipeng/piper_dataset$ ls -R
.:
data  meta

./data:
chunk-000

./data/chunk-000:
episode_000000.parquet  ...共采集n条信息，则有n个.parquet文件

./meta:
episodes.jsonl  episodes_stats.jsonl  info.json  tasks.jsonl
```
6.~~数据集版本为3.0时，openpi/src/openpi/transforms.py可能报错，需修改309行左右||||无报错不要修改~~
```python
@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')
        # task_index = int(data["task_index"])
        # print(f"\33[93m{self.tasks}\33[0m")
        # for key, value in self.tasks.items():
        #     print(f"Key: '{key}' | Type: {type(key)}")
        #     print(f"Value: {value} | Type: {type(value)}")
        #     print("-" * 20)
        # if (prompt := self.tasks.get(task_index)) is None:
        #     raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")
        if (task_series := self.tasks.get("task_index")) is not None:
            # 再将输入的 task_index 转为整数，并从 Series 中取值
            target_idx = int(data["task_index"])
            prompt = task_series[target_idx]
        else:
            raise ValueError(f"Key 'task_index' not found in {self.tasks.keys()}")
        return {**data, "prompt": prompt}
```

## openpi修改：
### 配置文件
1.文件修改

将unit/dataset_config.py 复制到 /openpi/src/openpi/training/dataset_config.py（替换代替config）

将processer/libero_policy.py 复制到 /openpi/src/openpi/policies/liberopolicy.py（替换libero_policy）

或者修改原文件，则不必复制与修改
修改/openpi/scripts/train.py23行

```
# openpi/scr相对于openpi修改配置文件（也可以合并）
# import openpi.training.config as _config
import openpi.training.dataset_config as _config
```
2.文件修改说明:
```python
'''
******************************************
dataset_config.py
******************************************
'''
# 数据集处理
@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    数据转换，修改为自己的数据集
    """
    # 离散化?
    # True:需要；False:不需要
    extra_delta_transform: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        # repack_transform:重新映射键名。
        """
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        """
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "image": "image",
                        "wrist_image": "wrist_image",
                        "state": "state",
                        "actions": "actions",
                        "task": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        # 模型 输入格式、输出格式  待修改
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.
        # pi0模型是在相对动作(抓手动作总是绝对的)上训练的（相对于第一个）
        # pi0模型训练[j1,j2,j3,...,jn,gripper],j*为相对，gripper为绝对
        # 如果数据有“绝对”动作（例如目标关节角度），取消下面一行的注释，将动作转换为相对动作。
        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            # 前6个动作（关节）应用增量转换，并保持第7个动作（抓手）不变
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            # delta_action_mask = (True, True, True, True, True, True, False)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
            '''# DeltaActions
            # state= [s0, s1, s2, s3, s4, s5, s6, ...]  # 假设有更多维度
            # actions = [a0, a1, a2, a3, a4, a5, a6, a7, ...]  # 动作维度 ≥ 7
            # 将位置为True变为增量
            # actions[0] = a0 - s0  # 因为 mask[0] = True
            # actions[1] = a1 - s1  # 因为 mask[1] = True
            # actions[2] = a2 - s2  # 因为 mask[2] = True
            # actions[3] = a3 - s3  # 因为 mask[3] = True
            # actions[4] = a4 - s4  # 因为 mask[4] = True
            # actions[5] = a5 - s5  # 因为 mask[5] = True
            # actions[6] = a6  # 保持不变，因为 mask[6] = False
            # actions[7] = a7  # 保持不变，因为 dims=7，只处理前7维
            # AbsoluteActions
            # 将True部分变为绝对
            # actions[0] = a0 + s0  # 恢复原始 a0
            # actions[1] = a1 + s1  # 恢复原始 a1
            # actions[2] = a2 + s2  # 恢复原始 a2
            # actions[3] = a3 + s3  # 恢复原始 a3
            # actions[4] = a4 + s4  # 恢复原始 a4
            # actions[5] = a5 + s5  # 恢复原始 a5
            # actions[6] = a6  # 保持不变
            # actions[7] = a7  # 保持不变'''

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
```

```python
'''
******************************************
dataset_config.py
******************************************
'''
# 训练配置
# _CONFIGS增加配置：
#lora微调版本
    TrainConfig(
        name="piper_libero",
        # 1. 修改模型配置，启用 LoRA 变体
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="mazhipeng/piper_dataset",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 2. 设置冻结过滤器（LoRA 必需）
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),

        # 3. LoRA 建议关闭 EMA
        ema_decay=None,

        batch_size=4,  # 在 LoRA 模式下，4 应该可以跑通，甚至可以尝试 8 或 16
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
# 全量微调版本
    TrainConfig(
        # Change the name to reflect your model and dataset.
        # 使用"get_config(piper_libero)"获取配置"TrainConfig"，
        name="piper_liberosadasd",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),#使用pi0作为基础模型
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        # 更改repo_id以指向使用数据集。
        # 同时修改DataConfig以使用你在上面为你的数据集做的新配置
        data=LeRobotLiberoDataConfig(
            repo_id="mazhipeng/piper_dataset",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,#是否加载任务指令task
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        # 加载哪个模型?待修改
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        # 定义其他超参数，如学习率，训练步骤数
        num_train_steps=30_000,
    ),
```
```python
'''
******************************************
liberopolicy.py
******************************************
'''
修改数据映射关系
@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "task" in data:
            inputs["prompt"] = data["task"]
        return inputs
```
## openpi训练
在autodl云服务器运行：https://www.autodl.com/home
```bash
cd /root/autodl-tmp/openpi/
# 梯子
source /etc/network_turbo
# uv环境
source /root/autodl-tmp/openpi/.venv/bin/activate
# 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py piper_libero --exp-name=my_experiment --overwrite
```
### wandb
wandb.ai
### 1.datasets版本问题：
解决“版本代差”导致的元数据不识别最直接的办法。需要手动在你当前的虚拟环境中做一个简单的类型映射。

打开报错的文件： vi /root/autodl-tmp/openpi/.venv/lib/python3.11/site-packages/datasets/features/features.py

找到 generate_from_dict 函数（大约在第 1460-1470 行左右）。
在函数内部开头部分，添加一行代码，将 'List' 强制映射为 'Sequence'：
```Python
def generate_from_dict(obj):
    # ... 原有代码 ...
    _type = obj.get("_type")
    
    # --- 添加下面这一行 ---
    if _type == "List": _type = "Sequence"
    # ----------------------
```
# 模型使用
## 仿真
### ~~1.本地推理：4060带不动~~
### 2.云服务器本地推理：
```bash
uv run /scripts/inference_local.py
```
### 3.远程推理

本地获取数据，发送至服务器推理，结果返回本地：
```
# 端口映射
ssh -CNg -L 8000:127.0.0.1:8000 <服务器ssh> -p 40053
# 本地端
uv run /openpi/scripts/client_policy.py
```
```
# 服务器端
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir="/root/autodl-tmp/openpi/pi05_libero"
```
## 实际使用
### 远程推理
本地获取数据，发送至服务器推理，结果返回本地：

#### 修改：
```python
'''
******************************************
/root/autodl-tmp/openpi/src/openpi/policies/policy_config.py
******************************************
'''
# from openpi.training import config as _config
from openpi.training import dataset_config as _config
```
#### 运行：
```bash
# 端口映射
ssh -CNg -L 8000:127.0.0.1:8000 root@connect.westb.seetacloud.com -p 40053
# 本地端
/home/mzp/ROBOT_DATASET/client_policy_piper.py
```
```bash
# 服务器端 checkpoint:2000：
uv run scripts/serve_policy.py policy:checkpoint --policy.config=piper_libero --policy.dir=checkpoints/piper_libero/my_experiment/20000
```







=======
piper &amp; openpi
>>>>>>> c89ee8554ea5bf445bfe8ded0ad0aa1ae055e1bb
