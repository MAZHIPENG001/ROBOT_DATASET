import dataclasses
import numpy as np
import collections
import cv2
import tyro
import time

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from device.robot import robot_piper
from device.realsense_camera import RealSenseCamera
from device.keyboard import KeystrokeCounter,KeyCode,Key

'''
@client端口映射
ssh -CNg -L 8000:127.0.0.1:8000 <your_ssh> -p 40053

@serve服务器推理
cd /root/autodl-tmp/openpi/
source /root/autodl-tmp/openpi/.venv/bin/activate
仿真：
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir="/root/autodl-tmp/openpi/pi05_libero"
微调模型checkpoint:20000：
uv run scripts/serve_policy.py policy:checkpoint --policy.config=piper_libero --policy.dir=checkpoints/piper_libero/my_experiment/20000
'''

'''
bash find_all_can_port.sh
bash can_activate.sh can0 1000000 "1-1.2:1.0"
bash can_activate.sh can1 1000000 "1-1.3:1.0"
'''
@dataclasses.dataclass
class Args:
    resize_size: int = 224
    replan_steps: int = 6# 重规划步数(replan_steps),模型每预测一串动作，执行其中的前replan_steps个动作，然后再请求新的预测。

    # Utils
    video_out_path: str = "result/videos"  # Path to save videos


def eval_libero(args: Args) -> None:
    # # client config
    task="Grab the pink square and put it in the box"
    client = websocket_client_policy.WebsocketClientPolicy(host='127.0.0.1',port=8000,)
    # 加载机械臂
    # # can0使能
    # piper_l = robot_piper('can0')
    # piper_l.ConnectPort()
    # piper_l.disable_arm()
    # can1使能
    piper_r = robot_piper('can1')
    piper_r.ConnectPort()
    piper_r.EnableArm()
    # 加载相机
    camera1 = RealSenseCamera(width=640, height=480,serial_number="233622070932")
    camera2 = RealSenseCamera(width=640, height=480,serial_number="938422074612")
    time.sleep(2)

    stop = False
    action_plan = collections.deque()
    with KeystrokeCounter() as key_counter:
        while not stop:
            # Get image
            image_head, _ = camera1.get_images()
            image_wrist, _ = camera2.get_images()
            color_image_combine = np.hstack((image_head, image_wrist))
            cv2.putText(color_image_combine, f'task:{task}', (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('RealSense - Color', color_image_combine)
            cv2.waitKey(1)

            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_head, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(image_wrist, args.resize_size, args.resize_size))

            # # 机械臂
            joint_data = piper_r.read_joint()
            joints = [getattr(joint_data.joint_state, f'joint_{i}') for i in range(1, 7)]
            gripper_data = piper_r.read_gripper()
            gripper = gripper_data.gripper_state.grippers_angle
            state = np.array(joints + [gripper], dtype=np.float32)

            if not action_plan:
                observation = {
                    "image": img,
                    "wrist_image": wrist_img,
                    "state": state,
                    "task": task,
                }
                action_chunk = client.infer(observation)["actions"]
                assert (len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[: args.replan_steps])
            action = action_plan.popleft()
            # print(f"\33[95m{action}\33[0m")
            action = [int(x) for x in action]  # Convert to integers
            piper_r.move_a(action[0], action[1], action[2], action[3], action[4], action[5])
            # piper_r.move_a(action[0],action[1],action[2],action[3],action[4],action[5])
            piper_r.gripper(action[6])
            # 按键判断
            press_events = key_counter.get_press_events()
            # Q: 退出程序
            # C: 开始录制
            # S: 保存当前数据
            # Backspace: 删除最近录制的episode
            for key_stroke in press_events:
                if key_stroke == KeyCode(char='q'):
                    # Exit program
                    print(f'\33[91mExit program!\33[0m')
                    stop = True
                elif key_stroke == KeyCode(char='1'):
                    task = "Grab the pink square and put it in the box"
                    print(f"\33[95mcurrent task:{task}\33[0m")
                elif key_stroke == KeyCode(char='2'):
                    task = "Grab the yellow square and put it in the box"
                    print(f"\33[95mcurrent task:{task}\33[0m")
                elif key_stroke == KeyCode(char='3'):
                    task = "Grab the gray square and put it in the box"
                    print(f"\33[95mcurrent task:{task}\33[0m")
            # time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(eval_libero)