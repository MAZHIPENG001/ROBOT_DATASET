import time
import click
import cv2
from unit.lerobot_data import lerobotdata
from processer.libero_policy import _parse_image as parase_image
from device.keyboard import KeystrokeCounter,KeyCode,Key
from device.robot import robot_piper
from device.realsense_camera import RealSenseCamera
import numpy as np
import os

PROXY_URL = "http://127.0.0.1:7897"
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL
os.environ['all_proxy']   = PROXY_URL
'''
bash find_all_can_port.sh
bash can_activate.sh can0 1000000 "1-1.2:1.0"
bash can_activate.sh can1 1000000 "1-1.3:1.0"
'''
def main():
    task="Grab the pink square and put it in the box"
    # 加载机械臂
    # can0使能
    piper_l = robot_piper('can0')
    piper_l.ConnectPort()
    piper_l.disable_arm()
    # can1使能
    piper_r = robot_piper('can1')
    piper_r.ConnectPort()
    piper_r.EnableArm()
    # 加载相机
    camera1 = RealSenseCamera(width=640, height=480,serial_number="233622070932")
    camera2 = RealSenseCamera(width=640, height=480,serial_number="938422074612")
    # 定义数据集
    ld = lerobotdata(resume=True)
    # # 上传到huggingface
    # ld.dataset.push_to_hub(
    #     repo_id="mazhipeng/piper_dataset",  # 替换为你的仓库名
    #     private=False,  # 可选：设为True保持私有
    #     commit_message="Initial dataset upload"
    # )

    # ld.dataset.compute_stats()
    print(f'\33[93m{ld.dataset}\33[0m')

    # # dataset read example
    # print(ld.dataset[100]['task_index'])
    # daimg=np.array(ld.dataset[100]['image'].permute(1, 2, 0))
    # print(f"\33[93m{daimg.shape}\33[0m")
    # cv2.imshow('data_img', daimg)

    stop=False
    is_recording=False
    idx=0
    prev_actions=None
    with KeystrokeCounter() as key_counter:
        while not stop:
            # # 机械臂
            joint_data = piper_l.read_joint()
            joints = [getattr(joint_data.joint_state, f'joint_{i}') for i in range(1, 7)]
            gripper_data=piper_l.read_gripper()
            gripper=gripper_data.gripper_state.grippers_angle
            actions = np.array(joints + [gripper], dtype=np.float32)
            if prev_actions is None:
                state = actions.copy()  # 或np.zeros_like(current_actions)
            else:
                state = prev_actions  # 使用上一时刻动作作为state
            prev_actions = actions.copy()
            # 跟随
            piper_r.move_a(*joints)
            piper_r.gripper(gripper)
            # 相机
            image_head, _ = camera1.get_images()
            image_wrist, _ = camera2.get_images()
            color_image_combine=np.hstack((image_head, image_wrist))
            cv2.putText(color_image_combine, f'is_recording:{is_recording}', (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image_combine, f'episodes:{ld.dataset.num_episodes}', (0, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image_combine, f'idx:{idx}', (0, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('RealSense - Color', color_image_combine)
            cv2.waitKey(1)
            image_head = parase_image(image_head)
            image_wrist = parase_image(image_wrist)
            # 缓存数据
            if is_recording:
                ld.keep_record(image_head,image_wrist,state,actions,task)
                # ld.keeping_record()
                idx+=1

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
                elif key_stroke == KeyCode(char='c'):
                    print(f'\33[92mStart Recording!\33[0m')
                    key_counter.clear()
                    is_recording = True
                elif key_stroke == KeyCode(char='s'):
                    # Stop recording
                    key_counter.clear()
                    is_recording = False
                    idx=0
                    ld.save_record()
                    print(ld.dataset)
                    print(f'\33[91mStopped.\33[0m')
                elif key_stroke == KeyCode(char='1'):
                    task = "Grab the pink square and put it in the box"
                    print(f"\33[95mcurrent task:{task}\33[0m")
                elif key_stroke == KeyCode(char='2'):
                    task = "Grab the yellow square and put it in the box"
                    print(f"\33[95mcurrent task:{task}\33[0m")
                elif key_stroke == KeyCode(char='3'):
                    task = "Grab the gray square and put it in the box"
                    print(f"\33[95mcurrent task:{task}\33[0m")
                elif key_stroke == Key.backspace:
                    # Delete the most recent recorded episode
                    is_recording = False
                    if click.confirm('Are you sure to drop an episode?'):
                        key_counter.clear()
                        ld.del_record()
                        idx=0

            time.sleep(0.1)

if __name__ == "__main__":
    # tyro.cli(main)
    main()