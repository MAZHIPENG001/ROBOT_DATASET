from piper_sdk import *
import numpy as np
from processer.transfer import euler_to_quaternion,transform_matrix

class robot_piper(C_PiperInterface_V2):
    """
    _a/_p:关节控制/末端控制
    gripper:夹爪
    read:信息读取
    """
    def __init__(self,can='can0'):
        super().__init__(can)
        # 初始位置
        self.init = [0, 0.4, -0.7, 0.0, 1.1, 0.0]
        factor = 180 * 1000 / 3.14
        self.joint_0_init = round(self.init[0] * factor)
        self.joint_1_init = round(self.init[1] * factor)
        self.joint_2_init = round(self.init[2] * factor)
        self.joint_3_init = round(self.init[3] * factor)
        self.joint_4_init = round(self.init[4] * factor)
        self.joint_5_init = round(self.init[5] * factor)
        self.T_flange_tcp = np.array([
                            [1, 0, 0, 0],#大前小后
                            [0, 1, 0, 0],#大右
                            [0, 0, 1, -0.10],#大高小矮
                            [0, 0, 0, 1]])

    def start(self):
        self.ConnectPort()
        self.EnableArm()

    def enable_arm(self):#使能
        self.EnableArm()
    def disable_arm(self):#失能
        self.DisableArm()

    def move_a(self,j1,j2,j3,j4,j5,j6):
        self.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.JointCtrl(j1,j2,j3,j4,j5,j6)
    def move_p(self,X_target,Y_target,Z_target,RX_target, RY_target, RZ_target):
        self.MotionCtrl_2(0x01, 0x00, 20, 0x00)
        self.EndPoseCtrl(round(X_target), round(Y_target), round(Z_target),
                          RX_target, RY_target, RZ_target)
    def move_zero_a(self):#关节控制移动到零点
        self.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        self.JointCtrl(0, 0, 0, 0, 0, 0)
    def move_init_p(self):#末端控制移动到初始位置
        self.MotionCtrl_2(0x01, 0x00, 50, 0x00)
        self.EndPoseCtrl(40000, 0, 333000, 180000, 58000, 180000)
    def move_init_a(self):#关节控制移动到初始位置
        self.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        self.JointCtrl(self.joint_0_init, self.joint_1_init, self.joint_2_init,
                             self.joint_3_init, self.joint_4_init, self.joint_5_init)

    def gripper(self,width):
        self.GripperCtrl(width,1000,0x01,0)
    def gripper_on(self):
        self.gripper(70000)
    def gripper_off(self):
        self.gripper(0)

    def grasp_end(self):
        '''收尾工作'''
        # 夹取物体
        self.gripper_off()
        time.sleep(1)
        # 复位
        # self.move_a(0, 95000, -80000, 0, 0, 0)
        # time.sleep(0.3)
        self.move_a(self.joint_0_init, self.joint_1_init, self.joint_2_init,
                    self.joint_3_init, self.joint_4_init, self.joint_5_init)
        time.sleep(1)
        # 移动到盒子上方
        self.move_a(-90000, 80000, -50000, 0, 60000, 0)
        time.sleep(1)
        # 松开夹爪
        self.gripper_on()
        time.sleep(1)
        self.move_a(self.joint_0_init, self.joint_1_init, self.joint_2_init,
                    self.joint_3_init, self.joint_4_init, self.joint_5_init)
        self.gripper_off()

    def read_end_pose(self):
        ArmEndPoseMsgs=self.GetArmEndPoseMsgs()
        return ArmEndPoseMsgs
    def read_joint(self):
        ArmJointMsgs=self.GetArmJointMsgs()
        return ArmJointMsgs
    def read_gripper(self):
        GetArmGripperMsgs=self.GetArmGripperMsgs()
        return GetArmGripperMsgs
    def cal_T0(self):
        ArmEndPoseMsgs=self.read_end_pose()
        '''输入机械臂位置信息->计算T0'''
        X = ArmEndPoseMsgs.end_pose.X_axis / 1000 / 1000
        Y = ArmEndPoseMsgs.end_pose.Y_axis / 1000 / 1000
        Z = ArmEndPoseMsgs.end_pose.Z_axis / 1000 / 1000
        RX = ArmEndPoseMsgs.end_pose.RX_axis / 180 * 3.1415926 / 1000
        RY = ArmEndPoseMsgs.end_pose.RY_axis / 180 * 3.1415926 / 1000
        RZ = ArmEndPoseMsgs.end_pose.RZ_axis / 180 * 3.1415926 / 1000

        rotation_euler = [RX, RY, RZ]
        rotation_quat = euler_to_quaternion(rotation_euler)
        # print("末端位置：", X, Y, Z, rotation_quat)

        t0 = transform_matrix([X, Y, Z], rotation_quat=rotation_quat)
        return t0

    def grasp_order(self,translations,rotation_mats_3x3,Pixel):
        # 从新排序：由近及远
        translations = np.array(translations)
        rotation_mats_3x3 = np.array(rotation_mats_3x3)
        Pixel = np.array(Pixel)
        # 顺序
        index = np.argsort(translations[:, 2])  # 第三分量索引是2
        # 按照排序索引重新排列数据
        translations = translations[index]
        rotation_mats_3x3 = rotation_mats_3x3[index]
        Pixel = Pixel[index]
        return translations, rotation_mats_3x3, Pixel

if __name__ == '__main__':
    from device.keyboard import KeystrokeCounter, KeyCode
    import time
    piper = robot_piper()
    piper.ConnectPort()
    while True:
        with KeystrokeCounter() as key_counter:
            try:
                while True:
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        # 按下o：归零(关节控制)
                        if key_stroke == KeyCode(char='o'):
                            print("按下o：归零(关节控制)")
                            piper.move_zero_a()
                        # 按下i: 复位(关节控制)
                        elif key_stroke == KeyCode(char='i'):
                            print("按下i:复位(关节控制)")
                            piper.move_init_p()
                        elif key_stroke == KeyCode(char='p'):
                            print("按下i:复位(末端控制)")
                            piper.move_init_p()
                        elif key_stroke == KeyCode(char='m'):
                            print("按下m:读取末端位置")
                            piper.read_end_pose()
                        elif key_stroke == KeyCode(char='e'):
                            print("按下e:使能")
                            piper.enable_arm()
                        elif key_stroke == KeyCode(char='d'):
                            print("按下d:失能")
                            piper.disable_arm()
            except KeyboardInterrupt:
                events = key_counter.get_press_events()
                print(events)