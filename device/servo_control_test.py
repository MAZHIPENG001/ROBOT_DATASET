import  time

from device.robot import robot_piper
from device.servo import Servo


def main():
    piper_l = robot_piper('can0')
    piper_l.ConnectPort()
    piper_l.EnableArm()
    # piper_l.disable_arm()

    time.sleep(1)
    servo_l = Servo(port='/dev/ttyUSB0')

    while True:
        servo_l_joints = servo_l.read_all_angles()
        # piper_joints=piper_l.read_joint()

        print(f"\33[93m servo_l_joints:{servo_l_joints}\33[0m")
        # print(f"\33[92m piper_joints:{piper_joints}\33[0m")
        *map_joints_l, map_gripper_l = servo_l.map_angle_piper(val=servo_l_joints)

        piper_l.move_a(*map_joints_l)
        piper_l.gripper(map_gripper_l)
        time.sleep(0.2)

if __name__ == '__main__':
    main()
