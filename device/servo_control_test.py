import  time

from device.robot import robot_piper
from device.servo import Servo


def main():
    piper_l = robot_piper('can0')
    piper_l.ConnectPort()
    piper_l.EnableArm()

    piper_r = robot_piper('can1')
    piper_r.ConnectPort()
    piper_r.EnableArm()

    # 红色
    in_min_red = [2200, 870, 1480, 800, 1100, 1900, 800]
    in_max_red = [790, 2240, 2000, 2000, 2100, 1000, 500]
    # 橙色
    in_min_orange = [2179, 1059, 880, 1000, 1300, 1980, 1580]
    in_max_orange = [776, 2321, 2000, 1900, 2500, 1020, 1280]
    servo_l = Servo(port='/dev/ttyUSB0', in_min=in_min_orange, in_max=in_max_orange)
    servo_r = Servo(port='/dev/ttyUSB1', in_min=in_min_red, in_max=in_max_red)

    while True:
        servo_l_joints = servo_l.read_all_angles()
        # piper_joints=piper_l.read_joint()
        servo_r_joints = servo_r.read_all_angles()

        print(f"\33[93m servo_l_joints:{servo_l_joints}\33[0m")
        # print(f"\33[92m piper_joints:{piper_joints}\33[0m")
        *map_joints_l, map_gripper_l = servo_l.map_angle_piper(val=servo_l_joints)
        *map_joints_r, map_gripper_r = servo_r.map_angle_piper(val=servo_r_joints)

        piper_l.move_a(*map_joints_l)
        piper_l.gripper(map_gripper_l)
        piper_r.move_a(*map_joints_r)
        piper_r.gripper(map_gripper_r)
        time.sleep(0.2)

if __name__ == '__main__':
    main()
