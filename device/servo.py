# https://item.taobao.com/item.htm?from=cart&id=549063298947&mi_id=0000KG1LbWfhTG_c-d-6xvrhBRKi7DFDeuHy6h3fMfW1ALE&skuId=5275236974021&spm=a1z0d.6639537%2F202410.item.d549063298947.18ca7484yW8F4a&upStreamPrice=2673
import serial
import time
import numpy as np

class Servo():
    def __init__(self,port='/dev/ttyUSB0',baudrate=115200,num_joints=7,in_min=[0] * 7,in_max=[0] * 7):
        self.port=port
        self.num_joints = num_joints
        self.baudrate=baudrate
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
        print(f"✅ 成功连接舵机控制板 (端口: {self.port})")

        self.in_min = in_min
        self.in_max = in_max

        self.out_min =  [90000  , 100   , 0         , -100000   , 70000     , 90000     , 0]
        self.out_max =  [-90000 , 180000, -170000   , 100000    , -78000    , -90000    , 70000]
        ''''
        # |joint_name|     limit(rad)       |    limit(angle)    |
        # |----------|     ----------       |     ----------     |
        # |joint1    |   [-2.6179, 2.6179]  |    [-150.0, 150.0] |
        # |joint2    |   [0, 3.14]          |    [0, 180.0]      |
        # |joint3    |   [-2.967, 0]        |    [-170, 0]       |
        # |joint4    |   [-1.745, 1.745]    |    [-100.0, 100.0] |
        # |joint5    |   [-1.22, 1.22]      |    [-70.0, 70.0]   |
        # |joint6    |   [-2.09439, 2.09439]|    [-120.0, 120.0] |
        '''
        # self.in_min = [500        , 500       , 500       , 500       , 500       , 500       , 500]
        # self.in_max = [2500       , 2500      , 2500      , 2500      , 2500      , 2500      , 2500]
        # self.out_min = [-150000   , 0         , -170000   , -100000   , -70000    , -120000   , 0]
        # self.out_max = [150000    , 180000    , 0         , 100000    , 70000     , 120000    , 70000]
        self.servo_init()
        self.read_version()

    def servo_init(self):
        print("⚙️ 正在初始化机械臂...")
        self.torque_off()

    def id_set(self, old_id, new_id):
        """
        修改舵机 ID (表格序号 4)
        注意：修改 ID 时总线上只能连接【这一个】舵机！
        """
        # :03d 确保数字格式化为 3 位，例如 1 变成 001
        command = f"#{old_id:03d}PID{new_id:03d}!"
        response = self.send_command(command)
        print(f"修改 ID: {old_id} -> {new_id}, 返回值: {response}")
    def id_read(self,id):
        command = f"#{id:03d}PID!"
        response = self.send_command(command)
        print(response)

    def read_version(self):
        version = []
        for i in range(self.num_joints):
            command = f"#{i:03d}PID!"
            version.append(self.send_command(command))
            time.sleep(0.2)
        print(f"version read ready\33[92m{version}\33[0m")

    def send_command(self, command: str, expect_response: bool = True, timeout: float = 0.05) -> str:
        """
        发送指令并智能读取返回值
        :param command: 要发送的 ASCII 指令
        :param expect_response: 是否期望有返回值 (如果是广播动作指令，可设为 False)
        :param timeout: 读取超时时间 (默认 0.05 秒/50毫秒)
        """
        # 1. 发送前必须清空接收缓存，丢弃之前的残留/错位数据
        self.ser.reset_input_buffer()

        # 2. 发送当前指令
        self.ser.write(command.encode('ascii'))
        self.ser.flush()

        # 如果明确知道该指令无返回值，直接结束
        if not expect_response:
            return ""

        # 3. 智能等待：最多等 timeout 秒，一旦收到 '!' 立即提前结束
        start_time = time.time()
        response = b""

        while (time.time() - start_time) < timeout:
            if self.ser.in_waiting > 0:
                # 把当前缓冲区的内容追加进来
                response += self.ser.read(self.ser.in_waiting)
                # 检查是否已经收到了完整的结束符 '!'
                if b'!' in response:
                    break  # 收到完整数据，立刻打断循环，不浪费一毫秒

            # 极短的休眠，防止死循环导致 CPU 占用 100%
            time.sleep(0.001)

        return response.decode('ascii', errors='ignore')

    def torque_off(self):
        """
        释放扭力/卸载 (表格序号 5)
        卸载后可以用手掰动机械臂。默认 ID=255 广播全体卸载。
        """
        for id in range(self.num_joints):
            command = f"#{id:03d}PULM!"
            self.send_command(command)
        print("机械臂已失能。")
        time.sleep(1)
    def torque_on(self):
        for id in range(self.num_joints):
            command = f"#{id:03d}PULR!"
            self.send_command(command)
        print("机械臂已使能")
        time.sleep(1)

    def zero_set(self):
        for id in range(self.num_joints):
            command = f"#{id:03d}PSCK!"
            res=self.send_command(command)
            print(res)
    def reset(self):
        for id in range(self.num_joints):
            self.send_command(f"#{id:03d}PCLE!")
        print("✅ 已恢复出场设置")

    def read_position(self, servo_id):
        """
        读取舵机当前位置 (表格序号 9)
        """
        command = f"#{servo_id:03d}PRAD!"
        response = self.send_command(command)
        return response

    def read_all_angles(self):
        """
        读取所有关节当前的角度，并存入 servo_angle 列表
        返回: 包含 7 个整数的列表，例如 [1500, 1500, 1500, 1500, 1500, 1500, 1500]
              如果某个舵机读取失败，对应位置会存入 None
        """
        servo_angle = []  # 初始化空列表

        for servo_id in range(self.num_joints):
            # 1. 发送读取指令
            response = self.read_position(servo_id)

            # 2. 解析返回的字符串
            # 正常返回格式例如: "#001P1500!"
            if response and 'P' in response and response.endswith('!'):
                try:
                    # 按照 'P' 切割字符串，取后半部分，再去掉末尾的 '!'
                    # "#001P1500!" -> 分割得 ["#001", "1500!"] -> 取 "1500!" -> 去掉 "!" 得 "1500"
                    pos_str = response.split('P')[1].replace('!', '')
                    angle = int(pos_str)
                    servo_angle.append(angle)
                except ValueError:
                    print(f"⚠️ 解析舵机 {servo_id} 的数据失败: {response}")
                    servo_angle.append(None)
            else:
                print(f"❌ 舵机 {servo_id} 无响应或返回格式错误")
                servo_angle.append(None)

        return servo_angle
    def control_all_angles(self, servo_angles, duration=1000):
        """
            遍历七维列表并发送舵机指令
            :param servo_angles: 长度为 7 的列表，如 [1500, 1200, 1800, ...]
            :param duration: 运动执行的时间/力度 T 参数，默认 1000
            """
        for index, angle in enumerate(servo_angles):
            # 格式化指令：
            # {:03d} -> ID 补零至3位 (000, 001...)
            # {:04d} -> 角度通常为4位 (1500)
            # T{:d}  -> 时间参数
            command = f"#{index:03d}P{int(angle):04d}T{duration}!"

            # 调用你定义的发送函数
            # 如果是连续控制，为了效率，通常将 expect_response 设为 False
            self.send_command(command, expect_response=False)

    def map_angle_piper(self, val):
        val = np.array(val)

        # 线性映射
        in_min, in_max = np.array(self.in_min), np.array(self.in_max)
        out_min, out_max = np.array(self.out_min), np.array(self.out_max)

        raw_val = (val - in_min) * (out_max - out_min) / (in_max - in_min)
        raw_val += out_min

        # 获取每个通道的绝对最小值和最大值作为边界
        lower = np.minimum(out_min, out_max)
        upper = np.maximum(out_min, out_max)

        # 限幅并转为整数
        safe_val = np.clip(raw_val, lower, upper).astype(int)

        return safe_val.tolist()

if __name__ == "__main__":
    import math

    # 红色
    in_min_red = [2200,870,1480,800,1100,1900,800]
    in_max_red = [790,2240,2000,2000,2100,1000,500]
    # 橙色
    in_min_orange = [2179,1059,880,1000,1300,1980,1580]
    in_max_orange = [776,2321,2000,1900,2500,1020,1280]
    servo = Servo(port='/dev/ttyUSB0',num_joints=7,in_min=in_min_red,in_max=in_max_red)
    start_time = time.time()
    frequency = 0.1  # 频率，数值越大变化越快
    while True:
        servo_angles = servo.read_all_angles()
        print(servo_angles)
        time.sleep(0.2)
        t = time.time() - start_time
        val = 1500 #+ 400 * math.sin(2 * math.pi * frequency * t)
        servo_angles = [int(val)] * 7
        # servo.control_all_angles(servo_angles)
