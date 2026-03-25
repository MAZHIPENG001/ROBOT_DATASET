# https://item.taobao.com/item.htm?from=cart&id=549063298947&mi_id=0000KG1LbWfhTG_c-d-6xvrhBRKi7DFDeuHy6h3fMfW1ALE&skuId=5275236974021&spm=a1z0d.6639537%2F202410.item.d549063298947.18ca7484yW8F4a&upStreamPrice=2673
import serial
import time
class Servo():
    def __init__(self,port='/dev/ttyUSB',baudrate=115200,num_joints=7):
        self.port=port
        self.baudrate=baudrate
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.5)
        print(f"✅ 成功连接舵机控制板 (端口: {self.port})")
        self.in_min = [500, 500, 500, 500, 500, 500, 500]
        self.in_max = [2500, 2500, 2500, 2500, 2500, 2500, 2500]
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
        self.out_min = [-150000, 0, -170000, -100000, -70000, -120000, 0]
        self.out_max = [150000, 180000, 0, 100000, 70000, 120000, 70000]
        self.num_joints=num_joints
        self.servo_init()

    def servo_init(self):
        print("⚙️ 正在初始化机械臂...")
        # self.zero_set()
        self.reset()
        self.torque_off()
        time.sleep(0.1)
        print("✅ 初始化完成！机械臂已失能。")

    def id_set(self, old_id, new_id):
        """
        修改舵机 ID (表格序号 4)
        注意：修改 ID 时总线上只能连接【这一个】舵机！
        """
        # :03d 确保数字格式化为 3 位，例如 1 变成 001
        command = f"#{old_id:03d}PID{new_id:03d}!"
        response = self.send_command(command)
        print(f"修改 ID: {old_id} -> {new_id}, 返回值: {response}")

    def send_command(self, command: str) -> str:
        """Send serial command and read response

        Args:
            command: Command string to send

        Returns:
            Response string, returns empty string if no response
        """
        self.ser.write(command.encode('ascii'))
        time.sleep(0.008)
        response = self.ser.read_all()
        return response.decode('ascii', errors='ignore') if response else ""

    def torque_off(self, servo_id=255):
        """
        释放扭力/卸载 (表格序号 5)
        卸载后可以用手掰动机械臂。默认 ID=255 广播全体卸载。
        """
        command = f"#{servo_id:03d}PULK!"
        self.send_command(command)

    def torque_on(self, servo_id=255):
        command = f"#{servo_id:03d}PULR!"
        self.send_command(command)

    def zero_set(self, servo_id=255):
        command = f"#{servo_id:03d}PSCK!"
        self.send_command(command)

    def reset(self):
        self.send_command("#255PCLE!")
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

        for i in range(self.num_joints):
            servo_id = i + 1

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

    def map_angle_piper(self, val):
        # 长度校验
        lengths = [len(val), len(self.in_min), len(self.in_max), len(self.out_min), len(self.out_max)]
        assert all(l == lengths[0] for l in lengths), "错误：传入的关节数据列表长度不一致！"

        mapped_val = [] # 初始化空列表

        for i in range(lengths[0]):
            # 1. 核心映射公式计算出原始浮点值
            raw_val = (val[i] - self.in_min[i]) * (self.out_max[i] - self.out_min[i]) / (self.in_max[i] - self.in_min[i]) + self.out_min[i]

            # 2. 针对当前关节计算硬件安全边界 (必须在循环内计算)
            current_min_limit = min(self.out_min[i], self.out_max[i])
            current_max_limit = max(self.out_min[i], self.out_max[i])

            # 3. 硬件安全限幅，取整，并追加到结果列表中
            safe_val = int(max(current_min_limit, min(raw_val, current_max_limit)))
            mapped_val.append(safe_val)

        return mapped_val


if __name__ == "__main__":
    servo = Servo()
    runing=True
    while runing:
        print(servo.read_all_angles())
        time.sleep(1)