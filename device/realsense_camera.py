import pyrealsense2 as rs
import numpy as np
import cv2
from processer.transfer import transform_matrix
from device.keyboard import KeystrokeCounter,KeyCode,Key

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30,serial_number=None):
        """
        初始化RealSense相机
        参数:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_number = serial_number
        # 创建管道和配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # 如果有指定序列号，则只连接该设备
        if serial_number:
            self.config.enable_device(serial_number)
        # 配置流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 对齐工具（将深度图对齐到彩色图）
        self.align = rs.align(rs.stream.color)

        # 深度颜色化工具
        self.colorizer = rs.colorizer()

        self.profile = self.pipeline.start(self.config)

        # color_frame, depth_frame = self.get_frames()
        # self.depth_intrin = depth_frame.profile.as_video_stream_profile().get_intrinsics()

        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.depth_intrinsics = self.depth_profile.get_intrinsics()
        self.color_intrinsics = self.color_profile.get_intrinsics()
        # 标定
        # 眼在手上：T1
        x = -0.06793272468909138
        y = 0.0472148510076286
        z = 0.05432937007996423
        q = [0.019399398882275886, 0.038359533848541405, -0.7361777959688619, 0.6754216921363702]
        self.T1 = transform_matrix([x, y, z], rotation_quat=q)

    def start(self):
        """启动相机"""
        try:
            # self.profile = self.pipeline.start(self.config)
            print(f"\033[92m相机启动成功\033[0m")

            # 获取深度传感器和深度标尺
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"\033[96m深度标尺: {self.depth_scale}\033[0m")

            c_fx, c_fy, c_cx, c_cy = self.get_intrinsics()
            d_fx, d_fy, d_cx, d_cy = self.get_depth_intrinsics()

            self.camera_config = {
                'intrinsics': {
                    'color': {
                        'fx': c_fx,
                        'fy': c_fy,
                        'ppx': c_cx,
                        'ppy': c_cy,
                    },
                    'depth': {
                        'fx': d_fx,
                        'fy': d_fy,
                        'ppx': d_cx,
                        'ppy': d_cy,

                    },
                    'depth_scale': self.depth_scale
                }
            }

        except Exception as e:
            print(f"\033[91m相机启动失败: {e}\033[0m")
            return False
        return True
    def __enter__(self):
        self.start()
        return self

    def get_frames(self):
        """
        获取对齐后的帧
        返回:
            color_frame: 彩色帧
            depth_frame: 深度帧
            aligned_depth_frame: 对齐后的深度帧
        """
        try:
            # 等待一组连贯的帧
            frames = self.pipeline.wait_for_frames()

            # 将对齐深度帧到彩色帧
            aligned_frames = self.align.process(frames)

            # 获取对齐后的深度帧和彩色帧
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                return None, None

            return color_frame, depth_frame

        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None

    def get_images(self):
        """
        获取对齐后的图像数组
        返回:
            color_image: 彩色图像 (BGR格式)
            depth_image: 深度图像 (16位)
            colored_depth: 彩色化的深度图像
        """
        color_frame, depth_frame = self.get_frames()

        if color_frame is None or depth_frame is None:
            return None, None

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        # depth_image = depth_frame
        depth_image = np.asanyarray(depth_frame.get_data())
        #
        # # 彩色化深度图
        # colored_depth = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        return color_image, depth_image

    def get_real_position(self, u, v):
        """
        获取指定像素点的深度值（米）
        参数:
            x, y: 像素坐标
        返回:
            x:米
            y:米
            深度值（米），如果无效返回0
        """
        _, depth_frame= self.get_frames()
        if depth_frame is None:
            return 0

        depth = depth_frame.get_distance(u, v)
        if depth >0:
            camera_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], depth)
            print(
                f"\033[1;93m像素: ({u}, {v}) -> 真实坐标 (米): X={camera_coordinate[0]:.3f}, Y={camera_coordinate[1]:.3f}, Z={camera_coordinate[2]:.3f}\033[0m")
            return camera_coordinate[0], camera_coordinate[1],camera_coordinate[2]
        else:
            print("深度无效，depth：",depth)
            return None,None,None

    def get_point_cloud(self, depth_frame=None):
        """
        生成点云数据
        参数:
            depth_frame: 深度帧，如果为None则获取新帧
        返回:
            vertices: 点云顶点数组
        """
        if depth_frame is None:
            _, depth_frame, _ = self.get_frames()
            if depth_frame is None:
                return None

        # 创建点云对象
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)

        return points

    def stop(self):
        """停止相机"""
        self.pipeline.stop()
        print("相机已停止")

    def get_intrinsics(self):
        """获取相机内参"""

        c_fx = self.color_intrinsics.fx
        c_fy = self.color_intrinsics.fy
        c_cx = self.color_intrinsics.ppx
        c_cy = self.color_intrinsics.ppy
        print(f'\033[96m彩图内参:{c_fx}, {c_fy}, {c_cx}, {c_cy}\033[0m')
        return c_fx, c_fy, c_cx, c_cy

    def get_intrinsic_matrix(self):
        c_fx, c_fy, c_cx, c_cy = self.get_intrinsics()
        intrinsic_matrix = np.array([[c_fx, 0, c_cx],
                                     [0, c_fy, c_cy],
                                     [0, 0, 1]])
        return intrinsic_matrix

    def get_depth_intrinsics(self):
        d_fx = self.depth_intrinsics.fx
        d_fy = self.depth_intrinsics.fy
        d_cx = self.depth_intrinsics.ppx
        d_cy = self.depth_intrinsics.ppy
        print(f'\033[96m彩图内参深度图内参:{d_fx}, {d_fy}, {d_cx}, {d_cy}')
        return d_fx, d_fy, d_cx, d_cy

    def display_images(self):
        """实时显示彩色和深度图像"""
        try:
            while True:
                color_image, depth_image = self.get_images()

                if color_image is None or depth_image is None:
                    continue

                # 显示图像
                cv2.imshow('RealSense - Color', color_image)
                cv2.imshow('RealSense - Depth', depth_image)

                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    def save_images(self, save_path):
        color_image, _= self.get_images()
        cv2.imwrite(save_path, color_image)

def serial_number():
    ctx = rs.context()
    devices = ctx.query_devices()

    print("检测到的设备:")
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"\33[92m设备 {i}: {name}, 序列号: {serial}\33[0m")


def list_camera_framerates(serial_number=None):
    """
    列出相机支持的分辨率和帧率组合
    """
    ctx = rs.context()

    # 获取设备
    if serial_number:
        devices = ctx.query_devices()
        for dev in devices:
            if dev.get_info(rs.camera_info.serial_number) == serial_number:
                print(f"设备序列号: {serial_number}")
                list_stream_profiles(dev)
                break
    else:
        devices = ctx.query_devices()
        for i, dev in enumerate(devices):
            print(f"\n设备 {i + 1}:")
            print(f"  名称: {dev.get_info(rs.camera_info.name)}")
            print(f"  序列号: {dev.get_info(rs.camera_info.serial_number)}")
            list_stream_profiles(dev)


def list_stream_profiles(device):
    """
    列出设备的流配置
    """
    # 获取所有传感器
    sensors = device.query_sensors()

    for sensor in sensors:
        print(f"  \n传感器类型: {sensor.get_info(rs.camera_info.name)}")

        # 获取传感器支持的所有流配置
        stream_profiles = sensor.get_stream_profiles()

        # 按流类型和分辨率分组
        profiles_by_res = {}

        for profile in stream_profiles:
            # 将profile转换为视频流profile
            if profile.is_video_stream_profile():
                vprofile = profile.as_video_stream_profile()

                # 获取流类型
                stream_type = str(vprofile.stream_type())

                # 获取分辨率
                width = vprofile.width()
                height = vprofile.height()
                res_key = f"{stream_type}_{width}x{height}"

                if res_key not in profiles_by_res:
                    profiles_by_res[res_key] = []

                # 获取帧率
                fps = vprofile.fps()
                if fps not in profiles_by_res[res_key]:
                    profiles_by_res[res_key].append(fps)

        # 打印结果
        for res_key, fps_list in profiles_by_res.items():
            stream_type, resolution = res_key.split('_')
            fps_list.sort()
            print(f"    流类型: {stream_type:<15} 分辨率: {resolution:<10} 支持帧率: {fps_list}")
if __name__ == "__main__":
    camera=RealSenseCamera(serial_number="233622070932")#135122074270
    serial_number()
    list_camera_framerates()
    savepath='./photo.jpg'
    if camera.start():
        fx,fy,cx,cy=camera.get_intrinsics()
        # # 显示实时图像
        # camera.display_images()
    while True:
        with KeystrokeCounter() as key_counter:
            try:
                while True:
                    color_image, depth_image = camera.get_images()
                    if color_image is None or depth_image is None:
                        continue
                    # 显示图像
                    cv2.imshow('RealSense - Color', color_image)
                    cv2.imshow('RealSense - Depth', depth_image)
                    # 按'q'退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    press_events = key_counter.get_press_events()
                    for key_stroke in press_events:
                        # 按下o：归零(关节控制)
                        if key_stroke == KeyCode(char='s'):
                            print("按下s：保存图片")
                            camera.save_images(savepath)
                        # 按下i: 复位(关节控制)
                        elif key_stroke == Key.backspace:
                            print("按下backspace:退出")
                            exit()
            except KeyboardInterrupt:
                events = key_counter.get_press_events()
                print(events)
