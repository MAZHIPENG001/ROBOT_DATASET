import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_quaternion(rotation_eular):
    # 欧拉角（弧度）->四元数asdfasdf
    roll, pitch, yaw = rotation_eular[0],rotation_eular[1],rotation_eular[2]
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    rotation_quat = [x, y, z, w]
    return rotation_quat

def transform_matrix(translation, rotation_quat=None, rotation_matrix=None,rotation_euler=None,sequence='xyz'):
    """
    功能：创建齐次变换矩阵T=[[R11,R12,R13,X],
                        [R21,R22,R23,Y],
                        [R31,R32,R33,Z],
                        [0, 0,  0,  1]]
    输入：   1.位置偏移XYZ
            2.旋转矩阵(optional)
            3.四元数(optional)
            4.欧拉角(optional)
    """
    T = np.eye(4)
    T[:3, 3] = translation
    if rotation_euler is not None:
        rotation_quat = euler_to_quaternion(rotation_euler)
    if rotation_quat is not None:
        rotation = R.from_quat(rotation_quat)
        T[:3, :3] = rotation.as_matrix()
    elif rotation_matrix is not None:
        T[:3, :3] = rotation_matrix
    return T

def T_6dpose(ap):
    '''将齐次变换矩阵转换为X、Y、Z、RX、RY、RZ'''
    # 提取平移部分 (单位: 米)
    X_m = ap[0, 3]
    Y_m = ap[1, 3]
    Z_m = ap[2, 3]

    # 提取旋转矩阵并转换为欧拉角
    rotation_matrix = ap[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('xyz', degrees=False)  # 弧度

    # 转换为度
    RX_rad = euler_angles[0]
    RY_rad = euler_angles[1]
    RZ_rad = euler_angles[2]

    # 转换为机械臂需要的单位 (X,Y,Z: 0.001mm, RX,RY,RZ: 0.001度)
    X_target = round(X_m * 1000 * 1000)  # 米 -> 毫米 -> 0.001毫米
    Y_target = round(Y_m * 1000 * 1000)
    Z_target = round(Z_m * 1000 * 1000)
    RX_target = round(np.rad2deg(RX_rad) * 1000)  # 弧度 -> 度 -> 0.001度
    RY_target = round(np.rad2deg(RY_rad) * 1000)
    RZ_target = round(np.rad2deg(RZ_rad) * 1000)
    return X_target,Y_target,Z_target,RX_target,RY_target,RZ_target

"""
xyz顺序
    绕固定轴旋转：
        R_fixed = R_z(γ) @ R_y(β) @ R_x(α)  # 从右到左
    绕当前轴旋转：
        R_current = R_x(α) @ R_y(β) @ R_z(γ)  # 从左到右
"""
def rotation_x(theta,degree=True):
    theta=np.deg2rad(theta) if degree else theta
    """绕X轴旋转矩阵"""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
def rotation_y(theta,degree=True):
    theta = np.deg2rad(theta) if degree else theta
    """绕Y轴旋转矩阵"""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
def rotation_z(theta,degree=True):
    theta = np.deg2rad(theta) if degree else theta
    """绕Z轴旋转矩阵"""
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])