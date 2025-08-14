
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation
def mujoco_quat_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])

def xyzw_to_mujoco_quat(q):
    return np.array([q[3], q[0], q[1], q[2]])

def world_frame_to_pelvis_frame(model, data, position, orientation):
    """
    Transform a point's position and orientation from world frame to pelvis frame.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        position: (3,) array, position in world frame.
        orientation: (4,) array, quaternion (w, x, y, z) in world frame.

    Returns:
        rel_position: (3,) array, position in pelvis frame.
        rel_orientation: (4,) array, quaternion (w, x, y, z) in pelvis frame.
    """
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_pos = data.xpos[pelvis_id]     # (3,)
    pelvis_quat = data.xquat[pelvis_id]   # (4,)
    # norm = np.linalg.norm(pelvis_quat)
    # if norm < 1e-8:
    #     print(f"Warning: pelvis quaternion zero norm, replacing with identity quaternion.")
    #     pelvis_quat = np.array([1, 0, 0, 0])

    # Position
    pelvis_rot = Rotation.from_quat(mujoco_quat_to_xyzw(pelvis_quat))
    point_vec = position - pelvis_pos
    rel_position = pelvis_rot.inv().apply(point_vec)

    # Orientation
    point_rot = Rotation.from_quat(mujoco_quat_to_xyzw(orientation))
    rel_rot = pelvis_rot.inv() * point_rot
    rel_orientation = xyzw_to_mujoco_quat(rel_rot.as_quat())

    return rel_position, rel_orientation

def world_frame_to_reduced_world_frame(model, data, position, orientation):
    """
    Transform a point's position and orientation from world frame to reduced world frame.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        position: (3,) array, position in world frame.
        orientation: (4,) array, quaternion (w, x, y, z) in world frame.

    Returns:
        rel_position: (3,) array, position in reduced world frame.
        rel_orientation: (4,) array, quaternion (w, x, y, z) in reduced world frame.
    """
    base_height_in_reduced_world = 0.793
    rel_position, rel_orientation = world_frame_to_pelvis_frame(model, data, position, orientation)
    # Position
    rel_position[2] += base_height_in_reduced_world  # Adjust height to reduced world frame

    # # Orientation
    # rel_orientation = orientation  # No change in orientation

    return rel_position, rel_orientation


def contact_world_to_pelvis(model, data, contact_index):
    """
    把 mjData.contact[contact_index] 的接触点和接触框架
    从 world frame 变换到 body_name（默认 pelvis）坐标系下。

    Returns
    -------
    rel_pos : (3,) np.ndarray
        接触点在 pelvis 坐标系下的位置
    rel_quat : (4,) np.ndarray
        接触框架在 pelvis 坐标系下的四元数 (w, x, y, z)
    """
    body_name="torso_link"
    # ---------- 1. 提取接触信息（世界系） ----------
    c = data.contact[contact_index]
    p_cw = c.pos                         # (3,) 接触点世界坐标
    R_cw = c.frame.reshape(3, 3)         # (3,3) 接触框架，行主

    # ---------- 2. 计算 world -> pelvis 旋转 ----------
    body_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    p_pw      = data.xpos[body_id]                       # pelvis 世界位置
    R_pw      = data.xmat[body_id].reshape(3, 3)         # pelvis 世界方向
    R_wp      = R_pw.T                                   # world -> pelvis

    # ---------- 3. 位置与方向变换 ----------
    rel_pos = R_wp @ (p_cw - p_pw)                       # (3,)
    R_cp    = R_wp @ R_cw.T                               # (3,3)

    # # 转为四元数 (SciPy 给 xyzw，要换成 MuJoCo 的 wxyz)
    # quat_xyzw = Rotation.from_matrix(R_cp).as_quat()
    # rel_quat  = np.concatenate([[quat_xyzw[3]], quat_xyzw[:3]])

    return R_cp, R_wp

def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec

def euler_xyz_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])

def quat_mul(q1, q2):
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def apply_wrist_rotation(q_current, roll, pitch, yaw):
    q_delta = euler_xyz_to_quat(roll, pitch, yaw)
    q_new   = quat_mul(q_current, q_delta)     # 末端自身坐标系
    return q_new / np.linalg.norm(q_new)
