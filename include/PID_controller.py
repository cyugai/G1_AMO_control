import numpy as np
import mujoco.viewer
from dataclasses import dataclass, field
# ====== 向量化PID控制器 ======
class MatrixPIDController:
    def __init__(self, Kp, Kd, Ki, filter_coefficient=100.0):
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.integral = np.zeros(Kp.shape[0])
        # --- 滤波微分项初始化 ---
        self.filter_coefficient = filter_coefficient
        self.prev_error = np.zeros(Kp.shape[0])
        self.prev_derivative = np.zeros(Kp.shape[0])

    def update(self, error, d_error, dt):
        self.integral += error * dt
         # --- 一阶滤波微分（矢量化实现）---
        alpha = self.filter_coefficient
        # 支持传入标量或每个关节一个系数
        if np.isscalar(alpha):
            alpha = np.full(error.shape, alpha)
        derivative = (alpha * (error - self.prev_error) + self.prev_derivative) / (alpha * dt + 1.0)
        self.prev_error = error.copy()
        self.prev_derivative = derivative.copy()
        return self.Kp @ error + self.Kd @ d_error + self.Ki @ self.integral
    



def create_biarm_pid_controllers():
    """
    创建位置的PID控制器及其增益参数。
    返回 pos_pid_controller
    """
    # ---- 位置PID参数 ----
    n_joints = 14
    Kp = np.diag([
                  200, 200, 90,  90,  90, 75, 75,
                  200, 200, 90,  90,  90, 75, 75])
    Kd = np.diag([  
                  12,  18,  10,  10, 12, 8, 8,
                  12,  18,  10,  10, 12, 8, 8])
    Ki = np.zeros((n_joints, n_joints))
    pos_pid_controller = MatrixPIDController(Kp, Kd, Ki)

    return pos_pid_controller

def create_homie_biarm_pid_controllers():
    """
    创建位置PID控制器及其增益参数。
    返回 pos_pid_controller
    """
    n_joints = 15
    # ---- 位置PID参数 ----
    # Kp = np.diag([100,
    #               80, 80, 40,  60,  36, 30, 30,
    #               80, 80, 40,  60,  36, 30, 30])
   
    # Kd = np.diag([0.5,
    #               0, 0, 0, 0, 0, 0, 0,
    #               0, 0, 0, 0, 0, 0, 0])
    
    Kp = np.diag([200,
                  200, 200, 90,  90,  90, 75, 75,
                  200, 200, 90,  90,  90, 75, 75])
    Kd = np.diag([12,  
                  12,  18,  10,  10, 12, 8, 8,
                  12,  18,  10,  10, 12, 8, 8])*0.5
    Ki = np.zeros((n_joints, n_joints))
    pos_pid_controller = MatrixPIDController(Kp, Kd, Ki)

    return pos_pid_controller

def create_homie_cascade_biarm_pid_controllers():
    """
    创建位置和速度的PID控制器及其增益参数。
    返回 pos_pid_controller, vel_pid_controller
    """
    n_joints = 15
    # ---- 位置PID参数 ----
    Kp = np.diag([100,
                  80, 80, 40,  60,  36, 30, 30,
                  80, 80, 40,  60,  36, 30, 30])
    # Kd = np.diag([5,
    #               2,  2,  1,  2, 2, 2, 2,
    #               2,  2,  1,  2, 2, 2, 2])
    Kd = np.diag([0.5,
                  0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0])
    Ki = np.zeros((n_joints, n_joints))
    pos_pid_controller = MatrixPIDController(Kp, Kd, Ki)
    
    # ---- 速度PID参数 ----
    vel_Kp = np.diag([10, 
                      8, 8, 12, 12, 10, 12, 12,
                      8, 8, 12, 12, 10, 12, 12])
    vel_Kd = np.diag([0,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0])
    vel_Ki = np.zeros((n_joints, n_joints))
    vel_pid_controller = MatrixPIDController(vel_Kp, vel_Kd, vel_Ki)
    
    return pos_pid_controller, vel_pid_controller

def compute_upper_body_control_torque(
    mujoco_model,
    mujoco_data,
    dof_ids,
    pos_pid_controller,
    qd,                 # 期望关节位置
    desired_dq,         # 期望关节速度（仅用于位置PID的微分项）
    desired_ddq,        # 期望关节加速度（前馈）
    max_torque,         # 力矩上限
    dt
):
    """
    计算控制力矩（前馈 + 位置 PID），返回 tau 与 acc_cmd（位置 PID 输出）
    """
    # ------- 当前关节状态 -------
    qpos = mujoco_data.qpos[dof_ids + 6].copy()
    qvel = mujoco_data.qvel[dof_ids + 5].copy()

    # ------- 位置 PID：直接给出加速度修正 -------
    error   = qd - qpos                 # 位置误差
    d_error = desired_dq - qvel         # 速度误差（微分项）
    acc_cmd = pos_pid_controller.update(error, d_error, dt)  # rad/s²

    # ------- 动力学补偿（逆动力学）-------
    qacc_bak = mujoco_data.qacc.copy()        # 备份
    mujoco_data.qacc[:] = 0.0
    mujoco_data.qacc[dof_ids + 5] = acc_cmd + desired_ddq
    mujoco.mj_inverse(mujoco_model, mujoco_data)            # 求逆动力学
    tau = mujoco_data.qfrc_inverse[dof_ids + 5].copy()
    mujoco_data.qacc[:] = qacc_bak            # 恢复

    # ------- 饱和 + 抗积分饱和 -------
    tau_clipped = np.clip(tau, -max_torque, max_torque)
    if not np.allclose(tau, tau_clipped):
        # 只衰减位置 PID 的积分项
        pos_pid_controller.integral *= 0.95
    tau = tau_clipped

    # 如无需要，可改成只 return tau
    return tau

def compute_homie_control_torque(
    mujoco_model,
    mujoco_data,
    dof_ids,
    pos_pid_controller,
    qd,                 # 期望关节位置
    desired_dq,         # 期望关节速度
    desired_ddq,        # 期望关节加速度
    max_torque,         # 力矩上限
    dt
):
    """
    计算控制力矩（前馈+PID），返回tau与dq_desired_pid
    """
    # ------- 获取当前关节状态 -------
    qpos = mujoco_data.qpos[dof_ids + 6].copy()
    qvel = mujoco_data.qvel[dof_ids + 5].copy()
    
    # ------- 向量化PID反馈 -------
    # 1. position PID 控制器
    error = qd - qpos
    d_error = desired_dq - qvel
    acc_cmd = pos_pid_controller.update(error, d_error, dt)

    # 2. 求逆动力学：把期望加速度写进 data.qacc，再调 mj_inverse
    qacc_bak = mujoco_data.qacc.copy()         # 备份
    mujoco_data.qacc[:] = 0.0
    mujoco_data.qacc[dof_ids + 5] = acc_cmd + desired_ddq
    mujoco.mj_inverse(mujoco_model, mujoco_data)  # 填充 qfrc_inverse
    tau = mujoco_data.qfrc_inverse[dof_ids + 5].copy()
    mujoco_data.qacc[:] = qacc_bak              # 恢复，防止污染

    # 3. 极简 anti-wind-up／饱和处理
    tau_clipped = np.clip(tau, -max_torque, max_torque)
    if not np.allclose(tau, tau_clipped):
        pos_pid_controller.integral *= 0.95
    tau = tau_clipped

    return tau

def compute_homie_cascade_control_torque(
    mujoco_model,
    mujoco_data,
    pos_pid_controller,
    vel_pid_controller,
    qd,                 # 期望关节位置
    desired_dq,         # 期望关节速度
    desired_ddq,        # 期望关节加速度
    max_torque,         # 力矩上限
    dt
):
    """
    计算控制力矩（前馈+PID），返回tau与dq_desired_pid
    """
    # ------- 获取当前关节状态 -------
    upper_body_idx = np.array([0,
                    1,2,3,4,5,6,7,
                    14,15,16,17,18,19,20])
    qpos = mujoco_data.qpos[19+upper_body_idx].copy()
    qvel = mujoco_data.qvel[18+upper_body_idx].copy()
    qacc = mujoco_data.qacc[18+upper_body_idx].copy()
    
    # ------- 向量化PID反馈 -------
    # 1. position PID 控制器
    error = qd - qpos
    d_error = desired_dq - qvel
    dq_desired_pid = pos_pid_controller.update(error, d_error, dt)

    # 2. 速度 PID 控制器  --> 期望“关节角加速度”
    vel_error = dq_desired_pid - qvel
    d_vel_error = desired_ddq - qacc
    acc_cmd = vel_pid_controller.update(vel_error, d_vel_error, dt)  # rad/s²

    # 3. 求逆动力学：把期望加速度写进 data.qacc，再调 mj_inverse
    qacc_bak = mujoco_data.qacc.copy()         # 备份
    mujoco_data.qacc[:] = 0.0
    mujoco_data.qacc[18+upper_body_idx] = acc_cmd + desired_ddq
    mujoco.mj_inverse(mujoco_model, mujoco_data)  # 填充 qfrc_inverse
    tau = mujoco_data.qfrc_inverse[18+upper_body_idx].copy()
    mujoco_data.qacc[:] = qacc_bak              # 恢复，防止污染

    # 4. 极简 anti-wind-up／饱和处理
    tau_clipped = np.clip(tau, -max_torque, max_torque)
    if not np.allclose(tau, tau_clipped):
        pos_pid_controller.integral *= 0.95
        vel_pid_controller.integral *= 0.95
    tau = tau_clipped

    return tau, dq_desired_pid


def gripper_pid_combine(mujoco_data,target_force, left_idx, right_idx,
                left_follower_idx, right_follower_idx,
                dt, 
                ):
    """
    Gripper PID control to maintain a target force.
    """
    # 获取传感器数据
    left_gripper_left_force = mujoco_data.sensordata[left_idx]
    left_gripper_right_force = mujoco_data.sensordata[right_idx]
    left_gripper_left_follower_force = mujoco_data.sensordata[left_follower_idx]
    left_gripper_right_follower_force = mujoco_data.sensordata[right_follower_idx]
    # 计算平均力
    avg_force = (left_gripper_left_force + left_gripper_right_force +
                 left_gripper_left_follower_force + left_gripper_right_follower_force) / 2.0

    KP=0.25
    KI=0
    KD=0
    # PID
    err = target_force - avg_force
    integral = err * dt
    deriv = (err - (target_force - avg_force)) / dt
    u = KP * err + KI * integral + KD * deriv
    
    return u
