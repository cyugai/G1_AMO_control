# -----------------------------------------------------------------------------
# Copyright [2025] [Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is based on an initial draft generously provided by Zixuan Chen.
# -----------------------------------------------------------------------------

import types
import numpy as np
import mujoco, mujoco_viewer
import glfw
from collections import deque
import torch
import mink
import pygame
import time
import math
import asyncio
import threading
from include.PID_controller import *
from include.frame_transform import *
from scipy.spatial.transform import Rotation as R
from XLeVR.vr_monitor import VRMonitor

def vr_to_world(pos):
    x = (pos[2]) * -1.0   # vr_scale_x
    y = (pos[0]) * -1.0  # vr_scale_y
    z = (pos[1]) * 1.0  # vr_scale_z
    return np.array([x, y, z], dtype=np.float32)


multiplier = 5
out = False
def _key_callback(self, window, key, scancode, action, mods):
    global multiplier
    global out
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_S: 
        self.commands[0] -= 0.05 
    elif key == glfw.KEY_W:
        self.commands[0] += 0.05
    elif key == glfw.KEY_A: 
        self.commands[1] += 0.1
    elif key == glfw.KEY_D: 
        self.commands[1] -= 0.1
    elif key == glfw.KEY_Q: 
        self.commands[2] += 0.05
    elif key == glfw.KEY_E:
        self.commands[2] -= 0.05
    elif key == glfw.KEY_Z: 
        self.commands[3] += 0.05 
    elif key == glfw.KEY_X:
        self.commands[3] -= 0.05
    elif key == glfw.KEY_J: 
        self.commands[4] += 0.1
    elif key == glfw.KEY_U:
        self.commands[4] -= 0.1
    elif key == glfw.KEY_K: 
        self.commands[5] += 0.05
    elif key == glfw.KEY_I:
        self.commands[5] -= 0.05
    elif key == glfw.KEY_L: 
        self.commands[6] += 0.05
    elif key == glfw.KEY_O:
        self.commands[6] -= 0.1
    elif key == glfw.KEY_T:
        multiplier *= -1
        print("multiplier changed to", multiplier)
    elif key == glfw.KEY_1:
        self.commands[7] += 0.01 * multiplier
    elif key == glfw.KEY_2:
        self.commands[8] += 0.01 * multiplier
    elif key == glfw.KEY_3:
        self.commands[9] += 0.01 * multiplier
    elif key == glfw.KEY_4:
        self.commands[10] += 0.01 * multiplier
    elif key == glfw.KEY_5:
        self.commands[11] += 0.01 * multiplier
    elif key == glfw.KEY_6:
        self.commands[12] += 0.01 * multiplier
    elif key == glfw.KEY_7:
        self.commands[13] += 0.01 * multiplier
    elif key == glfw.KEY_8:
        self.commands[14] += 0.01 * multiplier
    elif key == glfw.KEY_9:
        self.commands[15] += 0.01 * multiplier
    elif key == glfw.KEY_0:
        self.commands[16] += 0.01 * multiplier
    elif key == glfw.KEY_MINUS:
        self.commands[17] += 0.01 * multiplier
    elif key == glfw.KEY_EQUAL:
        self.commands[18] += 0.01 * multiplier
    elif key == glfw.KEY_LEFT_BRACKET:
        self.commands[19] += 0.01 * multiplier
    elif key == glfw.KEY_RIGHT_BRACKET:
        self.commands[20] += 0.01 * multiplier
    elif key == glfw.KEY_SPACE:
        out = not out
        
    elif key == glfw.KEY_ESCAPE:
        print("Pressed ESC")
        print("Quitting.")
        glfw.set_window_should_close(self.window, True)
        return
    print(
        f"vx: {self.commands[0]:<{8}.2f}"
        f"vy: {self.commands[2]:<{8}.2f}"
        f"yaw: {self.commands[1]:<{8}.2f}"
        f"height: {(0.75 + self.commands[3]):<{8}.2f}"
        f"torso yaw: {self.commands[4]:<{8}.2f}"
        f"torso pitch: {self.commands[5]:<{8}.2f}"
        f"torso roll: {self.commands[6]:<{8}.2f}"
    )

class HumanoidEnv:
    def __init__(self, policy_jit, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device
        
        if robot_type == "g1":

            model_path = "assets/scene_g1_29_VR_amo.xml"
            reduced_upper_model_path = "assets/g1_29_2f85_reduced.xml"

            self.stiffness = np.array([
                150, 150, 150, 300, 80, 20,
                150, 150, 150, 300, 80, 20,
                400, 400, 400,
                80, 80, 40, 60, 60, 60, 60,
                80, 80, 40, 60, 60, 60, 60
            ])
            self.damping = np.array([
                2, 2, 2, 4, 2, 1,
                2, 2, 2, 4, 2, 1,
                15, 15, 15,
                2, 2, 1, 1, 1, 1 ,1,
                2, 2, 1, 1, 1, 1, 1
            ])
            self.num_actions = 15
            self.num_dofs = 23
            self.default_dof_pos = np.array([
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0, 0.0, 0, 0,
                0, 0.0, -0, 0,
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25, 25, 25, 25,
                25, 25, 25, 25, 25, 25, 25
            ])
            # Arm joint limits for 14 DoFs (left arm first, then right arm)
            self.arm_dof_lower_range = np.array([
                -3.0892,  # left_shoulder_pitch_joint
                -1.5882,  # left_shoulder_roll_joint
                -2.618,   # left_shoulder_yaw_joint
                -1.0472,  # left_elbow_joint
                -1.97222, # left_wrist_roll_joint
                -1.61443, # left_wrist_pitch_joint
                -1.61443, # left_wrist_yaw_joint
                -3.0892,  # right_shoulder_pitch_joint
                -2.2515,  # right_shoulder_roll_joint
                -2.618,   # right_shoulder_yaw_joint
                -1.0472,  # right_elbow_joint
                -1.97222, # right_wrist_roll_joint
                -1.61443, # right_wrist_pitch_joint
                -1.61443  # right_wrist_yaw_joint
            ])
            self.arm_dof_upper_range = np.array([
                2.6704,   # left_shoulder_pitch_joint
                2.2515,   # left_shoulder_roll_joint
                2.618,    # left_shoulder_yaw_joint
                2.0944,   # left_elbow_joint
                1.97222,  # left_wrist_roll_joint
                1.61443,  # left_wrist_pitch_joint
                1.61443,  # left_wrist_yaw_joint
                2.6704,   # right_shoulder_pitch_joint
                1.5882,   # right_shoulder_roll_joint
                2.618,    # right_shoulder_yaw_joint
                2.0944,   # right_elbow_joint
                1.97222,  # right_wrist_roll_joint
                1.61443,  # right_wrist_pitch_joint
                1.61443   # right_wrist_yaw_joint
            ])
            # 6*2+3+4*2= 23
            self.dof_names = [
                # waist joints.
                # "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                # Left arm joints.
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                # Right arm joints.
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ]

            self.upper_dof_names = [
                # waist joints.
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                # Left arm joints.
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                # Right arm joints.
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ]

            # Add 12 leg joints before the waist joints
            self.whole_body_dof_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ]

        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.obs_indices = np.arange(self.num_dofs)
        
        self.sim_duration = 100 * 20.0
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        # ----- reduce the upper body model -----
        self.upper_model = mujoco.MjModel.from_xml_path(reduced_upper_model_path)
        self.upper_model.opt.timestep = self.sim_dt
        self.upper_data = mujoco.MjData(self.upper_model)

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, hide_menus=True)
        self.viewer.commands = np.zeros(21, dtype=np.float32) #7 origin commands + 14 arm commands 
        self.viewer.cam.distance = 2.5 
        self.viewer.cam.azimuth = -130
        self.viewer.cam.elevation = -35
        self.viewer.cam.lookat = np.array([0.0, 0.0, 0.6], dtype=np.float32)
        self.viewer._key_callback = types.MethodType(_key_callback, self.viewer)
        glfw.set_key_callback(self.viewer.window, self.viewer._key_callback)
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25
        self.arm_action = self.default_dof_pos[15:]
        self.prev_arm_action = self.default_dof_pos[15:]
        self.arm_blend = 0.0
        self.toggle_arm = False

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05
        
        self.nj = 23 # 23+7+6=36
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15 
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8
        
        self.dof_pos = np.zeros(self.nj, dtype=np.float32)
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3, ))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75

        self.target_yaw = 0.0 

        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for i in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))
    
        self.policy_jit = policy_jit

        self.adapter = torch.jit.load("amo/adapter_jit.pt", map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False
        
        norm_stats = torch.load("amo/adapter_norm_stats.pt")
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32) # to ues the old model
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)

        # ---- mink IK初始化 ----
        self.configuration = mink.Configuration(self.upper_model)
        pose_cost = np.ones((self.upper_model.nv)) * 0.1
        pose_cost[:3] = 10  # constraints on waist
        self.posture_task = mink.PostureTask(self.upper_model, cost=pose_cost, lm_damping=1.0)
        base_cost = np.zeros((self.upper_model.nv))
        base_cost[:3] = 100  # Exclude root position and orientation and legs from the cost
        self.damping_task = mink.DampingTask(self.upper_model, base_cost)
        self.tasks = [
            self.posture_task,
            self.damping_task,
            ]
        self.hands = ["Left_gripper_center", "Right_gripper_center"]
        self.hand_tasks = [
            mink.FrameTask(
                frame_name="Left_gripper_center",
                frame_type="site",
                position_cost=5.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
            mink.FrameTask(
                frame_name="Right_gripper_center",
                frame_type="site",
                position_cost=5.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            ),
        ]
        self.tasks.extend(self.hand_tasks)
        self.collision_pairs = [
            
            (["left_arm_hand_capsule", "right_arm_hand_capsule",
            "left_elbow_stl", "right_elbow_stl",
            "left_wrist_roll_stl", "right_wrist_roll_stl",
                "left_wrist_pitch_stl", "right_wrist_pitch_stl",
                "left_wrist_yaw_stl", "right_wrist_yaw_stl"
            ], ["torso_stl"]),
            (["left_arm_hand_capsule"], ["right_arm_hand_capsule"]),
           
            
        ]
        self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.upper_model,
            geom_pairs=self.collision_pairs, 
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )

        self.limits = [
            mink.ConfigurationLimit(self.upper_model),
            self.collision_avoidance_limit,
        ]

        self.left_gripper_center_id = self.model.site("Left_gripper_center").id
        self.right_gripper_center_id = self.model.site("Right_gripper_center").id
        self.dof_ids = np.array([self.model.joint(name).id for name in self.dof_names])
        self.upper_dof_ids = np.array([self.model.joint(name).id for name in self.upper_dof_names])
        self.whole_dof_ids = np.array([self.model.joint(name).id for name in self.whole_body_dof_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.whole_body_dof_names])
        self.qpos = self.data.qpos[self.dof_ids].copy()  # Exclude the root position and orientation
        self.mink_init = False
        self.configuration.update_from_keyframe("home")
        self.posture_task.set_target_from_configuration(self.configuration)

        # ---- PID增益参数和控制器 ----
        self.pos_pid_controller = create_biarm_pid_controllers()
        self.max_torque = np.array([          # 你的 XML 里 motor forcerange 如有不同请自己改
                        25, 25, 25, 25,  25,  25,  25,
                        25, 25, 25, 25,  25,  25,  25          
                          ])

        self.viewer.commands[5] = -0.2 # to balance the weight of the gripper
        

        # --- For gripper force control ---
        self.left_arm_left_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_left_pad_touch")
        ]
        self.left_arm_right_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_right_pad_touch")
        ]
        self.right_arm_left_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_left_pad_touch")
        ]
        self.right_arm_right_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_right_pad_touch")
        ]
        self.left_gripper_left_follower_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_left_follower_touch")
        ]
        self.left_gripper_right_follower_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_right_follower_touch")
        ]
        self.right_gripper_left_follower_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_left_follower_touch")
        ]
        self.right_gripper_right_follower_idx = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_right_follower_touch")
        ]
        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0
        self.left_gripper_pos = 0
        self.right_gripper_pos = 0

        # ---- VR Monitor ----
        self.vr_monitor = VRMonitor()
        self._vr_thread = threading.Thread(
            target=lambda: asyncio.run(self.vr_monitor.start_monitoring()),
            daemon=True)
        self._vr_thread.start()

    def extract_data(self):
        old_indices = np.r_[0:19, 22:26]
        self.dof_pos = self.data.qpos.astype(np.float32)[7+old_indices]
        self.dof_vel = self.data.qvel.astype(np.float32)[6+old_indices]
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        
    def get_observation(self):
        rpy = quatToEuler(self.quat)

        self.target_yaw = self.viewer.commands[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_dof_vel = self.dof_vel.copy()
        # Zero out velocities for specific joints:
        # | Index | Joint Name             | Description |
        # |-------|-----------------------|-------------|
        # |   4   | left_ankle_pitch      |             |
        # |   5   | left_ankle_roll       |             |
        # |  10   | right_ankle_pitch     |             |
        # |  11   | right_ankle_roll      |             |
        # |  13   | waist_roll            |             |
        # |  14   | waist_pitch           |             |
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), self.dof_pos[15:]])

        self.adapter_input[0] = 0.75 + self.viewer.commands[3]
        self.adapter_input[1] = self.viewer.commands[4]
        self.adapter_input[2] = self.viewer.commands[5]
        self.adapter_input[3] = self.viewer.commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)
            
        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
                    self.ang_vel * self.scales_ang_vel,
                    rpy[:2],
                    (np.sin(dyaw),
                    np.cos(dyaw)),
                    (self.dof_pos - self.default_dof_pos),
                    self.dof_vel * self.scales_dof_vel,
                    self.last_action,
                    gait_obs,
                    self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = self.dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.viewer.commands[0]
        obs_demo[self._n_demo_dof+1] = self.viewer.commands[2]
        self._in_place_stand_flag = np.abs(self.viewer.commands[0]) < 0.1 and np.abs(self.viewer.commands[2]) < 0.1
        obs_demo[self._n_demo_dof+3] = self.viewer.commands[4]
        obs_demo[self._n_demo_dof+4] = self.viewer.commands[5]
        obs_demo[self._n_demo_dof+5] = self.viewer.commands[6]
        obs_demo[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75 + self.viewer.commands[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)
        
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))
        
    def run(self): 
        self.left_gripper_center_id = self.model.site("Left_gripper_center").id
        self.right_gripper_center_id = self.model.site("Right_gripper_center").id       
        global out
        initial_left_target = mink.SE3.from_rotation_and_translation(
                                mink.SO3(np.array([0.5, 0.5, 0.5, 0.5])), 
                                np.array([0.38127486, 0.15162665, 0.88822298], dtype=np.float32))
        initial_right_target = mink.SE3.from_rotation_and_translation(
                                mink.SO3(np.array([0.5, 0.5, 0.5, 0.5])), 
                                np.array([0.38127486, -0.15162665, 0.88822298], dtype=np.float32))
        for i in range(int(self.sim_duration / self.sim_dt)):
            self.extract_data()
            left_site_pos = self.data.site_xpos[self.left_gripper_center_id]
            right_site_pos = self.data.site_xpos[self.right_gripper_center_id]
            if out:
                print(f"Left gripper position: {left_site_pos}")
                print(f"Right gripper position: {right_site_pos}")
                out = not out

            if i == 1000: 
                self.mink_init = True
                T_left  = initial_left_target
                T_right = initial_right_target
                print("Mink initialized successfully.")
            
 

            if i > 1000 and self.mink_init:
                dual_goals  = self.vr_monitor.get_latest_goal_nowait()
                left_goal   = dual_goals.get("left")  if dual_goals else None
                right_goal  = dual_goals.get("right") if dual_goals else None
                
                if left_goal and left_goal.target_position is not None:
                    left_pos = vr_to_world(left_goal.target_position)
                    p = initial_left_target.wxyz_xyz[4:].copy() + left_pos
                    q_current = np.array([0.5, 0.5, 0.5, 0.5])
                    roll_arc = left_goal.wrist_pitch_deg / 360 * np.pi #y world
                    pitch_arc = -left_goal.wrist_yaw_deg / 360 * np.pi #z world
                    yaw_arc = left_goal.wrist_roll_deg / 360 * np.pi #x world

                    q_target = apply_wrist_rotation(q_current, 
                                                    roll_arc, pitch_arc, yaw_arc)
                    T_left  = mink.SE3.from_rotation_and_translation(
                                mink.SO3(q_target), p)
                
                    left_thumb = left_goal.metadata.get('thumbstick', {})
                    if left_thumb:
                        thumb_x1 = left_thumb.get('x', 0)
                        thumb_y1 = left_thumb.get('y', 0)
                        if abs(thumb_x1) > 0.1: 
                            self.viewer.commands[1] += -thumb_x1*0.1 # yaw
                        if abs(thumb_y1) > 0.7:
                            self.viewer.commands[3] += -thumb_y1*0.0001 # height

                    if left_goal.metadata.get('trigger', 0) > 0.5:
                        self.left_gripper_val = 2  # Close
                    else:
                        self.left_gripper_val = -1


                if right_goal and right_goal.target_position is not None:
                    right_pos = vr_to_world(right_goal.target_position)
                    p = initial_right_target.wxyz_xyz[4:].copy() + right_pos
                    q_current = np.array([0.5, 0.5, 0.5, 0.5])
                    roll_arc = right_goal.wrist_pitch_deg / 360 * np.pi #y world
                    pitch_arc = -right_goal.wrist_yaw_deg / 360 * np.pi #z world
                    yaw_arc = right_goal.wrist_roll_deg / 360 * np.pi #x world
                    q_target = apply_wrist_rotation(q_current, 
                                                    roll_arc, pitch_arc, yaw_arc)
                    T_right = mink.SE3.from_rotation_and_translation(
                        mink.SO3(q_target), p)


                    right_thumb = right_goal.metadata.get('thumbstick', {})
                    if right_thumb:
                        thumb_x1 = right_thumb.get('x', 0)
                        thumb_y1 = right_thumb.get('y', 0)
                        if abs(thumb_x1) > 0.1: 
                            self.viewer.commands[2] = -thumb_x1*0.4 # vy
                        else:
                            self.viewer.commands[2] = 0.0
                        if abs(thumb_y1) > 0.1:
                            self.viewer.commands[0] = -thumb_y1*0.5 # vx
                        else:
                            self.viewer.commands[0] = 0.0

                    if right_goal.metadata.get('trigger', 0) > 0.5:
                        self.right_gripper_val = 2  # Close
                    else:
                        self.right_gripper_val = -1




                T_left_to_model = T_left.copy()
                T_right_to_model = T_right.copy()
                
                T_left_target_position = T_left_to_model.wxyz_xyz[4:].copy()
                T_left_target_wxyz = T_left_to_model.wxyz_xyz[:4].copy()
                T_right_target_position = T_right_to_model.wxyz_xyz[4:].copy()
                T_right_target_wxyz = T_right_to_model.wxyz_xyz[:4].copy()

                T_left_to_upper_model_world = mink.SE3.from_rotation_and_translation(
                        rotation=mink.SO3(T_left_target_wxyz),
                        translation=T_left_target_position
                )
                T_right_to_upper_model_world = mink.SE3.from_rotation_and_translation(
                        rotation=mink.SO3(T_right_target_wxyz),
                        translation=T_right_target_position
                )


             
                T_left_right = [T_left_to_upper_model_world, T_right_to_upper_model_world]

                for j, hand_task in enumerate(self.hand_tasks):
                    hand_task.set_target(T_left_right[j])
               
                current_upper_body_dof_pos = self.data.qpos[self.upper_dof_ids+6].copy()  # Exclude the root position and orientation

                self.configuration.update(current_upper_body_dof_pos)
                dq_desired = mink.solve_ik(
                    self.configuration, self.tasks, self.sim_dt, "daqp", 1e-1, limits=self.limits
                )

                dq_ctrl_desired = dq_desired[3:].copy()
                alpha = 0.1
                desired_dq = dq_ctrl_desired * alpha


                step_pos = desired_dq * self.sim_dt
                self.arm_action = current_upper_body_dof_pos[3:] + step_pos
                self.viewer.commands[4] += dq_desired[0]* alpha* self.sim_dt
                self.viewer.commands[6] += dq_desired[1]* alpha* self.sim_dt
                self.viewer.commands[5] += dq_desired[2]* alpha* self.sim_dt
                
            else:
                arm_action_add = np.array([self.viewer.commands[7], self.viewer.commands[8], self.viewer.commands[9], self.viewer.commands[10],
                                        self.viewer.commands[11], self.viewer.commands[12], self.viewer.commands[13], self.viewer.commands[14],
                                        self.viewer.commands[15], self.viewer.commands[16], self.viewer.commands[17], self.viewer.commands[18],
                                        self.viewer.commands[19], self.viewer.commands[20]])
                self.arm_action = np.clip(arm_action_add, self.arm_dof_lower_range, self.arm_dof_upper_range)
                desired_dq = np.zeros(14, dtype=np.float32)

            

            if i % self.sim_decimation == 0:                               
                obs = self.get_observation()
                
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    extra_hist = torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float).view(1, -1).to(self.device)
                    raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
                
                raw_action = np.clip(raw_action, -40., 40.)
                self.last_action = np.concatenate([raw_action.copy(), (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale])
                scaled_actions = raw_action * self.action_scale

                self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
                if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
                    self.gait_cycle = np.array([0.25, 0.25])
                if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
                    self.gait_cycle = np.array([0.25, 0.75])
                
                
                self.viewer.render()

            pd_target = np.concatenate([scaled_actions, np.zeros(8)]) + self.default_dof_pos
            
            target_full_pd = np.zeros(29)
    
            target_full_pd[:15] = pd_target[:15]
            
            target_full_pd[15:] = self.arm_action
            torque = (target_full_pd - self.data.qpos.astype(np.float32)[self.whole_dof_ids+6]) * self.stiffness - self.data.qvel.astype(np.float32)[self.whole_dof_ids+5] * self.damping

            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            
            
            if i > 1000 and self.mink_init:
                desired_ddq = np.zeros(14, dtype=np.float32)
                tau = compute_upper_body_control_torque(
                    self.model,
                    self.data,
                    self.dof_ids,
                    self.pos_pid_controller,
                    target_full_pd[15:],     # qd
                    desired_dq,
                    desired_ddq,
                    self.max_torque,
                    self.sim_dt
                )
                torque[15:] = tau
                
            self.data.ctrl[self.actuator_ids] = torque



            self.left_gripper_pos += gripper_pid_combine(self.data,self.left_gripper_val, 
                                           self.left_arm_left_idx,self.left_arm_right_idx,
                                           self.left_gripper_left_follower_idx, self.left_gripper_right_follower_idx,
                                           self.sim_dt)
            self.right_gripper_pos += gripper_pid_combine(self.data,self.right_gripper_val, 
                                            self.right_arm_left_idx,self.right_arm_right_idx,
                                            self.right_gripper_left_follower_idx, self.right_gripper_right_follower_idx,
                                            self.sim_dt)
            ctrl_min, ctrl_max = [0,255]
            self.left_gripper_pos = np.clip(self.left_gripper_pos, ctrl_min, ctrl_max)
            self.right_gripper_pos = np.clip(self.right_gripper_pos, ctrl_min, ctrl_max)
            
            self.data.ctrl[-2] = self.left_gripper_pos
            self.data.ctrl[-1] = self.right_gripper_pos
          
            
            mujoco.mj_step(self.model, self.data)
            self.upper_data.qpos[:] = self.data.qpos[self.upper_dof_ids+6].copy()  
            self.upper_data.qvel[:] = self.data.qvel[self.upper_dof_ids+5].copy()  
            self.upper_data.qacc[:] = self.data.qacc[self.upper_dof_ids+5].copy()  
            mujoco.mj_step(self.upper_model, self.upper_data)
            
        self.viewer.close()

if __name__ == "__main__":

    robot = "g1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_pth = 'amo/amo_jit.pt'
    
    policy_jit = torch.jit.load(policy_pth, map_location=device)
    
    env = HumanoidEnv(policy_jit=policy_jit, robot_type=robot, device=device)
    
    env.run()
        
        
