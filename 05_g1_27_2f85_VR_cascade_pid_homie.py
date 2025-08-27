"""
g1 27dofs with fixed waist pitch and roll
"""
import sys
import time
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
import mink
from include.frame_transform import *
from include.PID_controller import *
import csv
from XLeVR.vr_monitor import VRMonitor
import asyncio
import threading

def vr_to_world(pos):
    x = (pos[2]) * -1.0   # vr_scale_x
    y = (pos[0]) * -1.0  # vr_scale_y
    z = (pos[1]) * 1.0  # vr_scale_z
    return np.array([x, y, z], dtype=np.float32)

def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process paths with LEGGED_GYM_ROOT_DIR
    # for path_key in ['policy_path', 'xml_path', 'reduced_xml_path']:
    #     config[path_key] = config[path_key].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    
    # Convert lists to numpy arrays where needed
    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)
    
    return config

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    
    q_conj = np.array([w, -x, -y, -z])
    
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def compute_observation(d, config, action, cmd, height_cmd, n_joints):
    """Compute the observation vector from current state"""
    # Get state from MuJoCo
    leg_arm_idxs = np.array([0, 1, 2, 3, 4, 5, 
                             6, 7, 8, 9, 10, 11,
                             12, 
                             13, 14, 15, 16, 17, 18, 19,
                             25, 26, 27, 28, 29, 30, 31], dtype=np.int32)
    qj = d.qpos[7+leg_arm_idxs].copy()
    dqj = d.qvel[6+leg_arm_idxs].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    
    # Handle default angles padding
    if len(config['default_angles']) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(config['default_angles'])] = config['default_angles']
    else:
        padded_defaults = config['default_angles'][:n_joints]
    
    # Scale the values
    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']
    
    # Calculate single observation dimension
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12
    
    # Create single observation
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config['cmd_scale']
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+12] = action
    
    return single_obs, single_obs_dim


def main():
    # Load configuration2f85_
    config = load_config("homie/g1_27_2f85.yaml")
    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']
    
    n_joints = 27
    
    # Initialize variables
    action = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    height_cmd = config['height_cmd']
    
    # Initialize observation history
    single_obs, single_obs_dim = compute_observation(d, config, action, cmd, height_cmd, n_joints)
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    
    # Prepare full observation vector
    obs = np.zeros(config['num_obs'], dtype=np.float32)
    
    # Load policy
    policy = torch.jit.load(config['policy_path'])
    
    counter = 0

    upper_dof_names = [
                # waist joints.
                "waist_yaw_joint", 
                # Left arm joints.
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", 
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                # Right arm joints.
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
            ]

    upper_dof_ids = np.array([m.joint(name).id for name in upper_dof_names])

    # ---- mink IK初始化 ----
    upper_model = mujoco.MjModel.from_xml_path(config['reduced_xml_path'])
    upper_model.opt.timestep = config['simulation_dt']
    upper_data = mujoco.MjData(upper_model)
    configuration = mink.Configuration(upper_model)
    pose_cost = np.ones((upper_model.nv)) * 0.1 # 7 dofs for base, and 28 dofs for waist and arms
    pose_cost[0] = 10  # waist joints
    tasks = [
        posture_task := mink.PostureTask(upper_model, cost=pose_cost, lm_damping=1.0),
        
    ]

    hands = ["Left_gripper_center", "Right_gripper_center"]
    hand_tasks = []

    hand_tasks = [
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

    tasks.extend(hand_tasks)
    # collision_pairs = [
    #     # (["left_arm_hand_capsule", "right_arm_hand_capsule",], ["table"]),
    #     # (["left_arm_hand_capsule"], ["left_hip_roll_stl"]),
    #     # (["right_arm_hand_capsule"], ["right_hip_roll_stl"]),
    #     ([
    #       "left_elbow_stl", "right_elbow_stl",
    #       "left_wrist_roll_stl", "right_wrist_roll_stl",
    #         "left_wrist_pitch_stl", "right_wrist_pitch_stl",
    #         "left_wrist_yaw_stl", "right_wrist_yaw_stl"
    #       ], [ "torso_stl"]),
    # ]
    # collision_avoidance_limit = mink.CollisionAvoidanceLimit(
    #     model=m,
    #     geom_pairs=collision_pairs,  # type: ignore
    #     minimum_distance_from_collisions=0.01,
    #     collision_detection_distance=0.05
    # )

    limits = [
        mink.ConfigurationLimit(upper_model),
        # collision_avoidance_limit,
    ]

    configuration.update_from_keyframe("home")
    posture_task.set_target_from_configuration(configuration)

    # for hand in hands:
    #         mink.move_mocap_to_frame(m, d, f"{hand}_target", hand, "site")

    max_torque = np.array([ 200,         # 你的 XML 里 motor forcerange 如有不同请自己改
                        25, 25, 25, 25,  25,  25,  25,
                        25, 25, 25, 25,  25,  25,  25          
                          ])
    pos_pid_controller, vel_pid_controller = create_homie_cascade_biarm_pid_controllers()
    mink_init = False

    # --- For gripper force control ---
    left_arm_left_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_left_pad_touch")
    ]
    left_arm_right_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_right_pad_touch")
    ]
    right_arm_left_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_left_pad_touch")
    ]
    right_arm_right_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_right_pad_touch")
    ]
    left_gripper_left_follower_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_left_follower_touch")
    ]
    left_gripper_right_follower_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_arm_right_follower_touch")
    ]
    right_gripper_left_follower_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_left_follower_touch")
    ]
    right_gripper_right_follower_idx = m.sensor_adr[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_arm_right_follower_touch")
    ]
    left_gripper_val = 0
    right_gripper_val = 0
    left_gripper_pos = 0
    right_gripper_pos = 0

    # ---- VR Monitor ----
    vr_monitor = VRMonitor()
    _vr_thread = threading.Thread(
        target=lambda: asyncio.run(vr_monitor.start_monitoring()),
        daemon=True)
    _vr_thread.start()

    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()   
        initial_left_target = mink.SE3.from_rotation_and_translation(
                                mink.SO3(np.array([0.5, 0.5, 0.5, 0.5])), 
                                np.array([0.38127486, 0.15162665, 0.88822298], dtype=np.float32))
        initial_right_target = mink.SE3.from_rotation_and_translation(
                                mink.SO3(np.array([0.5, 0.5, 0.5, 0.5])), 
                                np.array([0.38127486, -0.15162665, 0.88822298], dtype=np.float32))
        T_left  = initial_left_target
        T_right = initial_right_target
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            step_start = time.time()


        
            # Control leg joints with policy
            leg_tau = pd_control(
                target_dof_pos,
                d.qpos[7:7+config['num_actions']],
                config['kps'],
                np.zeros_like(config['kps']),
                d.qvel[6:6+config['num_actions']],
                config['kds']
            )
            
            d.ctrl[:config['num_actions']] = leg_tau
            

            dual_goals  = vr_monitor.get_latest_goal_nowait()
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
                        cmd[2] += -thumb_x1*0.1 # yaw
                    if abs(thumb_y1) > 0.7:
                        height_cmd += -thumb_y1*0.0001 # height

                if left_goal.metadata.get('trigger', 0) > 0.5:
                    left_gripper_val = 2  # Close
                else:
                    left_gripper_val = -1


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
                        cmd[1] = -thumb_x1*0.4 # vy
                    else:
                        cmd[1] = 0.0
                    if abs(thumb_y1) > 0.1:
                        cmd[0] = -thumb_y1*0.5 # vx
                    else:
                        cmd[0] = 0.0

                if right_goal.metadata.get('trigger', 0) > 0.5:
                    right_gripper_val = 2  # Close
                else:
                    right_gripper_val = -1
                    
            
            T_left_target_position = T_left.wxyz_xyz[4:]
            T_left_target_wxyz = T_left.wxyz_xyz[:4]
            T_right_target_position = T_right.wxyz_xyz[4:]
            T_right_target_wxyz = T_right.wxyz_xyz[:4]     
            T_left_target_position_reduced, T_left_target_wxyz_reduced = world_frame_to_reduced_world_frame(
                                m, d, T_left_target_position, T_left_target_wxyz
                            )
            T_left_to_upper_model_world = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(T_left_target_wxyz_reduced),
                translation=T_left_target_position_reduced
            )
            T_right_target_position_reduced, T_right_target_wxyz_reduced = world_frame_to_reduced_world_frame(
                            m, d, T_right_target_position, T_right_target_wxyz
                        )
            T_right_to_upper_model_world = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(T_right_target_wxyz_reduced),
                translation=T_right_target_position_reduced
            )
            T_left_right = [T_left_to_upper_model_world, T_right_to_upper_model_world]
            for j, hand_task in enumerate(hand_tasks):
                hand_task.set_target(T_left_right[j])

            current_upper_body_dof_pos = d.qpos[upper_dof_ids+6].copy() #19 = 7 (base) + 12 (leg)
            configuration.update(current_upper_body_dof_pos)
            dq_desired = mink.solve_ik(
                    configuration, tasks, config['simulation_dt'], "daqp", 1e-1, limits=limits
                )
            alpha = 0.1
            desired_dq = dq_desired * alpha
            step_pos = desired_dq * config['simulation_dt']

            arm_target_positions = current_upper_body_dof_pos + step_pos
            desired_ddq = np.zeros(15)


            arm_tau, dq_desired_pid = compute_homie_cascade_control_torque(
                m,
                d,
                pos_pid_controller,
                vel_pid_controller,
                arm_target_positions,     # qd
                desired_dq,
                desired_ddq,
                max_torque,
                config['simulation_dt']
            )
            
       
            d.ctrl[12:-2] = arm_tau

            left_gripper_pos += gripper_pid_combine(d,left_gripper_val, 
                                           left_arm_left_idx,left_arm_right_idx,
                                           left_gripper_left_follower_idx, left_gripper_right_follower_idx,
                                           config['simulation_dt'])
            right_gripper_pos += gripper_pid_combine(d,right_gripper_val, 
                                            right_arm_left_idx,right_arm_right_idx,
                                            right_gripper_left_follower_idx, right_gripper_right_follower_idx,
                                            config['simulation_dt'])
            ctrl_min, ctrl_max = [0,255]
            left_gripper_pos = np.clip(left_gripper_pos, ctrl_min, ctrl_max)
            right_gripper_pos = np.clip(right_gripper_pos, ctrl_min, ctrl_max)

            d.ctrl[-2] = left_gripper_pos
            d.ctrl[-1] = right_gripper_pos 
            # Step physics
            mujoco.mj_step(m, d)
            upper_data.qpos[:] = d.qpos[upper_dof_ids+6].copy() 
            upper_data.qvel[:] = d.qvel[upper_dof_ids+5].copy()  
            upper_data.qacc[:] = d.qacc[upper_dof_ids+5].copy() 
            mujoco.mj_step(upper_model, upper_data)


            counter += 1
            if counter % config['control_decimation'] == 0:
                # Update observation
                single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, n_joints)
                obs_history.append(single_obs)
                
                # Construct full observation with history
                for i, hist_obs in enumerate(obs_history):
                    start_idx = i * single_obs_dim
                    end_idx = start_idx + single_obs_dim
                    obs[start_idx:end_idx] = hist_obs
                
                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                
                # Transform action to target_dof_pos
                target_dof_pos = action * config['action_scale'] + config['default_angles']
            
            # Sync viewer
            viewer.sync()
            
            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()