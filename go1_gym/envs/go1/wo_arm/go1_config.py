from typing import Union

from params_proto import Meta

from go1_gym.envs.base.legged_robot_config import Cfg, RunnerArgs

class Go1RunnerArgs(RunnerArgs):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 1500  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True


class Go1Cfg(Cfg):
    class init_state(Cfg.init_state):
        pos = [0.0, 0.0, 0.34]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
        'FL_hip_joint': 0.1,  # [rad]
        'RL_hip_joint': 0.1,  # [rad]
        'FR_hip_joint': -0.1,  # [rad]
        'RR_hip_joint': -0.1,  # [rad]

        'FL_thigh_joint': 0.8,  # [rad]
        'RL_thigh_joint': 1.,  # [rad]
        'FR_thigh_joint': 0.8,  # [rad]
        'RR_thigh_joint': 1.,  # [rad]

        'FL_calf_joint': -1.5,  # [rad]
        'RL_calf_joint': -1.5,  # [rad]
        'FR_calf_joint': -1.5,  # [rad]
        'RR_calf_joint': -1.5  # [rad]
        }

    class control(Cfg.control):
        # Cnfg.control.control_type = "actuator_net"
        control_type = 'P'

        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(Cfg.asset):
        file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        fix_base_link = False

    class rewards(Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        reward_container_name = "CoRLRewards"
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02

    class reward_scales(Cfg.reward_scales):
        torques = -0.0001
        action_rate = -0.01
        dof_pos_limits = -10.0

        feet_contact_forces = 0.0
        feet_slip = -0.04
        action_smoothness_1 = -0.1
        action_smoothness_2 = -0.1
        dof_vel = -1e-4
        dof_pos = -0.0
        jump = 10.0
        base_height = 0.0
        estimation_bonus = 0.0
        raibert_heuristic = -10.0
        feet_impact_vel = -0.0
        feet_clearance = -0.0
        feet_clearance_cmd = -0.0
        feet_clearance_cmd_linear = -30.0
        orientation = 0.0
        orientation_control = -5.0
        tracking_stance_width = -0.0
        tracking_stance_length = -0.0
        lin_vel_z = -0.02
        ang_vel_xy = -0.001
        feet_air_time = 0.0
        hop_symmetry = 0.0
        tracking_contacts_shaped_force = 4.0
        tracking_contacts_shaped_vel = 4.0
        collision = -5.0

    class terrain(Cfg.terrain):
        measure_heights = False
        terrain_noise_magnitude = 0.0
        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        curriculum = False


        yaw_init_range = 3.14
        border_size = 0.0
        mesh_type = "trimesh"
        num_cols = 30
        num_rows = 30
        terrain_width = 5.0
        terrain_length = 5.0
        x_init_range = 0.2
        y_init_range = 0.2
        teleport_thresh = 0.3
        teleport_robots = False
        center_robots = True
        center_span = 4
        horizontal_scale = 0.10
        use_terminal_foot_height = False
        use_terminal_body_height = True
        terminal_body_height = 0.05
        use_terminal_roll_pitch = True
        terminal_body_ori = 1.6

    class normalization(Cfg.normalization):
        friction_range = [0, 1]
        ground_friction_range = [0, 1]
        clip_actions = 10.0

    class env(Cfg.env):
        observe_vel = False
        num_envs = 4000

        priv_observe_motion = False
        priv_observe_gravity_transformed_motion = False
        priv_observe_friction_indep = False
        priv_observe_friction = True
        priv_observe_restitution = True
        priv_observe_base_mass = False
        priv_observe_gravity = False
        priv_observe_com_displacement = False
        priv_observe_ground_friction = False
        priv_observe_ground_friction_per_foot = False
        priv_observe_motor_strength = False
        priv_observe_motor_offset = False
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        priv_observe_body_velocity = False
        priv_observe_body_height = False
        priv_observe_desired_contact_states = False
        priv_observe_contact_forces = False
        priv_observe_foot_displacement = False
        priv_observe_gravity_transformed_foot_displacement = False
        num_privileged_obs = 2
        num_observation_history = 30
        observe_two_prev_actions = True
        observe_yaw = False
        num_observations = 70
        num_scalar_observations = 70
        observe_gait_commands = True
        observe_timing_parameter = False
        observe_clock_inputs = True


    class commands(Cfg.commands):
        heading_command = False
        command_curriculum = True
   

        num_lin_vel_bins = 30
        num_ang_vel_bins = 30
        distributional_commands = True
        num_commands = 15
        resampling_time = 10.
        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-0.6, 0.6]
        ang_vel_yaw = [-1.0, 1.0]
        body_height_cmd = [-0.25, 0.15]
        gait_frequency_cmd_range = [2.0, 4.0]
        gait_phase_cmd_range = [0.0, 1.0]
        gait_offset_cmd_range = [0.0, 1.0]
        gait_bound_cmd_range = [0.0, 1.0]
        gait_duration_cmd_range = [0.5, 0.5]
        footswing_height_range = [0.03, 0.35]
        body_pitch_range = [-0.4, 0.4]
        body_roll_range = [-0.0, 0.0]
        stance_width_range = [0.10, 0.45]
        stance_length_range = [0.35, 0.45]
        limit_vel_x = [-5.0, 5.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]
        limit_body_height = [-0.25, 0.15]
        limit_gait_frequency = [2.0, 4.0]
        limit_gait_phase = [0.0, 1.0]
        limit_gait_offset = [0.0, 1.0]
        limit_gait_bound = [0.0, 1.0]
        limit_gait_duration = [0.5, 0.5]
        limit_footswing_height = [0.03, 0.35]
        limit_body_pitch = [-0.4, 0.4]
        limit_body_roll = [-0.0, 0.0]
        limit_stance_width = [0.10, 0.45]
        limit_stance_length = [0.35, 0.45]
        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_gait_duration = 1
        num_bins_footswing_height = 1
        num_bins_body_roll = 1
        num_bins_body_pitch = 1
        num_bins_stance_width = 1
        exclusive_phase_offset = False
        pacing_offset = False
        binary_phases = True
        gaitwise_curricula = True


    class curriculum_thresholds(Cfg.curriculum_thresholds):
        tracking_ang_vel = 0.7
        tracking_lin_vel = 0.8
        tracking_contacts_shaped_vel = 0.9
        tracking_contacts_shaped_force = 0.9

    class domain_rand(Cfg.domain_rand):
        max_push_vel_xy = 0.5
        restitution = 0.5  # default terrain restitution
        Kp_factor_range = [0.8, 1.3]
        Kd_factor_range = [0.5, 1.5]

        lag_timesteps = 6
        randomize_lag_timesteps = True
        randomize_rigids_after_start = False
        randomize_friction_indep = False
        randomize_friction = True
        friction_range = [0.1, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99
        randomize_com_displacement = False
        com_displacement_range = [-0.15, 0.15]
        randomize_ground_friction = True
        ground_friction_range = [0.0, 0.0]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        push_robots = False
        randomize_Kp_factor = False
        randomize_Kd_factor = False
        rand_interval_s = 4
        tile_height_range = [-0.0, 0.0]
        tile_height_curriculum = False
        tile_height_update_interval = 1000000
        tile_height_curriculum_step = 0.01


def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    
    return result

if __name__ == '__main__':
    cfg = Go1Cfg
    print(cfg.normalization.added_mass_range)
    print(cfg.init_state.pos)
    print(cfg.reward_scales.tracking_lin_vel)

    print(class_to_dict(cfg.reward_scales))
    print(cfg.reward_scales.__dict__)