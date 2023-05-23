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
    resume_curriculum = False


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
        base_height_target = 0.34
        use_terminal_body_height = True

    class reward_scales(Cfg.reward_scales):
        torques = -0.0002
        # action_rate = -0.01
        dof_pos_limits = -10.0
        # orientation = -0.
        # base_height = -0.5
        # collision = -5.


    class terrain(Cfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = False
        terrain_noise_magnitude = 0.0
        teleport_robots = False
        border_size = 50

        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        curriculum = False


    class normalization(Cfg.normalization):
        friction_range = [0, 1]
        ground_friction_range = [0, 1]
        clip_actions = 10.0

    class env(Cfg.env):
        num_observations = 42
        observe_vel = False
        num_envs = 4000
        priv_observe_Kp_factor = False
        priv_observe_Kd_factor = False
        num_privileged_obs = 6 # TODO


    class commands(Cfg.commands):
        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-1.0, 1.0]
        heading_command = False
        resampling_time = 10.0
        command_curriculum = False
        num_lin_vel_bins = 30
        num_ang_vel_bins = 30
        ang_vel_yaw = [-1, 1]

        # NOTE modify
        gaitwise_curricula = False


    class curriculum_thresholds(Cfg.curriculum_thresholds):
        tracking_ang_vel = 0.7
        tracking_lin_vel = 0.8
        tracking_contacts_shaped_vel = 0.9
        tracking_contacts_shaped_force = 0.9

    class domain_rand(Cfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        push_robots = False
        max_push_vel_xy = 0.5
        randomize_friction = True
        friction_range = [0.1, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        restitution = 0.5  # default terrain restitution
        randomize_motor_offset = False

        randomize_com_displacement = True
        com_displacement_range = [-0.1, 0.1]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = False
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = False
        Kd_factor_range = [0.5, 1.5]
        rand_interval_s = 6
        randomize_lag_timesteps = False
        randomize_rigids_after_start = False
        randomize_friction_indep = False

if __name__ == '__main__':
    cfg = Go1Cfg
    print(cfg.normalization.added_mass_range)
    print(cfg.init_state.pos)
    print(cfg.reward_scales.tracking_lin_vel)


