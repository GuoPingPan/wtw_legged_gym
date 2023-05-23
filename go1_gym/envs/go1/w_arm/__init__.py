import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
assert gymtorch

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.go1.w_arm.go1_arm_config import Go1ArmCfg


class Go1Arm(LeggedRobot):
    def __init__(self, cfg: Go1ArmCfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        
        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions
        self.num_actions_loco = cfg.env.num_actions_loco
        self.num_actions_arm = cfg.env.num_actions_arm

        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, sim_device=sim_device, headless=headless, eval_cfg=eval_cfg,
                 initial_dynamics_dict=initial_dynamics_dict)
        


        self.arm_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            if self.cfg.env.keep_arm_fixed:
                self._keep_arm_fixed()
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _keep_arm_fixed(self):
        self.dof_pos[:, self.num_actions:] = self.default_dof_pos[:, self.num_actions:]
        self.dof_vel[:, self.num_actions:] = 0.

        # HACK(pgp) fixed last 3 DOFs of arm
        self.gym.set_dof_state_tensor(self.sim,
                                        gymtorch.unwrap_tensor(self.dof_state))

        # HACK(pgp) fixed based height
        # self.root_states[:, 2] = 0.30 + self.measured_heights
        # self.gym.set_actor_root_state_tensor(self.sim,
        #                                       gymtorch.unwrap_tensor(self.root_states))

    def check_termination(self):
        super().check_termination()

        self.reverse_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)


        # NOTE 如果 resample arm action 会导致出问题，身体还没矫正，因此在走到 0.6 路程的时候进行判断
        if self.cfg.rewards.use_terminal_roll:
            roll_vec = quat_apply(self.base_quat, self.roll_vec) # [0,1,0]
            roll = torch.atan2(roll_vec[:, 2], roll_vec[:, 1]) # roll angle = arctan2(z, y)
            reverse_buf1 = torch.logical_and(roll > self.cfg.rewards.terminal_body_ori, self.commands[:, 4] > 0.0) # lpy
            reverse_buf2 = torch.logical_and(roll < -self.cfg.rewards.terminal_body_ori, self.commands[:, 4] < 0.0) # lpy
            self.reverse_buf |= reverse_buf1 | reverse_buf2

            print("roll: ", roll)

        if self.cfg.rewards.use_terminal_pitch:
            pitch_vec = quat_apply(self.base_quat, self.pitch_vec) # [0,0,1]
            pitch = torch.atan2(pitch_vec[:, 0], pitch_vec[:, 2]) # pitch angle = arctan2(x, z)
            reverse_buf3 = torch.logical_and(pitch > self.cfg.rewards.terminal_body_pitch, self.commands[:, 3] > 0.0) # lpy
            reverse_buf4 = torch.logical_and(pitch < -self.cfg.rewards.terminal_body_pitch, self.commands[:, 3] < 0.0) # lpy
            self.reverse_buf |= reverse_buf3 | reverse_buf4

            print("pitch: ", pitch)

        time_exceed_half = (self.arm_time_buf / (self.T_trajs / self.dt)) > 0.6
        self.reverse_buf = self.reverse_buf & time_exceed_half
        self.reset_buf |= self.reverse_buf


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        # pd controller
        actions_scaled = actions[:, :self.num_actions] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range
        actions_scaled = torch.nn.functional.pad(actions_scaled, (0, self.num_dof - self.num_actions), "constant", 0.0)

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        # import ipdb; ipdb.set_trace()
        # torques = torques * self.motor_strengths
        # # TODO(pgp) pad torques with 0 in dim1 from num_actions to num_dofs
        # torques_temp = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        # torques_temp[..., :self.num_actions] = torques
        # torques = torques_temp
        torques[:, self.num_actions:] = 0.0
        # print("toques: ", torques[0])
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ----------------------------------------
    def _init_buffers(self):
        super()._init_buffers()

        # TODO(pgp) joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # LF_HAA, LF_HFE, LF_KFE, ... 每个脚3个， 即4x3=12
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False

            if i >= self.num_actions: 
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                continue

            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness_leg[dof_name] \
                        if i < self.num_actions_loco else self.cfg.control.stiffness_arm[dof_name] # 刚性，[N*m/rad]
                    self.d_gains[i] = self.cfg.control.damping_leg[dof_name] \
                        if i < self.num_actions_loco else self.cfg.control.damping_arm[dof_name]  # 阻尼，[N*m*s/rad]
                    found = True

            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]: # P: position, V: velocity, T: torques
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0) # [1,20]
        
        # 加入DWB的约束
        self.end_effector_state = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, 22] # link6
        self.roll_vec = to_torch([0., 1., 0.], device=self.device).repeat((self.num_envs, 1))
        self.pitch_vec = to_torch([0., 0., 1.], device=self.device).repeat((self.num_envs, 1))


        self.T_trajs = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # command 时间
        self.arm_commands_start = torch.zeros(self.num_envs, self.cfg.commands.num_commands_arm, dtype=torch.float, device=self.device, requires_grad=False) 
        self.arm_commands_end = torch.zeros(self.num_envs, self.cfg.commands.num_commands_arm, dtype=torch.float, device=self.device, requires_grad=False) 


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from go1_gym.envs.rewards.corl_rewards import CoRLRewards
        from go1_gym.envs.rewards.icra_rewards import ICRARewards
        reward_containers = {"CoRLRewards": CoRLRewards, "ICRARewards": ICRARewards}
        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            # print(key, scale)
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def compute_reward(self):
        super().compute_reward()
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 1]) ** 2

    def compute_observations(self):
        super().compute_observations()

        # TODO
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.end_effector_state[:, :3]), dim=1)
        self.next_privileged_obs_buf = torch.cat((self.next_privileged_obs_buf, self.end_effector_state[:, :3]), dim=1)
        
        assert self.privileged_obs_buf.shape[
            1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    def _post_physics_step_callback(self):
        # teleport robots to prevent falling off the edge
        self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # TODO(pgp) _resample_arm_commands 
        traj_ids = (self.arm_time_buf % (self.T_trajs / self.dt).long()==0).nonzero(as_tuple=False).flatten()
        self._resample_arm_commands(traj_ids)

        self._step_contact_targets()

        # measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # push robots
        self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # randomize dof properties
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.arm_time_buf += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        # TODO(pgp)
        self.end_effector_state[:] = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, 22]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        
        # TODO 这里应该是先计算 reward 再进行 reflesh?
        self._reflesh_arm_command()

        self._render_headless()
        
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0: return

        # 随机x，y方向的速度指令
        self.commands[env_ids, 0] = torch_rand_float(self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_x[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.ang_vel_yaw[1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        self._resample_arm_commands(env_ids)

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.


    def _resample_arm_commands(self, env_ids):
        '''
            每次到了一定的时间间隔就重新resample arm的 lpy
            1. 记录当前 lpy
            2. 重新sample lpy
            3. sample T_traj
            4. 重置时间
            5. refresh _reflesh_arm_command
        
        '''
        self.arm_commands_start[env_ids] = self._get_lpy_in_base_coord(env_ids)

        self.arm_commands_end[env_ids, 0] = torch_rand_float(self.cfg.commands.l[0], self.cfg.commands.l[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.arm_commands_end[env_ids, 1] = torch_rand_float(self.cfg.commands.p[0], self.cfg.commands.p[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.arm_commands_end[env_ids, 2] = torch_rand_float(self.cfg.commands.y[0], self.cfg.commands.y[1], (len(env_ids), 1), device=self.device).squeeze(1)

        self._resample_Traj_commands(env_ids)

    def _resample_Traj_commands(self, env_ids):
        '''
            随机从1~3s中选择一个时间间隔
        '''
        time_range = (self.cfg.commands.T_traj[1] - self.cfg.commands.T_traj[0])/self.dt
        time_interval = torch.from_numpy(np.random.choice(int(time_range+1), len(env_ids))).to(self.device)

        self.T_trajs[env_ids] = torch.ones_like(self.T_trajs[env_ids]) * self.cfg.commands.T_traj[0] + time_interval * self.dt
        self.commands[env_ids, 5] = self.T_trajs[env_ids]

        self.arm_time_buf[env_ids] = torch.zeros_like(self.arm_time_buf[env_ids])

        self._reflesh_arm_command()

    def _reflesh_arm_command(self):
        '''
            插值更新arm的command
        '''
        
        # import ipdb; ipdb.set_trace()
        self.commands[:, 2:5] = \
                self.arm_commands_start * (1 - (self.arm_time_buf / (self.T_trajs / self.dt))).unsqueeze(-1) \
                + self.arm_commands_end * (self.arm_time_buf / (self.T_trajs / self.dt)).unsqueeze(-1) 

    def _get_lpy_in_base_coord(self, env_ids):
        # TODO(pgp) key isaac gym 中的 tensor 不能用于创建新的变量
        # rpy = quat_to_euler(self.base_quat[env_ids])
        # yaw = rpy[..., 2]
        forward = quat_apply(self.base_quat[env_ids], self.forward_vec[env_ids])
        yaw = torch.atan2(forward[:, 1], forward[:, 0])

        # import ipdb; ipdb.set_trace()

        # 先减去机器人坐标
        x = torch.cos(yaw) * (self.end_effector_state[env_ids, 0] - self.root_states[env_ids, 0]) \
            + torch.sin(yaw) * (self.end_effector_state[env_ids, 1] - self.root_states[env_ids, 1])
        y = -torch.sin(yaw) * (self.end_effector_state[env_ids, 0] - self.root_states[env_ids, 0]) \
            + torch.cos(yaw) * (self.end_effector_state[env_ids, 1] - self.root_states[env_ids, 1])
        # z = self.end_effector_state[env_ids, 2] - self.root_states[env_ids, 2]
        z = self.end_effector_state[env_ids, 2] - 0.3
        l = torch.sqrt(x**2 + y**2 + z**2)
        p = torch.atan2(z, torch.sqrt(x**2 + y**2)) # NOTE 这里的角度是否有问题？
        y_aw = torch.atan2(y, x)

        return torch.stack([l, p, y_aw], dim=-1)




class Go1ArmTrackingEasyEnv(Go1Arm):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                    cfg: Go1ArmCfg = None, eval_cfg: Go1ArmCfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

        if num_envs is not None:
            cfg.env.num_envs = num_envs

        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict)


    def step(self, actions):
        self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras = super().step(actions)

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                                0:3]

        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 1],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

