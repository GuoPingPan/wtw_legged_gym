import torch
from isaacgym.torch_utils import *



from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.go1.w_arm.go1_arm_config import Go1ArmCfg


class Go1(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                    cfg: Go1ArmCfg = None, eval_cfg: Go1ArmCfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):
        super().__init__(sim_device, headless, num_envs=num_envs, prone=prone, deploy=deploy,
                    cfg=cfg, eval_cfg=eval_cfg, initial_dynamics_dict=initial_dynamics_dict, physics_engine=physics_engine)

        self.arm_time_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)



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
        super().post_physics_step()
    
        self._reflesh_arm_command()

        

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        # 随机x，y方向的速度指令
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        self._resample_arm_commands(env_ids)

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

        self.arm_commands_end[env_ids, 0] = torch_rand_float(self.command_ranges["l"][0], self.command_ranges["l"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.arm_commands_end[env_ids, 1] = torch_rand_float(self.command_ranges["p"][0], self.command_ranges["p"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.arm_commands_end[env_ids, 2] = torch_rand_float(self.command_ranges["y"][0], self.command_ranges["y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self._resample_Traj_commands(env_ids)

    def _resample_Traj_commands(self, env_ids):
        '''
            随机从1~3s中选择一个时间间隔
        '''
        time_range = (self.command_ranges["T_traj"][1] - self.command_ranges["T_traj"][0])/self.dt
        time_interval = torch.from_numpy(np.random.choice(int(time_range+1), len(env_ids))).to(self.device)

        self.T_trajs[env_ids] = torch.ones_like(self.T_trajs[env_ids]) * self.command_ranges["T_traj"][0] + time_interval * self.dt
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
        z = self.end_effector_state[env_ids, 2] - self.root_states[env_ids, 2]

        l = torch.sqrt(x**2 + y**2 + z**2)
        p = torch.atan2(torch.sqrt(x**2 + y**2), z)
        y = torch.atan2(y, x)

        return torch.stack([l, p, y], dim=-1)
