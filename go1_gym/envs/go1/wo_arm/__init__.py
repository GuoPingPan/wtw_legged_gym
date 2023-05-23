import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
assert gymtorch

from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.envs.go1.wo_arm.go1_config import Go1Cfg

class Go1(LeggedRobot):
    def __init__(self, cfg: Go1Cfg, sim_params, physics_engine, sim_device, headless, eval_cfg=None,
                 initial_dynamics_dict=None):
        
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, sim_device=sim_device, headless=headless, eval_cfg=eval_cfg,
                 initial_dynamics_dict=initial_dynamics_dict)
    
    def _resample_commands(self, env_ids):
        if len(env_ids) == 0: return

        self.commands[env_ids, 0] = torch_rand_float(self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_x[1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.commands[env_ids, 1] = torch_rand_float(self.cfg.commands.lin_vel_y[0], self.cfg.commands.lin_vel_y[1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.commands[env_ids, 2] = torch_rand_float(self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.ang_vel_yaw[1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

class Go1TrackingEasyEnv(Go1):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                    cfg: Go1Cfg = None, eval_cfg: Go1Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX"):

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