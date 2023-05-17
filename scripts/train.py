import isaacgym
assert isaacgym
import torch

from go1_gym.envs.go1.wo_arm.go1_config import Go1Cfg, Go1RunnerArgs
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from ml_logger import logger

from go1_gym_learn.ppo_cse import Runner
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym_learn.ppo_cse.actor_critic import AC_Args
from go1_gym_learn.ppo_cse.ppo import PPO_Args



def train_go1(args):

  Go1Cfg.env.num_envs = args.num_envs

  env = VelocityTrackingEasyEnv(sim_device=args.sim_device, headless=args.headless, cfg=Go1Cfg)

  # log the experiment parameters
  logger.log_params(AC_Args=vars(AC_Args), PPO_Args=vars(PPO_Args), Go1RunnerArgs=vars(Go1RunnerArgs),
                    Go1Cfg=vars(Go1Cfg))

  env = HistoryWrapper(env)
  gpu_id = args.sim_device.split(":")[-1]
  runner = Runner(env, runner_args=Go1RunnerArgs, device=f"cuda:{gpu_id}")
  runner.learn(num_learning_iterations=args.num_learning_iterations, init_at_random_ep_len=True, eval_freq=args.eval_freq)


if __name__ == '__main__':
  from pathlib import Path
  from ml_logger import logger
  from go1_gym import MINI_GYM_ROOT_DIR
  import argparse

  parser = argparse.ArgumentParser(description="Go1")
  parser.add_argument('--headless', action='store_true', default=False)
  parser.add_argument('--sim_device', type=str, default="cuda:0")
  parser.add_argument('--num_learning_iterations', type=int, default=100000)
  parser.add_argument('--eval_freq', type=int, default=100)
  parser.add_argument('--num_envs', type=int, default=4096)

  args = parser.parse_args()
  

  stem = Path(__file__).stem
  logger.configure(logger.utcnow(f'gait-conditioned-agility/%Y-%m-%d/{stem}/%H%M%S.%f'),
                    root=Path(f"{MINI_GYM_ROOT_DIR}/runs").resolve(), )
  logger.log_text("""
              charts: 
              - yKey: train/episode/rew_total/mean
                xKey: iterations
              - yKey: train/episode/rew_tracking_lin_vel/mean
                xKey: iterations
              - yKey: train/episode/rew_tracking_contacts_shaped_force/mean
                xKey: iterations
              - yKey: train/episode/rew_action_smoothness_1/mean
                xKey: iterations
              - yKey: train/episode/rew_action_smoothness_2/mean
                xKey: iterations
              - yKey: train/episode/rew_tracking_contacts_shaped_vel/mean
                xKey: iterations
              - yKey: train/episode/rew_orientation_control/mean
                xKey: iterations
              - yKey: train/episode/rew_dof_pos/mean
                xKey: iterations
              - yKey: train/episode/command_area_trot/mean
                xKey: iterations
              - yKey: train/episode/max_terrain_height/mean
                xKey: iterations
              - type: video
                glob: "videos/*.mp4"
              - yKey: adaptation_loss/mean
                xKey: iterations
              """, filename=".charts.yml", dedent=True)

  # to see the environment rendering, set headless=False
  train_go1(args)
