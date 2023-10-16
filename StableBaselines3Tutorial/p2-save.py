import gym
from stable_baselines3 import PPO, DQN, A2C, DDPG
import os

models_dir = "models/DDPG"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('MountainCarContinuous-v0')
env.reset()

model = DDPG('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1, 31):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DDPG")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()