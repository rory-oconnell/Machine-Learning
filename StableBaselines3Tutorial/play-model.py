import gym
from stable_baselines3 import PPO, DQN, A2C, DDPG

models_dir = "models/DQN"

env = gym.make('MountainCar-v0', render_mode="human")  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/300000.zip"
model = DQN.load(model_path, env=env)

episodes = 10000

# Test the trained agent
observation, info = env.reset()
for _ in range(10000):
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated: # if the episode is terminated or truncated
        observation, info = env.reset() # reset the environment

env.close()

# Test the trained agent
observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated: # if the episode is terminated or truncated
        observation, info = env.reset() # reset the environment

env.close()