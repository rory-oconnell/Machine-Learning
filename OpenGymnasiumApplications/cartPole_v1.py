import gymnasium as gym
from stable_baselines3 import DQN

# Create the environment
env = gym.make('CartPole-v1', render_mode='human')

# Instantiate the agent
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=20000)

# Test the trained agent
observation, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated: # if the episode is terminated or truncated
        observation, info = env.reset() # reset the environment

env.close()