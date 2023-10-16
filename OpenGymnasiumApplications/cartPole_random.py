import gymnasium as gym
env = gym.make('CartPole-v1', render_mode='human')

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action) # the agent takes a step in the environment

    if terminated or truncated: # if the episode is terminated or truncated
        observation, info = env.reset() # reset the environment

env.close()