import gymnasium as gym
env = gym.make('MountainCar-v0', render_mode='human')

# Resets the environment to an initial state, required before calling step. 
# Returns the first agent observation for an episode and information, i.e. metrics, debug info.
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)


# When the end of an episode is reached (terminated or truncated), it is necessary 
# to call reset() to reset this environment’s state for the next episode.
    if terminated or truncated:
        observation, info = env.reset()

env.close()