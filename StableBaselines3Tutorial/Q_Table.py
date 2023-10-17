# After several attempts at training the Mountain Car environment with DQN, PPO, and A2C
# I was unable to get any of the algorithms to solve the environment. The training would
# plateau at around -150 reward. I tried changing the hyperparameters, but nothing seemed
# to work. Thus I am trying Q-Learning to see if I can get it to work.

# pip install gym
# pip install matplotlib

import gymnasium as gym
import numpy as np

env = gym.make('MountainCar-v0', render_mode='human')

# Hyperparameters
LEARNING_RATE = 0.1 # How much we update our Q-values at each iteration
DISCOUNT = 0.95 # How much we value future rewards over current rewards
EPISODES = 25000 # How many episodes we want to run

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# The Q-values contains the Q-values for every state-action pair.
# A Q-value is the expected future reward for a given state-action pair.
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

discrete_state = get_discrete_state(env.reset())

print(discrete_state)

# # Resets the environment to an initial state, required before calling step. 
# # Returns the first agent observation for an episode and information, i.e. metrics, debug info.
# observation, info = env.reset()
# 
# while True:
#     action = 2 
#     observation, reward, terminated, truncated, info = env.step(action)
#     print(observation)
# 
# # When the end of an episode is reached (terminated or truncated), it is necessary 
# # to call reset() to reset this environment’s state for the next episode.
# 
# env.close()