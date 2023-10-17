# After several attempts at training the Mountain Car environment with DQN, PPO, and A2C
# I was unable to get any of the algorithms to solve the environment. The training would
# plateau at around -150 reward. I tried changing the hyperparameters, but nothing seemed
# to work. Thus I am trying Q-Learning to see if I can get it to work.

# pip install gym
# pip install matplotlib

import gymnasium as gym
import numpy as np

# Hyperparameters
LEARNING_RATE = 0.1 # How much we update our Q-values at each iteration
DISCOUNT = 0.95 # How much we value future rewards over current rewards
EPISODES = 10000000 # How many episodes we want to run
SHOW_EVERY = 10000000 # How often we want to render the environment

DISCRETE_OS_SIZE = [20] * len(gym.make('MountainCar-v0').observation_space.high)
discrete_os_win_size = (gym.make('MountainCar-v0').observation_space.high - gym.make('MountainCar-v0').observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [gym.make('MountainCar-v0').action_space.n]))

def get_discrete_state(state):
    diff = state - gym.make('MountainCar-v0').observation_space.low
    division_result = diff / discrete_os_win_size
    discrete_state = division_result.astype(np.int32)
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODES):
    render = episode % SHOW_EVERY == 0
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    state, info = env.reset()
    discrete_state = get_discrete_state(state)

    if render:
        print(episode)

    while True:
        action = np.argmax(q_table[discrete_state])
        observation, reward, terminated, truncated, _ = env.step(action)  # Modified the unpacking of return values.
        new_discrete_state = get_discrete_state(observation)

        done = terminated or truncated  # Combining termination and truncation

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif observation[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    env.close()  # Close environment after each episode
