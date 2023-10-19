# pip install gym
# pip install matplotlib

import gymnasium as gym
import numpy as np
import time

env = gym.make('MountainCar-v0' )
env.reset()

# Hyperparameters
LEARNING_RATE = 0.1 # How much we update our Q-values at each iteration
DISCOUNT = 0.95 # How much we value future rewards over current rewards
EPISODES = 25000 # How many episodes we want to run
SHOW_EVERY = 1000 # How often we want to render the environment
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [gym.make('MountainCar-v0').action_space.n]))

def get_discrete_state(state):
    diff = state - gym.make('MountainCar-v0').observation_space.low
    division_result = diff / discrete_os_win_size
    discrete_state = division_result.astype(np.int32)
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODES):
    print(f"Episode {episode}")
    if episode % SHOW_EVERY == 0:
        render_mode1 = "human"
    else:
        render_mode1 = None
    env = gym.make("MountainCar-v0", render_mode =render_mode1)
    state = env.reset()
    discrete_state = get_discrete_state(state[0])
    done = False

    while not done:
        if render_mode1 == "human":
            start_time = time.time()

        action = np.argmax(q_table[discrete_state])
        observation, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(observation)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif observation[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

        if terminated or truncated:
            if render_mode1 == "human":
                end_time = time.time()
                print(f"Time taken: {end_time - start_time}")
            done = True

            
env.close()  # Close environment after each episode