# Code content is from Q-Learning playlist by sentdex

# pip install gym
# pip install matplotlib

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0' )
env.reset()

# Hyperparameters
LEARNING_RATE = 0.1 # How much we update our Q-values at each iteration
DISCOUNT = 0.95 # How much we value future rewards over current rewards
EPISODES = 2000 # How many episodes we want to run

SHOW_EVERY = 500 # How often we want to render the environment
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

epsilon = 0.5 # How much we want to explore
START_EPSILON_DECAYING = 1 # At what episode we start decaying epsilon
END_EPSILON_DECAYING = EPISODES // 2 # At what episode we end decaying epsilon, // is floor division

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [gym.make('MountainCar-v0').action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

def get_discrete_state(state):
    diff = state - gym.make('MountainCar-v0').observation_space.low
    division_result = diff / discrete_os_win_size
    discrete_state = division_result.astype(np.int32)
    return tuple(discrete_state.astype(np.int32))

for episode in range(EPISODES):
    episode_reward = 0
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
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        action = np.argmax(q_table[discrete_state])
    
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(observation)

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif observation[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

        if terminated or truncated:
            if terminated:
                print(f"We made it to the goal! on episode {episode}")
            done = True
        
        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:   # Decaying is for exploration
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}")

env.close()  # Close environment after each episode

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.legend(loc=4)
plt.show()