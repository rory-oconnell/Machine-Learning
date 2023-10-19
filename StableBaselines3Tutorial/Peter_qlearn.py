import gym
import numpy as np

env = gym.make('MountainCar-v0' )
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
SHOW_EVERY = 1000

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))
EPSILON = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = EPSILON/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

discrete_state = get_discrete_state(env.reset()[0])
print(np.argmax(q[discrete_state]))
i = 0
done = False

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
        if np.random.random() > EPSILON:
            action = np.argmax(q[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, termination, truncation, _  = env.step(action)
        if termination or truncation:
            done = True
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q[new_discrete_state])
            current_q = q[discrete_state + (action, )]

            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state
    env.close()
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        EPSILON -= epsilon_decay_value
env.render()
env.close()