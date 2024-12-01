from scipy.ndimage import gaussian_filter1d

from nlp_gym.data_pools.custom_multi_label_pools import ReutersDataPool
from nlp_gym.envs.multi_label.env import MultiLabelEnv
from nlp_gym.envs.multi_label.reward import F1RewardFunction
from stable_baselines.deepq import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from rich import print
import matplotlib.pyplot as plt
import numpy as np


def eval_model(model, env):
    done = False
    obs = env.reset()
    total_reward = 0.0
    actions = []
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        actions.append(env.action_space.ix_to_action(action))
        total_reward += rewards
    print("---------------------------------------------")
    print(f"Text: {env.current_sample.text}")
    print(f"Predicted Label {actions}")
    print(f"Oracle Label: {env.current_sample.label}")
    print(f"Total Reward: {total_reward}")
    print("---------------------------------------------")


# data pool
pool = ReutersDataPool.prepare(split="train")
labels = pool.labels()

# reward function
reward_fn = F1RewardFunction()

# multi label env
env = MultiLabelEnv(possible_labels=labels, max_steps=10, reward_function=reward_fn,
                    return_obs_as_vector=True)
for sample, weight in pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)

# train a MLP Policy
model = DQN(env=env, policy=DQNPolicy, gamma=0.99, batch_size=32, learning_rate=1e-3,
            double_q=True, exploration_fraction=0.1,
            prioritized_replay=False, policy_kwargs={"layers": [200]},
            verbose=1)

# Lists to store metrics for plotting
steps = []
rewards = []
current_step = 0

total_iterations = int(1e+3)
timesteps_per_iteration = int(1e+3)

for i in range(total_iterations):
    model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
    current_step += timesteps_per_iteration

    # Evaluate and store metrics
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    steps.append(current_step)
    rewards.append(episode_reward)

    eval_model(model, env)

# Plotting
plt.figure(figsize=(10, 6))

# Calculate cumulative average rewards
cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
smoothed_cumulative_avg = gaussian_filter1d(cumulative_avg, sigma=2)

# Plot average rewards
plt.plot(steps, smoothed_cumulative_avg, label='DQN', color='green')

plt.xlabel('Steps')
plt.ylabel('Average Episodic Reward')
plt.title('MLC with Reuters')
plt.legend(title='strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()