from nlp_gym.data_pools.custom_seq_tagging_pools import UDPosTagggingPool
from nlp_gym.envs.seq_tagging.env import SeqTagEnv
from nlp_gym.envs.seq_tagging.reward import EntityF1Score
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from rich import print
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


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
data_pool = UDPosTagggingPool.prepare(split="train")

# reward function
reward_fn = EntityF1Score(dense=True, average="micro")

# seq tag env
env = SeqTagEnv(data_pool.labels(), reward_function=reward_fn)
for sample, weight in data_pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)

# train a MLP Policy
model = DQN(env=env, policy=DQNPolicy, gamma=0.99, batch_size=32, learning_rate=5e-4,
            double_q=True, exploration_fraction=0.1,
            prioritized_replay=False, policy_kwargs={"layers": [100, 100]},
            verbose=1)

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

# Calculate cumulative average rewards and smooth them
cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
smoothed_cumulative_avg = gaussian_filter1d(cumulative_avg, sigma=2)

# Plot average rewards
plt.plot(steps, smoothed_cumulative_avg, label='DQN', color='green')

plt.xlabel('Steps')
plt.ylabel('Average Episodic Reward')
plt.title('ST with POS tags')
plt.legend(title='strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()