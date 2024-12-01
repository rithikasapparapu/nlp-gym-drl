from nlp_gym.data_pools.custom_multi_label_pools import ReutersDataPool
from nlp_gym.envs.multi_label.env import MultiLabelEnv
from nlp_gym.envs.multi_label.reward import F1RewardFunction
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
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
        action, _states = model.predict(obs, deterministic=True)
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

# train a PPO2 Policy
model = PPO2(
    policy=MlpPolicy,
    env=env,
    learning_rate=3e-4,
    n_steps=128,
    nminibatches=4,
    lam=0.95,
    gamma=0.99,
    noptepochs=4,
    ent_coef=0.01,
    cliprange=0.2,
    verbose=1,
    policy_kwargs={"net_arch": [dict(pi=[200], vf=[200])]}
)

# Add these before the training loop
steps = []
rewards = []
current_step = 0

total_timesteps = int(1e2)  # Increased total timesteps for better learning
num_iterations = 100  # Number of training iterations
timesteps_per_iteration = total_timesteps // num_iterations

for i in range(num_iterations):
    model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
    current_step += timesteps_per_iteration

    # Evaluate and store metrics
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    steps.append(current_step)
    rewards.append(episode_reward)

    print(f"Iteration {i + 1}/{num_iterations} completed")
    eval_model(model, env)

# Plotting
plt.figure(figsize=(10, 6))

# Calculate cumulative average rewards and smooth them
cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
smoothed_cumulative_avg = gaussian_filter1d(cumulative_avg, sigma=2)

# Plot average rewards
plt.plot(steps, smoothed_cumulative_avg, label='PPO', color='red')

plt.xlabel('Steps')
plt.ylabel('Average Episodic Reward')
plt.title('MLC with Reuters')
plt.legend(title='strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()