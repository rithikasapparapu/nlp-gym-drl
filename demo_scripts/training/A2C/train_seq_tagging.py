from nlp_gym.data_pools.custom_seq_tagging_pools import UDPosTagggingPool
from nlp_gym.envs.seq_tagging.env import SeqTagEnv
from nlp_gym.envs.seq_tagging.reward import EntityF1Score
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from rich import print

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
data_pool = UDPosTagggingPool.prepare(split="train")

# reward function
reward_fn = EntityF1Score(dense=True, average="micro")

# seq tag env
env = SeqTagEnv(data_pool.labels(), reward_function=reward_fn)
for sample, weight in data_pool:
    env.add_sample(sample, weight)

# check the environment
check_env(env, warn=True)

# train an A2C Policy
model = A2C(
    policy=MlpPolicy,
    env=env,
    learning_rate=5e-4,  # Kept the same as in the original DQN model
    n_steps=5,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.25,
    max_grad_norm=0.5,
    alpha=0.99,
    epsilon=1e-5,
    lr_schedule='constant',
    verbose=1,
    policy_kwargs={"net_arch": [100, 100]}  # Kept the same as in the original DQN model
)

total_timesteps = int(1e6)  # Increased total timesteps for better learning
num_iterations = 1000  # Number of training iterations

for i in range(num_iterations):
    model.learn(total_timesteps=total_timesteps // num_iterations, reset_num_timesteps=False)
    print(f"Iteration {i+1}/{num_iterations} completed")
    eval_model(model, env)