from nlp_gym.data_pools.custom_multi_label_pools import ReutersDataPool
from nlp_gym.envs.multi_label.env import MultiLabelEnv
from nlp_gym.envs.multi_label.reward import F1RewardFunction
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
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

# train a TRPO Policy
model = TRPO(
    policy=MlpPolicy,
    env=env,
    gamma=0.99,
    timesteps_per_batch=1024,
    max_kl=0.01,
    cg_iters=10,
    lam=0.98,
    entcoeff=0.0,
    cg_damping=0.01,
    vf_stepsize=0.0003,
    vf_iters=3,
    verbose=1,
    policy_kwargs={"net_arch": [200]}
)

total_timesteps = int(1e5)  # Increased total timesteps for better learning
num_iterations = 100  # Number of training iterations

for i in range(num_iterations):
    model.learn(total_timesteps=total_timesteps // num_iterations, reset_num_timesteps=False)
    print(f"Iteration {i+1}/{num_iterations} completed")
    eval_model(model, env)