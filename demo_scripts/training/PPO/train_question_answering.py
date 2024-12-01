from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
import tqdm
import matplotlib.pyplot as plt
import numpy as np

def eval_model(env, model, pool):
    correctly_answered = 0.0
    for sample, _ in tqdm.tqdm(pool, desc="Evaluating"):
        obs = env.reset(sample)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

        if info["selected_choice"] == sample.answer:
            correctly_answered += 1

    return correctly_answered/len(pool)

# data pool
data_pool = QASC.prepare(split="train")
val_pool = QASC.prepare(split="val")

# featurizer
featurizer = InformedFeaturizer()

# seq tag env
env = QAEnv(observation_featurizer=featurizer)
for sample, weight in data_pool:
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
    policy_kwargs={"net_arch": [dict(pi=[64, 64], vf=[64, 64])]}
)

steps = []
rewards = []
current_step = 0  # Initialize step counter

total_timesteps = int(1e6)  # Adjust this based on your needs
timesteps_per_iteration = total_timesteps // 1000

for i in range(1000):  # Number of training iterations
    model.learn(total_timesteps=timesteps_per_iteration)
    current_step += timesteps_per_iteration

    # Evaluate and store metrics
    reward = eval_model(env, model, val_pool)
    steps.append(current_step)
    rewards.append(reward)

    print(f"Iteration {i + 1} completed")
    print(f"Validation Accuracy: {reward}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, label='PPO', color='blue')

plt.xlabel('Steps')
plt.ylabel('Episodic Total Reward')
plt.title('QA with QASC')
plt.legend(title='strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()