from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
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

# train an A2C Policy
model = A2C(
    policy=MlpPolicy,
    env=env,
    learning_rate=7e-4,
    n_steps=5,
    gamma=0.99,
    ent_coef=0.01,
    vf_coef=0.25,
    max_grad_norm=0.5,
    alpha=0.99,
    epsilon=1e-5,
    lr_schedule='constant',
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]}
)

# Lists to store metrics for plotting
steps = []
rewards = []
current_step = 0

total_iterations = int(1e+2)
timesteps_per_iteration = int(1e+2)

for i in range(total_iterations):
    model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)
    current_step += timesteps_per_iteration

    # Evaluate and store metrics
    reward = eval_model(env, model, val_pool)
    steps.append(current_step)
    rewards.append(reward)

    print(f"Iteration {i + 1}/{total_iterations}, Validation Accuracy: {reward}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, label='A2C', color='red')

plt.xlabel('Steps')
plt.ylabel('Episodic Total Reward')
plt.title('QA with QASC (A2C)')
plt.legend(title='strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()