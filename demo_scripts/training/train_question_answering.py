from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
import tqdm
import matplotlib.pyplot as plt
import numpy as np

def eval_model(env, model, pool):
    correctly_answered = 0.0
    for sample, _ in tqdm.tqdm(pool, desc="Evaluating"):
        obs = env.reset(sample)
        state = None
        done = False
        while not done:
            action, state = model.predict(obs)
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

# train a MLP Policy
model = DQN(env=env, policy=DQNPolicy, gamma=0.99, batch_size=32, learning_rate=1e-4,
            double_q=True, exploration_fraction=0.1,
            prioritized_replay=False, policy_kwargs={"layers": [64, 64]},
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
    reward = eval_model(env, model, val_pool)
    steps.append(current_step)
    rewards.append(reward)

    print(f"Iteration {i + 1}/{total_iterations}, Validation Accuracy: {reward}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, label='DQN', color='green')

plt.xlabel('Steps')
plt.ylabel('Episodic Total Reward')
plt.title('QA with QASC (DQN)')
plt.legend(title='strategy')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()