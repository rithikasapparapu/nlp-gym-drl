from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
import tqdm

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

total_timesteps = int(1e5)  # Adjust this based on your needs
for i in range(10):  # Number of training iterations
    model.learn(total_timesteps=total_timesteps // 10)
    print(f"Iteration {i+1} completed")
    print(f"Validation Accuracy: {eval_model(env, model, val_pool)}")