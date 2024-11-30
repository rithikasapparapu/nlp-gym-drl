from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
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

total_timesteps = int(1e5)  # Increased total timesteps for better learning
num_iterations = 100  # Number of training iterations

for i in range(num_iterations):
    model.learn(total_timesteps=total_timesteps // num_iterations, reset_num_timesteps=False)
    print(f"Iteration {i+1}/{num_iterations} completed")
    accuracy = eval_model(env, model, val_pool)
    print(f"Validation Accuracy: {accuracy}")