from nlp_gym.data_pools.custom_question_answering_pools import QASC
from nlp_gym.envs.question_answering.env import QAEnv
from nlp_gym.envs.question_answering.featurizer import InformedFeaturizer, SimpleFeaturizer
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO
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
    policy_kwargs={"net_arch": [64, 64]}
)

total_timesteps = int(1e5)  # Increased total timesteps for better learning
num_iterations = 100  # Number of training iterations

for i in range(num_iterations):
    model.learn(total_timesteps=total_timesteps // num_iterations, reset_num_timesteps=False)
    accuracy = eval_model(env, model, val_pool)
    print(f"Iteration {i+1}/{num_iterations}, Validation Accuracy: {accuracy}")