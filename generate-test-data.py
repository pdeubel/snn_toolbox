import gym, roboschool
import os

import tensorflow as tf
import numpy as np
import json
from gym import wrappers

def rollout_evaluation(env, model, render=False, timestep_limit=None, random_stream=None):
    """
    If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
    Otherwise, no action noise will be added.
    """

    env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
    rews = []
    t = 0
    ob = env.reset()
    obs = []
    predictions = []
    for _ in range(timestep_limit):
        if render:
            env.render()
        obs.append(ob[None])
        pred = model.predict_on_batch(ob[None])
        predictions.append(pred)
        ac = pred[0]
        try:
            ob, rew, done, _ = env.step(ac)
        except AssertionError:
            # Is thrown when for example ac is a list which has at least one entry with NaN
            raise
        rews.append(rew)
        t += 1

        if done:
            break
    x_test = np.concatenate(obs)
    y_test = np.concatenate(predictions)
    np.savez_compressed('x_test', x_test)
    np.savez_compressed('y_test', y_test)
    return np.array(rews, dtype=np.float32), t


def run_model(model_file_path, model_file, save_directory, record=False):
    # with open(os.path.join(model_file_path, "config.json"), encoding='utf-8') as f:
    #     config = json.load(f)

    #env = gym.make(config['config']['env_id'])
    env = gym.make("RoboschoolAnt-v1")
    env.reset()
    if record:
        env = wrappers.Monitor(env, save_directory, force=True)

    model = tf.keras.models.load_model(os.path.join(model_file_path, model_file))

    try:
        rewards, length = rollout_evaluation(env, model)
    except AssertionError:
        print("The model file provided produces non finite numbers. Stopping.")
        return

    env.close()
    print(rewards)
    print([rewards.sum(), length])

    return [rewards.sum(), length]


# %%

model_file_path = "test-dir/"
model_file_name = "keras-ant.h5"

# Lets store the video file in the same directory as the model file
save_directory = model_file_path

# with Pool(os.cpu_count()) as pool:
#    pool.apply(func=run_model, args=(model_file_path, model_file_name, save_directory, True))

run_model(model_file_path, model_file_name, save_directory, False)