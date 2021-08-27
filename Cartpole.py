import sys
sys.path.insert(1, '/ifs/loni/faculty/dduncan/agarg/RL/')

import torch
import os
from datetime import datetime
import matplotlib as mpl
from Utils_noise import run_pets_with_noise, run_pets_without_noise, run_pets_deterministic, run_pets_stochastic
# from Utils import run_pets
import numpy as np

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import argparse
import random

parser = argparse.ArgumentParser(description="Custom RL Analysis Framework")
parser.add_argument('--env', type=str, default="cartpole", help="Type of Environment to use. ")
parser.add_argument('--reward_fns', type=str, default='cartpole')
parser.add_argument('--horizon', type=int, default=15, help="cartpole")
parser.add_argument('--trials', type=int, default=10, help="Total Trials")
parser.add_argument('--trial_length', type=int, default=200, help="Number of steps per trial")
parser.add_argument('--ensembles', type=int, default=5)
parser.add_argument('--gpu', action="store_true", help="Are you using gpus or not?")
parser.add_argument('--gpu_select', type=str, default="GPU:0", help="GPU you wish to train on")
parser.add_argument('--conf_intervals', type=int, default=1)
parser.add_argument('--save_loc', type=str,
                    default=r"/ifs/loni/faculty/dduncan/agarg/RL/Results/",
                    help="Directory storing prepared CSV Training Files")
parser.add_argument('--gpu_select_num', type=int, default=0,
                    help="GPU you wish to train on")
parser.add_argument('--num_samples', type=int, default=200)
parser.add_argument('--noise', type=float, default=0)

args = parser.parse_args()
print(args.horizon)
print(args.gpu_select_num)
mpl.rcParams.update({"font.size": 16})
device = f'cuda:{args.gpu_select_num}' if torch.cuda.is_available() else 'cpu'
time_started = datetime.now().strftime("%m_%d_%Y %H:%M:%S")

env_directory = os.path.join(args.save_loc, args.env)


try:
    os.mkdir(env_directory)
except:
    pass

reward_fn_input = reward_fns.cartpole
termination_fn_input = termination_fns.cartpole

fields = list(range(args.trials))
rewards_list_planning = []
rewards_list_ensemble = []
rewards_list_deterministic_ensemble = []

seed = 0

print("RUnning latest script")


"""
for i in range(args.conf_intervals):
    # rewards = run_pets_with_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, 0, args.noise)
    rewards = run_pets_deterministic(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, 0, args.num_samples, args.noise)
    rewards_list_deterministic.append(rewards)
    path_add_on = f"Deterministic {args.env}-{args.horizon}-normal_env-noise{int(10 * args.noise)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_deterministic, delimiter=', ')
"""


"""
for i in range(args.conf_intervals):
    rewards = run_pets_without_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, 5, args.horizon, 0, args.num_samples, noise_center = 0, noise_threshhold = 0.01)
    rewards_list_deterministic_ensemble.append(rewards)
    path_add_on = f"5 dbl determ ensembles {args.env}-{args.horizon}-normal_env-noise{int(1000 * args.noise)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_deterministic_ensemble, delimiter=', ')

for i in range(args.conf_intervals):
    rewards = run_pets_without_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, 5, args.horizon, 0, args.num_samples, noise_center = 0, noise_threshhold = 0.015)
    rewards_list_deterministic_ensemble.append(rewards)
    path_add_on = f"5 dbl determ ensembles {args.env}-{args.horizon}-normal_env-noise{int(1000 * args.noise)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_deterministic_ensemble, delimiter=', ')

"""

"""
for i in range(args.conf_intervals):
    # rewards = run_pets_with_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, 0, args.noise)
    rewards = run_pets_with_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, 5, args.horizon, 0, args.num_samples, noise_center = 0, noise_threshhold = args.noise)
    rewards_list_ensemble.append(rewards)
    path_add_on = f"1 stoch model determ plan {args.env}-{args.horizon}-normal_env-noise{int(1000 * args.noise)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_ensemble, delimiter=', ')

rewards_list_ensemble = []
"""

"""
for i in range(args.conf_intervals):
    # rewards = run_pets_with_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, 0, args.noise)
    rewards = run_pets_with_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, 1, args.horizon, 0, args.num_samples, noise_center = 0, noise_threshhold = 0.015)
    rewards_list_ensemble.append(rewards)
    path_add_on = f"1 stoch model determ plan {args.env}-{args.horizon}-normal_env-noise{int(1000 * 0.015)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_ensemble, delimiter=', ')

rewards_list_ensemble = []

for i in range(args.conf_intervals):
    # rewards = run_pets_with_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, 0, args.noise)
    rewards = run_pets_with_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, 1, args.horizon, 0, args.num_samples, noise_center = 0, noise_threshhold = 0.02)
    rewards_list_ensemble.append(rewards)
    path_add_on = f"1 stoch model determ plan {args.env}-{args.horizon}-normal_env-noise{int(1000 * 0.02)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_ensemble, delimiter=', ')"""

"""   
for i in range(args.conf_intervals):
    rewards = run_pets_without_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, 1, args.horizon, 0, args.num_samples, noise_center = 0, noise_threshhold = args.noise)
    rewards_list_deterministic_ensemble.append(rewards)
    path_add_on = f"5 dbl determ ensembles {args.env}-{args.horizon}-normal_env-noise{int(1000 * args.noise)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_deterministic_ensemble, delimiter=', ')
"""

for i in range(args.conf_intervals):
    # rewards = run_pets_without_noise(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, 0, args.noise)
    rewards = run_pets_stochastic(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, 0, args.num_samples, noise_center = 0, noise_threshhold = args.noise)
    rewards_list_planning.append(rewards)
    path_add_on = f"determ model stoch plan {args.env}-ensembles-{args.ensembles}-{args.horizon}-normal_env-noise{int(1000*args.noise)}-{time_started}.csv"
    CSV_path = os.path.join(env_directory, 'CSV')
    np.savetxt(os.path.join(CSV_path, path_add_on), rewards_list_planning, delimiter=', ')


"""for noise in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for horizons in [10,15,20,25,30]:
        for i in range(args.conf_intervals):
            rewards = run_pets(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials,
                               args.trial_length, args.ensembles, horizons, random.randint(1, 1000), noise)
            # rewards = run_pets(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length, args.ensembles, args.horizon, random.randint(1, 1000), args.noise)
            rewards_list.append(rewards)
            np.savetxt(f"{args.env}-{args.horizon}-{time_started}-normal_env-noise{int(10*noise)}.csv", rewards_list, delimiter=', ')

"""
"""
Adding noise pseudcode:
We can add it to the main original model. 
- add all the way at the beginning
- add at the end only when evaluating
"""

# Stoch Ensemble
# Stoch Planning
# Stoch Single Model
# Double Determ


