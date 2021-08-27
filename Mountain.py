import sys
sys.path.insert(1, '/ifs/loni/faculty/dduncan/agarg/RL/')

import torch
import os
from datetime import datetime
import matplotlib as mpl
from Utils import run_pets
import numpy as np

import mbrl.env.termination_fns as termination_fns
import argparse
import random

parser = argparse.ArgumentParser(description="Custom RL Analysis Framework")
parser.add_argument('--env', type=str, default="MountainCarContinuous-v0", help="Type of Environment to use. ")
parser.add_argument('--reward_fns', type=str, default='inverted_pendulum')
parser.add_argument('--horizon', type=int, default=15, help="Planning Horizon")
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


def mountain_car_reward(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    import math
    position = next_obs[:, 0]
    action = act[:, 0]
    done_tensor = termination_fns.mountain_car(act, next_obs).float().view(-1, 1)
    reward = done_tensor * 100.0 - math.pow(action[0], 2) * 0.1 # + 1 * torch.exp(abs(-0.5 - position)).float().view(-1, 1)
    return reward


def mountain_car_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2
    position, velocity = next_obs[:, 0], next_obs[:, 1]
    done = (position >= 0.45) & (velocity >= 0)
    done = done[:, None]
    return done


reward_fn_input = mountain_car_reward
termination_fn_input = mountain_car_termination

fields = list(range(args.trials))
rewards_list = []
for i in range(args.conf_intervals):
    rewards = run_pets(device, env_directory, args.env, reward_fn_input, termination_fn_input, args.trials, args.trial_length,
         args.ensembles, args.horizon, random.randint(1, 1000))
    rewards_list.append(rewards)
    np.savetxt(f"{args.env}-{args.horizon}-{time_started}-normal_env.csv", rewards_list, delimiter=', ')


"""
Adding noise pseudcode:
We can add it to the main original model. 
- add all the way at the beginning
- add at the end only when evaluating
"""

