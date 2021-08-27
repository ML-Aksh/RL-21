# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns

def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)

"""

def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    import math
    assert len(next_obs.shape) == len(act.shape) == 2
    x, theta = next_obs[:, 0], next_obs[:, 2]
    diff = (0 - theta)
    a = diff  % (2 * math.pi)
    b = -diff  % (2 * math.pi)
    intermediary_tensor = torch.minimum(a,b)
    exponentiated_tensor = torch.exp(-intermediary_tensor)
    # print('b')
    # print(theta)
    # print(exponentiated_tensor)
    # print(intermediary_tensor)
    # print(torch.exp(intermediary_tensor))
    # print((~termination_fns.cartpole(act, next_obs)).float())
    return (torch.unsqueeze(exponentiated_tensor, 1))
    # return(torch.minimum(a,b))
    # print(diff)
    # return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)

"""
def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    import numpy as np
    assert len(next_obs.shape) == len(act.shape) == 2
    th, thdot = next_obs[:, 0], next_obs[:, 1]
    u = act[:, 0]

    def angle_normalize_func(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    costs = angle_normalize_func(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return (torch.unsqueeze(-costs, 1))    	

def mountain_car(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    import math
    import numpy as np
    position = next_obs[:,0]
    # print(position)
    action = act[:,0]
    done_tensor = termination_fns.mountain_car(act, next_obs).float().view(-1, 1)
    reward = done_tensor * 100.0 - math.pow(action[0], 2) * 0.1 # + 1 * torch.exp(abs(-0.5 - position)).float().view(-1, 1)
    # print(reward)
    return reward

def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act ** 2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)