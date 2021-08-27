import sys
sys.path.insert(1, '/ifs/loni/faculty/dduncan/agarg/RL/')

from IPython import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf
from datetime import datetime
import os


import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import argparse
import gym
import mbrl.util as util

parser = argparse.ArgumentParser(description="Custom RL Analysis Framework")
parser.add_argument('--env', type=str, default="Pendulum-v0", help="Type of Environment to use. ")
parser.add_argument('--reward_fns', type=str, default='inverted_pendulum')
parser.add_argument('--horizon', type=int, default=15, help="Planning Horizon")
parser.add_argument('--trials', type=int, default=10, help="Total Trials")
parser.add_argument('--ensembles', type=int, default=5)
parser.add_argument('--gpu', action="store_true", help="Are you using gpus or not?")
parser.add_argument('--gpu_select', type=str, default="GPU:0", help="GPU you wish to train on")
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

seed = 0
env = gym.make(args.env)
env.seed(seed)
rng = np.random.default_rng(seed=0)

generator = torch.Generator(device=device)
generator.manual_seed(seed)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

# This functions allows the model to evaluate the true rewards given an observation
reward_fn = eval(f"reward_fns.{args.reward_fns}")
# This function allows the model to know if an observation should make the episode end
term_fn = eval(f"termination_fns.{args.reward_fns}")

trial_length = 200
num_trials = args.trials
ensemble_size = args.ensembles

# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the
# environment information
cfg_dict = {
    # dynamics model configuration
    "dynamics_model": {
        "model": {
            "_target_": "mbrl.models.GaussianMLP",
            "in_size": 4,
            "device": device,
            "num_layers": 3,
            "ensemble_size": ensemble_size,
            "hid_size": 200,
            "use_silu": True,
            "in_size": "???",
            "out_size": "???",
            "deterministic": False,
            "propagation_method": "fixed_model"
        }
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": False,
        "target_is_delta": True,
        "normalize": True,
    },
    # these are experiment specific options
    "overrides": {
        "trial_length": trial_length,
        "num_steps": num_trials * trial_length,
        "model_batch_size": 32,
        "validation_ratio": 0.05
    }
}
cfg = omegaconf.OmegaConf.create(cfg_dict)

# Create a 1-D dynamics model for this environment
dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

# Create a gym-like environment to encapsulate the model
model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)

replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)

common_util.rollout_agent_trajectories(
    env,
    trial_length, # initial exploration steps
    planning.RandomAgent(env),
    {}, # keyword arguments to pass to agent.act()
    replay_buffer=replay_buffer,
    trial_length=trial_length
)

print("# samples stored", replay_buffer.num_stored)

agent_cfg = omegaconf.OmegaConf.create({
    # this class evaluates many trajectories and picks the best one
    "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
    "planning_horizon": 15,
    "replan_freq": 1,
    "verbose": False,
    "action_lb": "???",
    "action_ub": "???",
    # this is the optimizer to generate and choose a trajectory
    "optimizer_cfg": {
        "_target_": "mbrl.planning.CEMOptimizer",
        "num_iterations": 5,
        "elite_ratio": 0.1,
        "population_size": 500,
        "alpha": 0.1,
        "device": device,
        "lower_bound": "???",
        "upper_bound": "???",
        "return_mean_elites": True
    }
})

agent = planning.create_trajectory_optim_agent_for_model(
    model_env,
    agent_cfg,
    num_particles=20
)

train_losses = []
val_scores = []

def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
    train_losses.append(tr_loss)
    val_scores.append(val_score.mean().item())

def update_axes(_axs, _text, _trial, _steps_trial, _all_rewards, force_update=False):
    if not force_update and (_steps_trial % 10 != 0):
        return
    # _axs[0].imshow(_frame)
    # _axs[0].set_xticks([])
    # _axs[0].set_yticks([])
    _axs[1].clear()
    _axs[1].set_xlim([0, num_trials + .1])
    _axs[1].set_ylim([0, 200])
    _axs[1].set_xlabel("Trial")
    _axs[1].set_ylabel("Trial reward")
    _axs[1].plot(_all_rewards, 'bs-')
    _text.set_text(f"Trial {_trial + 1}: {_steps_trial} steps")
    display.display(plt.gcf())

model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

all_rewards = [0]
for trial in range(num_trials):
    # print(trial)
    obs = env.reset()
    agent.reset()

    done = False
    total_reward = 0.0
    steps_trial = 0
    # update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)
    # update_axes(axs, ax_text, trial, steps_trial, all_rewards, force_update=True)
    while not done:
        # --------------- Model Training -----------------
        if steps_trial == 0:
            dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats

            dataset_train, dataset_val = replay_buffer.get_iterators(
                batch_size=cfg.overrides.model_batch_size,
                val_ratio=cfg.overrides.validation_ratio,
                train_ensemble=True,
                ensemble_size=ensemble_size,
                shuffle_each_epoch=True,
                bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
            )

            model_trainer.train(
                dataset_train, dataset_val=dataset_val, num_epochs=50, patience=50, callback=train_callback)

        # --- Doing env step using the agent and adding to model dataset ---
        next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(env, obs, agent, {}, replay_buffer)

        # update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)

        obs = next_obs
        total_reward += reward
        steps_trial += 1

        if steps_trial % 20 == 0:
            print(f"Trial: {trial}, Steps: {steps_trial}")
        # print(steps_trial)

        if done:
            print(f"Done: {trial}")
        if steps_trial == trial_length:
            break

    all_rewards.append(total_reward)
    plt.ylabel("Reward")
    plt.xlabel("Trial")
    plt.xlim([0, num_trials + .1])
    plt.plot(all_rewards, 'bs-')
    plt_text = plt.text(150, 25, "")
    display.display(plt.gcf())
    store_location = os.path.join(env_directory, time_started)
    print(all_rewards)
    plt.savefig(store_location)
