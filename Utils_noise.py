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

def run_pets_with_noise(device, env_directory, input_env, reward_fn_input, terminal_fn_input, trials, trial_length_input, ensembles, horizon, input_seed, num_samples, noise_center = 0, noise_threshhold=0):
    print(f"Ensembles: {ensembles}")
    time_started = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
    print("stochastic {}-{}-{}-{}-horizon-{}".format(input_env, trials, trial_length_input, ensembles, horizon))
    seed = input_seed
    if 'cartpole' in input_env.lower():
        import mbrl.env.cartpole_continuous as cartpole_env
        env = cartpole_env.CartPoleEnv()
        env.set_noise(noise_threshhold)
        env.set_noise_center(noise_center)
        print(env.noise)
    else:
        env = gym.make(input_env)
    env.seed(seed)
    rng = np.random.default_rng(seed=input_seed)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fn_input

    # This function allows the model to know if an observation should make the episode end
    term_fn = terminal_fn_input

    trial_length = trial_length_input
    num_trials = trials
    ensemble_size = ensembles

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the
    # environment information
    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "model": {
                "_target_": "mbrl.models.GaussianMLP",
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
        num_samples, # initial exploration steps
        planning.RandomAgent(env),
        {}, # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer,
        trial_length=trial_length
    )

    print("# samples stored", replay_buffer.num_stored)

    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": horizon,
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
        while not done:
            # --------------- Model Training -----------------
            if steps_trial == 0 and trial == 0:
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

            if steps_trial == trial_length:
                break

        all_rewards.append(total_reward)
        plt.ylabel("Reward")
        plt.xlabel("Trial")
        plt.xlim([0, num_trials + .1])
        plt.plot(all_rewards, 'bs-')
        display.display(plt.gcf())
        new_text = f"no-mod-{input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold*10)}"
        store_location_temp = os.path.join(env_directory, 'Images')
        store_location = os.path.join(store_location_temp, new_text)

        plt.savefig(store_location)
        plt.clf()
    print(f"stochastic rewards: {all_rewards}, {input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold*10)}")

    return all_rewards

def run_pets_without_noise(device, env_directory, input_env, reward_fn_input, terminal_fn_input, trials, trial_length_input, ensembles, horizon, input_seed, num_samples, noise_center = 0, noise_threshhold=0):
    time_started = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
    print("deterministic {}-{}-{}-{}".format(input_env, trials, trial_length_input, ensembles, horizon))
    seed = input_seed
    if 'cartpole' in input_env.lower():
        import mbrl.env.cartpole_continuous_noiseless as cartpole_env_noiseless
        env_noiseless = cartpole_env_noiseless.CartPoleEnv()
        import mbrl.env.cartpole_continuous as cartpole_env
        env = cartpole_env.CartPoleEnv()
        env.set_noise(noise_threshhold)
        env.set_noise_center(noise_center)
        print(env.noise)
    else:
        env_noiseless = gym.make(input_env)

    env_noiseless.seed(seed)
    rng_noiseless = np.random.default_rng(seed=input_seed)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env_noiseless.observation_space.shape
    act_shape = env_noiseless.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn_noiseless = reward_fn_input

    # This function allows the model to know if an observation should make the episode end
    term_fn_noiseless = terminal_fn_input

    trial_length = trial_length_input
    num_trials = trials
    ensemble_size = ensembles

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the
    # environment information

    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "model": {
                "_target_": "mbrl.models.GaussianMLP",
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
    dynamics_model_noiseless = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # Create a gym-like environment to encapsulate the model
    model_env_noiseless = models.ModelEnv(env_noiseless, dynamics_model_noiseless, term_fn_noiseless, reward_fn_noiseless, generator=generator)

    replay_buffer_noiseless = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng_noiseless)

    common_util.rollout_agent_trajectories(
        env_noiseless,
        num_samples,  # initial exploration steps
        planning.RandomAgent(env_noiseless),
        {},  # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer_noiseless,
        trial_length=trial_length
    )

    print("# samples stored", replay_buffer_noiseless.num_stored)

    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": horizon,
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

    agent_noiseless = planning.create_trajectory_optim_agent_for_model(
        model_env_noiseless,
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
        _axs[1].clear()
        _axs[1].set_xlim([0, num_trials + .1])
        _axs[1].set_ylim([0, 200])
        _axs[1].set_xlabel("Trial")
        _axs[1].set_ylabel("Trial reward")
        _axs[1].plot(_all_rewards, 'bs-')
        _text.set_text(f"Trial {_trial + 1}: {_steps_trial} steps")
        display.display(plt.gcf())

    model_trainer_noiseless = models.ModelTrainer(dynamics_model_noiseless, optim_lr=1e-3, weight_decay=5e-5)

    all_rewards = [0]
    for trial in range(num_trials):
        # print(trial)
        obs = env.reset()
        agent_noiseless.reset()

        done = False
        total_reward = 0.0
        steps_trial = 0
        while not done:
            # --------------- Model Training -----------------
            if steps_trial == 0:
                dynamics_model_noiseless.update_normalizer(replay_buffer_noiseless.get_all())  # update normalizer stats

                dataset_train, dataset_val = replay_buffer_noiseless.get_iterators(
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    train_ensemble=True,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )

                model_trainer_noiseless.train(
                    dataset_train, dataset_val=dataset_val, num_epochs=50, patience=50, callback=train_callback)

            # --- Doing env step using the agent and adding to model dataset ---
            # env_noiseless --> env
            next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(env, obs, agent_noiseless, {}, replay_buffer_noiseless)

            # update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)

            obs = next_obs
            total_reward += reward
            steps_trial += 1

            if steps_trial == trial_length:
                break

        all_rewards.append(total_reward)
        plt.ylabel("Reward")
        plt.xlabel("Trial")
        plt.xlim([0, num_trials + .1])
        plt.plot(all_rewards, 'bs-')
        display.display(plt.gcf())
        new_text = f"deterministic-{input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold * 1000)}"
        store_location_temp = os.path.join(env_directory, 'Images')
        store_location = os.path.join(store_location_temp, new_text)

        plt.savefig(store_location)
        plt.clf()
    print(
        f"deterministic rewards: {all_rewards}, {input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold * 1000)}")

    all_rewards_noise = all_rewards

    return all_rewards

def run_pets_deterministic(device, env_directory, input_env, reward_fn_input, terminal_fn_input, trials, trial_length_input, ensembles, horizon, input_seed, num_samples, noise_center = 0, noise_threshhold=0):
    time_started = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
    print("deterministic env {}- trials {}- trial_length {}- horizon {}".format(input_env, trials, trial_length_input, horizon))
    seed = input_seed
    if 'cartpole' in input_env.lower():
        import mbrl.env.cartpole_continuous_noiseless as cartpole_env_noiseless
        env_noiseless = cartpole_env_noiseless.CartPoleEnv()
        import mbrl.env.cartpole_continuous as cartpole_env
        env = cartpole_env.CartPoleEnv()
        env.set_noise(noise_threshhold)
        env.set_noise_center(noise_center)
        print(env.noise)
    else:
        env_noiseless = gym.make(input_env)

    env_noiseless.seed(seed)
    rng_noiseless = np.random.default_rng(seed=input_seed)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env_noiseless.observation_space.shape
    act_shape = env_noiseless.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn_noiseless = reward_fn_input

    # This function allows the model to know if an observation should make the episode end
    term_fn_noiseless = terminal_fn_input

    trial_length = trial_length_input
    num_trials = trials
    ensemble_size = ensembles

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the
    # environment information

    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "model": {
                "_target_": "mbrl.models.GaussianMLP",
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
    dynamics_model_noiseless = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # Create a gym-like environment to encapsulate the model
    model_env_noiseless = models.ModelEnv(env_noiseless, dynamics_model_noiseless, term_fn_noiseless, reward_fn_noiseless, generator=generator)

    replay_buffer_noiseless = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng_noiseless)

    common_util.rollout_agent_trajectories(
        env_noiseless,
        # trial_length,  # initial exploration steps
        # num_trajectories,
        num_samples,
        planning.RandomAgent(env_noiseless),
        {},  # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer_noiseless,
        trial_length=trial_length,
        # collect_full_trajectories = True
    )

    print("# samples stored", replay_buffer_noiseless.num_stored)

    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": horizon,
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

    agent_noiseless = planning.create_trajectory_optim_agent_for_model(
        model_env_noiseless,
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
        _axs[1].clear()
        _axs[1].set_xlim([0, num_trials + .1])
        _axs[1].set_ylim([0, 200])
        _axs[1].set_xlabel("Trial")
        _axs[1].set_ylabel("Trial reward")
        _axs[1].plot(_all_rewards, 'bs-')
        _text.set_text(f"Trial {_trial + 1}: {_steps_trial} steps")
        display.display(plt.gcf())

    model_trainer_noiseless = models.ModelTrainer(dynamics_model_noiseless, optim_lr=1e-3, weight_decay=5e-5)

    all_rewards = [0]
    for trial in range(num_trials):
        # print(trial)
        obs = env.reset()
        agent_noiseless.reset()

        done = False
        total_reward = 0.0
        steps_trial = 0
        while not done:

            # --------------- Model Training -----------------
            if steps_trial == 0 and trial == 0:
                dynamics_model_noiseless.update_normalizer(replay_buffer_noiseless.get_all())  # update normalizer stats

                dataset_train, dataset_val = replay_buffer_noiseless.get_iterators(
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    train_ensemble=True,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )

                model_trainer_noiseless.train(
                    dataset_train, dataset_val=dataset_val, num_epochs=50, patience=50, callback=train_callback)

                # model_env_noiseless.set_noise(noise_threshhold)

            # --- Doing env step using the agent and adding to model dataset ---
            # env_noiseless --> env
            next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(env, obs, agent_noiseless, {}, replay_buffer_noiseless)

            # update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)

            obs = next_obs
            total_reward += reward
            steps_trial += 1

            if steps_trial == trial_length:
                break

        all_rewards.append(total_reward)
        plt.ylabel("Reward")
        plt.xlabel("Trial")
        plt.xlim([0, num_trials + .1])
        plt.plot(all_rewards, 'bs-')
        display.display(plt.gcf())
        new_text = f"deterministic-{input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold * 1000)}"
        store_location_temp = os.path.join(env_directory, 'Images')
        store_location = os.path.join(store_location_temp, new_text)

        plt.savefig(store_location)
        plt.clf()
    print(
        f"deterministic planning rewards: {all_rewards}, {input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold * 1000)}")

    all_rewards_noise = all_rewards

    return all_rewards


def run_pets_stochastic(device, env_directory, input_env, reward_fn_input, terminal_fn_input, trials, trial_length_input, ensembles, horizon, input_seed, num_samples, noise_center = 0, noise_threshhold=0):
    print(f"Ensembles: {ensembles}")
    time_started = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
    print("stochastic env {}- trials {}- trial_length {}- horizon {}".format(input_env, trials, trial_length_input, horizon))
    seed = input_seed
    if 'cartpole' in input_env.lower():
        import mbrl.env.cartpole_continuous_noiseless as cartpole_env_noiseless
        env_noiseless = cartpole_env_noiseless.CartPoleEnv()
        import mbrl.env.cartpole_continuous as cartpole_env
        env = cartpole_env.CartPoleEnv()
        env.set_noise(noise_threshhold)
        env.set_noise_center(noise_center)
        print(env.noise)
    else:
        env_noiseless = gym.make(input_env)

    env_noiseless.seed(seed)
    rng_noiseless = np.random.default_rng(seed=input_seed)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env_noiseless.observation_space.shape
    act_shape = env_noiseless.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn_noiseless = reward_fn_input

    # This function allows the model to know if an observation should make the episode end
    term_fn_noiseless = terminal_fn_input

    trial_length = trial_length_input
    num_trials = trials
    ensemble_size = ensembles

    # Everything with "???" indicates an option with a missing value.
    # Our utility functions will fill in these details using the
    # environment information

    cfg_dict = {
        # dynamics model configuration
        "dynamics_model": {
            "model": {
                "_target_": "mbrl.models.GaussianMLP",
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
    dynamics_model_noiseless = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

    # Create a gym-like environment to encapsulate the model
    model_env_noiseless = models.ModelEnv(env_noiseless, dynamics_model_noiseless, term_fn_noiseless, reward_fn_noiseless, generator=generator)

    replay_buffer_noiseless = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng_noiseless)

    common_util.rollout_agent_trajectories(
        env_noiseless,
        # trial_length,  # initial exploration steps
        # num_trajectories,
        num_samples,
        planning.RandomAgent(env_noiseless),
        {},  # keyword arguments to pass to agent.act()
        replay_buffer=replay_buffer_noiseless,
        trial_length=trial_length,
        # collect_full_trajectories = True
    )

    print("# samples stored", replay_buffer_noiseless.num_stored)

    agent_cfg = omegaconf.OmegaConf.create({
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": horizon,
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

    agent_noiseless = planning.create_trajectory_optim_agent_for_model(
        model_env_noiseless,
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
        _axs[1].clear()
        _axs[1].set_xlim([0, num_trials + .1])
        _axs[1].set_ylim([0, 200])
        _axs[1].set_xlabel("Trial")
        _axs[1].set_ylabel("Trial reward")
        _axs[1].plot(_all_rewards, 'bs-')
        _text.set_text(f"Trial {_trial + 1}: {_steps_trial} steps")
        display.display(plt.gcf())

    model_trainer_noiseless = models.ModelTrainer(dynamics_model_noiseless, optim_lr=1e-3, weight_decay=5e-5)

    all_rewards = [0]
    for trial in range(num_trials):
        # print(trial)
        obs = env.reset()
        agent_noiseless.reset()

        done = False
        total_reward = 0.0
        steps_trial = 0
        while not done:

            # --------------- Model Training -----------------
            if steps_trial == 0 and trial == 0:
                dynamics_model_noiseless.update_normalizer(replay_buffer_noiseless.get_all())  # update normalizer stats

                dataset_train, dataset_val = replay_buffer_noiseless.get_iterators(
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    train_ensemble=True,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )

                model_trainer_noiseless.train(
                    dataset_train, dataset_val=dataset_val, num_epochs=50, patience=50, callback=train_callback)

                model_env_noiseless.set_noise(noise_threshhold)

            # --- Doing env step using the agent and adding to model dataset ---
            # env_noiseless --> env
            next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(env, obs, agent_noiseless, {}, replay_buffer_noiseless)

            # update_axes(axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards)

            obs = next_obs
            total_reward += reward
            steps_trial += 1

            if steps_trial == trial_length:
                break

        all_rewards.append(total_reward)
        plt.ylabel("Reward")
        plt.xlabel("Trial")
        plt.xlim([0, num_trials + .1])
        plt.plot(all_rewards, 'bs-')
        display.display(plt.gcf())
        new_text = f"stochastic-{input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold * 1000)}"
        store_location_temp = os.path.join(env_directory, 'Images')
        store_location = os.path.join(store_location_temp, new_text)

        plt.savefig(store_location)
        plt.clf()
    print(
        f"stochastic planning rewards: {all_rewards}, {input_env}-{trials}-{trial_length_input}-{horizon}-{time_started}-noisex10-{int(noise_threshhold * 1000)}")

    all_rewards_noise = all_rewards

    return all_rewards


"""
Pseudo Code (Stochastic)
Initialize Cartpole Environment with Noise  env
Initialize Dynamics Model 1  dynamics_model; {obs_shape, act_shape, standard configuration dict}
Wrap model in gym like environment  model_env; {dynamics_model, terminal_fn, reward_fn}
Instantiate replay_buffer(cfg, obs_shape, act_shape, rng)
Rollout Agent Trajectories(replay buffer, env, …)  stores samples in replay buffer.
Initialize agent (model_env, agent_cfg, …)
Model_trainer (dynamics_model)
For i in range num_trials:
    If steps_trial == 0:
        Normalize replay buffer + get a random set of iterators
        train dynamics model on this dataset. 
        
    obtain next observation, reward, done_flag by making the agent take a step in the environment (Action found 
                                                        via dynamics model, but executed on real environment)
                                                        
Pseudo Code (Deterministic)
Initialize Cartpole Environment with and without Noise  env, env_noiseless
Initialize Dynamics Model 1  dynamics_model_noiseless; {obs_shape, act_shape, standard configuration dict}
Wrap model in gym like environment  model_env_noiseless; {dynamics_model_noiseless, terminal_fn, reward_fn}
Instantiate replay_buffer(cfg, obs_shape, act_shape, rng)
Rollout Agent Trajectories(replay_buffer_noiseless, env, …)  stores samples in replay_buffer_noiseless.
Initialize agent (model_env_noiseless, agent_cfg, …)
Model_trainer (dynamics_model_noiseless)
For i in range num_trials:
    If steps_trial == 0:
        Normalize replay_buffer_noiseless + get a random set of iterators
        train dynamics_model_noiseless on this dataset. 
        
    obtain next observation, reward, done_flag by making the agent take a step in the environment (Action found 
                                                        via dynamics_model_noiseless, but executed on real environment with noise)
    
"""

