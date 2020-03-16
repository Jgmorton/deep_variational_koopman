import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import progressbar
import pdb
import tensorflow as tf

from controller import iLQR

def visualize_predictions(model, replay_memory, env, e=0):
    """Plot predictions for a system against true time evolution
    Args:
        model: Neural network dynamics model
        replay_memory: Object containing training/validation data
        env: Simulation environment
        e: Current training epoch
    """
	# Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((model.batch_size, 2*model.seq_length, model.state_dim), dtype=np.float32)
    u = np.zeros((model.batch_size, 2*model.seq_length-1, model.action_dim), dtype=np.float32)
    x[:] = replay_memory.x_test
    u[:] = replay_memory.u_test

    # Find number of times to feed in input
    n_passes = 200//model.batch_size

    # Initialize array to hold predictions
    preds = np.zeros((1, 2*model.seq_length-1, model.state_dim))
    for t in range(n_passes):
        _, _, x_pred = model.forward_pass(np.reshape(x, (2*model.batch_size*model.seq_length, model.state_dim)), u)
        x_pred = x_pred[:, :-1]

        preds = np.concatenate((preds, x_pred), axis=0)       
    preds = preds[1:]

    # Find mean, max, and min of predictions
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    pred_min = np.amin(preds, axis=0)
    pred_max = np.amax(preds, axis=0)

    diffs = np.linalg.norm(preds[:, :model.seq_length] - x[0, :model.seq_length], axis=(1, 2))
    best_pred = np.argmin(diffs)
    worst_pred = np.argmax(diffs)
        
    # Plot different quantities
    x = x

    # # Find indices for random predicted trajectories to plot
    ind0 = best_pred
    ind1 = worst_pred

    # Plot values
    plt.close()
    f, axs = plt.subplots(model.state_dim, sharex=True, figsize=(15, 15))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    for i in range(model.state_dim):
        axs[i].plot(range(1, 2*model.seq_length), x[0, :-1, i], 'k')
        axs[i].plot(range(1, 2*model.seq_length), preds[ind0, :, i], 'r')
        axs[i].plot(range(1, 2*model.seq_length), preds[ind1, :, i], 'g')
        axs[i].plot(range(1, 2*model.seq_length), pred_mean[:, i], 'b')
        axs[i].fill_between(range(1, 2*model.seq_length), pred_min[:, i], pred_max[:, i], facecolor='blue', alpha=0.5)
        axs[i].set_ylim([np.amin(x[0, :, i])-0.2, np.amax(x[0, :, i]) + 0.2])
    plt.xlabel('Time Step')
    plt.xlim([1, 2*model.seq_length-1])
    plt.savefig('vk_predictions/predictions_' + str(e) + '.png')
 
def pendulum_cost(states, us, gamma):
    """Define cost function for inverted pendulum
    Args:
        states: Sequence of state values [num_models, N, state_dim]
        us: Sequence of control inputs [N-1, action_dim]
        gamma: Discount factor
    Returns:
        List of (discounted) cost values at each time step
    """
    num_models = len(states)
    N = states.shape[1]
    thetas = np.arctan2(states[:, :, 1], states[:, :, 0])

    # Find cost of states alone, averaged across models
    cost = np.square(thetas) + 0.1*np.square(states[:, :, 2])
    cost[:, :-1] += 0.001*np.square(np.sum(us, axis=1))
    exp_cost = np.mean(cost, axis=0)

    # Return discounted cost
    return [gamma**t*exp_cost[t] for t in range(N)]

def perform_mpc(model, env, render=False, seed=-1, worst_case=False):
    """Function to perform model predictive control
    Args:
        model: Neural network dynamics model
        env: Simulation environment
        render: Whether to render environment during rollout
        replay_memory: Object containing training/validation data
        seed: Random seed
        worst_case: Whether to optimize for worst case  
    Returns:
        Tuple of
            reward: Number of time steps vertical (specific to inverted pendulum environment)
            x_replay: Observed states during rollout
            u_replay: Observed actions during rollout
            cost: Cumulative environment cost
            fall_count: Number of falls from vertical (specific to inverted pendulum environment)
    """
    # Initialize arrays to hold recent states and actions
    x = np.zeros((model.seq_length, model.state_dim), dtype=np.float32)
    u = np.zeros((model.seq_length-1, model.action_dim), dtype=np.float32)

    # Initialize arrays to hold data to be added to replay buffer
    x_replay = np.zeros((model.trial_len, model.state_dim), dtype=np.float32)
    u_replay = np.zeros((model.trial_len-1, model.action_dim), dtype=np.float32)

    # Get initial state and initialize variables to track reward and cost
    new_state = env.reset()
    x = np.concatenate((x[1:], np.expand_dims(new_state, axis=0)), axis=0)
    x_replay[0] = new_state
    reward = 0.0
    cost = 0.0
    sum_cost = 100.0

    # Define ilQR controller
    if model.domain_name == 'Pendulum-v0':
        ilqr = iLQR(model, pendulum_cost, worst_case=worst_case)
    else:
        raise NotImplementedError

    # Set seeds
    if seed >= 0:
        np.random.seed(seed)

    # Define initial action sequence
    us_init = np.random.uniform(-1, 1, (model.mpc_horizon, model.action_dim))

    # Define progress bar
    bar = progressbar.ProgressBar(maxval=model.trial_len).start()

    # Define variables to track whether pendulum is vertical and falls
    vertical = False
    vertical_count = 0
    fall_count = 0
    time_sum = 0.0

    # Loop through time
    for t in range(1, model.trial_len):
        bar.update(t)
        if render: env.render()
        
        # If enough time steps have passed, find action through MPC, otherwise take random action
        if t >= model.seq_length:

            # Construct state and action inputs for network
            x_in = np.tile(x, (model.batch_size, 2, 1))
            x_in = x_in.reshape(model.batch_size*2*model.seq_length, model.state_dim)
            x_in = tf.convert_to_tensor(x_in, dtype=tf.float32)
            u_in = np.tile(u, (model.batch_size, 1, 1))
            u_in = np.concatenate((u_in, np.zeros((model.batch_size, model.seq_length, model.action_dim))), axis=1)
            u_in = tf.convert_to_tensor(u_in, dtype=tf.float32)

            # Find number of times to do pass through network
            n_passes = int(math.ceil(model.num_models/float(model.batch_size)))

            # Initialize arrays to hold set of A, B, and z0 values
            As_full = np.zeros((n_passes*model.batch_size, model.latent_dim, model.latent_dim))
            Bs_full = np.zeros((n_passes*model.batch_size, model.action_dim, model.latent_dim))
            z0s_full = np.zeros((n_passes*model.batch_size, model.latent_dim))

            # Make passes through network to find models and ICs
            for n in range(n_passes):
                zT, A, B = model.get_latent_dynamics(x_in, u_in)

                z0s_full[model.batch_size*n:(model.batch_size*(n+1))] = zT
                As_full[model.batch_size*n:(model.batch_size*(n+1))] = A
                Bs_full[model.batch_size*n:(model.batch_size*(n+1))] = B
                
            As = As_full[:model.num_models]
            Bs = Bs_full[:model.num_models]
            z0s = z0s_full[:model.num_models]

            # Define number of iterations
            n_iterations = 5
            
            # At initial time step perform 100+ iterations to get good initial action sequence
            if t == model.seq_length:
                n_iterations += 99

            # Find action sequence, execute first action, and update initial action sequence
            xs, us, L_opt = ilqr.fit(z0s, us_init, As, Bs, n_iterations = n_iterations)
            action = model.action_max*np.tanh(us[0])
            us_init = np.vstack([us[1:], np.random.uniform(-1, 1, (1, model.action_dim))])

        # If not enough time has passed, take random action
        else:
            action = env.action_space.sample() 
        
        # Update record of actions
        u = np.concatenate((u[1:], np.expand_dims(action, axis=0)), axis=0)
        u_replay[t-1] = action

        # Observe new state and update array
        step_info = env.step(np.array([action]))
        new_state = step_info[0][:, 0]
        x = np.concatenate((x[1:], np.expand_dims(new_state, axis=0)), axis=0)
        x_replay[t] = new_state
        
        # If enough time has passed, extract metrics
        if t >= model.seq_length:
            cost -= step_info[1]
            
            theta = np.arctan2(new_state[1], new_state[0])
            r = 1.0*(np.abs(theta) <= np.pi/8.0)
            reward += r
            
            # Check for falling from vertical
            if r > 0.0 and not vertical:
                vertical = True
                vertical_count = 1
            elif r > 0.0:
                vertical_count += 1
            elif vertical:
                vertical = False
                if vertical_count > 20:
                    fall_count += 1
                vertical_count = 0
                
    bar.finish()
    return reward, x_replay, u_replay, cost[0], fall_count

def perform_rollouts(model, env, replay_memory):
    """Function to perform rollouts with trained model
    Args:
        model: Neural network dynamics model
        env: Simulation environment
        replay_memory: Object containing training/validation data
    Returns:
        Average reward across rollouts
    """
    print('Performing rollouts...')
    n_trials = model.n_trials//10
    rewards = 0.0

    # Initialize arrays to hold data to be added to replay buffer
    x_replay = np.zeros((n_trials, model.n_subseq, replay_memory.seq_length, model.state_dim), dtype=np.float32)
    u_replay = np.zeros((n_trials, model.n_subseq, replay_memory.seq_length-1, model.action_dim), dtype=np.float32)
    for n in range(n_trials):
        # Perform MPC to evaluate model and get new training data
        reward, x_replay_n, u_replay_n, cost_norm, falls = perform_mpc(model, env, worst_case=model.worst_case)

        # Save trial data to file
        print('saving trial data...')
        f = h5py.File('trial_data.h5', 'r+')
        states = f['x'][()]
        actions = f['u'][()]
        states = np.concatenate((states, np.expand_dims(x_replay_n, axis=0)), axis=0)
        actions = np.concatenate((actions, np.expand_dims(u_replay_n, axis=0)), axis=0)
        del f['x']
        del f['u']
        f['x'] = states
        f['u'] = actions
        f.close()
        print('done.')

        # Divide into subsequences
        for j in range(model.n_subseq):
            x_replay[n, j] = x_replay_n[int(replay_memory.start_idxs[j]):(int(replay_memory.start_idxs[j])+replay_memory.seq_length)]
            u_replay[n, j] = u_replay_n[int(replay_memory.start_idxs[j]):(int(replay_memory.start_idxs[j])+replay_memory.seq_length-1)]
        print('trial ' + str(n) + ': ' + str(reward))
        rewards += reward

    # Update training data
    x_replay = x_replay.reshape(n_trials*model.n_subseq, replay_memory.seq_length, model.state_dim)
    u_replay = u_replay.reshape(n_trials*model.n_subseq, replay_memory.seq_length-1, model.action_dim)
    replay_memory.update_data(x_replay, u_replay)

    # Return average reward
    return rewards/n_trials

