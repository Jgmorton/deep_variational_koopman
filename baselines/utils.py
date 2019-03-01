import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import progressbar
import pdb
import tensorflow as tf

# Plot predictions against true time evolution
def visualize_predictions(args, sess, net, replay_memory, env, e=0):
	# Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, 2*args.seq_length, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x_test
    u[:] = replay_memory.u_test

    # Find number of times to feed in input
    n_passes = 200//args.batch_size

    # Construct inputs for network
    feed_in = {}
    feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
    feed_in[net.u] = u
    feed_out = net.state_pred
    out = sess.run(feed_out, feed_in)
    preds = out.reshape(args.batch_size, 2*args.seq_length, args.state_dim)[:, :-1]

    
    # # Now run through again to get next set of predictions
    # feed_in = {}
    # feed_in[net.z1] = z1
    # feed_in[net.generative] = True
    # feed_in[net.u] = u[:, (args.seq_length-1):-1]

    # feed_out = net.state_pred
    # out = sess.run(feed_out, feed_in)
    # x_pred_new = out.reshape(args.batch_size, args.seq_length, args.state_dim)[:, 1:]
    # preds = np.concatenate((x_pred, x_pred_new), axis=1)

    # Find mean, max, and min of predictions
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    pred_min = np.amin(preds, axis=0)
    pred_max = np.amax(preds, axis=0)

    diffs = np.linalg.norm((preds[:, :args.seq_length] - sess.run(net.shift))/sess.run(net.scale) - x[0, :args.seq_length], axis=(1, 2))
    best_pred = np.argmin(diffs)
    worst_pred = np.argmax(diffs)
        
    # Plot different quantities
    x = x*sess.run(net.scale) + sess.run(net.shift)

    # Find indices for random predicted trajectories to plot
    ind0 = best_pred
    ind1 = worst_pred

    # Plot values
    hi_idx = 2*args.seq_length
    lo_idx = 0
    plt.close()
    f, axs = plt.subplots(args.state_dim, sharex=True, figsize=(15, 15))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(args.state_dim):
        axs[i].plot(range(1, hi_idx-lo_idx), x[0, lo_idx:-1, i], 'k')
        axs[i].plot(range(1, hi_idx-lo_idx), preds[ind0, lo_idx:, i], 'r')
        axs[i].plot(range(1, hi_idx-lo_idx), preds[ind1, lo_idx:, i], 'g')
        axs[i].plot(range(1, hi_idx-lo_idx), pred_mean[lo_idx:, i], 'b')
        axs[i].fill_between(range(1, hi_idx-lo_idx), pred_min[lo_idx:, i], pred_max[lo_idx:, i], facecolor='blue', alpha=0.5)
        # axs[i].set_ylabel(r"$\cos(\theta)$")
        axs[i].set_ylim([np.amin(x[0, lo_idx:, i])-0.2, np.amax(x[0, lo_idx:, i]) + 0.2])

    plt.xlabel('Time Step')
    plt.xlim([1, hi_idx-1-lo_idx])
    plt.savefig('bf_predictions/predictions_' + str(e) + '.png')

# Function to generate predictions with trained network
def generate_bf_predictions(args, sess, net):
    # Load data from file
    if args.domain_name == 'Pendulum-v0':
        filename = '../results/pendulum_data_64.h5'
    elif args.domain_name == 'CartPole-v1':
        filename = '../results/cartpole_data_64.h5'
    elif args.domain_name == 'Acrobot-v1':
        filename = '../results/acrobot_data_64.h5'
    f = h5py.File(filename, 'r')
    x_true = f['states'].value.reshape(-1, 2*args.seq_length, args.state_dim)
    u_true = f['actions'].value.reshape(-1, 2*args.seq_length-1, args.action_dim)

    # Normalize data
    x_true = (x_true - sess.run(net.shift))/sess.run(net.scale)
    u_true = (u_true - sess.run(net.shift_u))/sess.run(net.scale_u)

    # Initialize array to store predictions
    n_trials = len(x_true)
    states_pred = np.zeros((n_trials, args.batch_size, 2*args.seq_length, args.state_dim))

    # Define progress bar
    bar = progressbar.ProgressBar(maxval=n_trials).start()

    # Loop through each trajectory and generate predictions
    for i in range(len(x_true)):
        # Get inputs (test trajectory that is twice the size of a standard sequence)
        x = np.zeros((args.batch_size, 2*args.seq_length, args.state_dim), dtype=np.float32)
        u = np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32)
        x[:] = x_true[i]
        u[:] = u_true[i]

        # Construct inputs for network
        feed_in = {}
        feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
        feed_in[net.u] = u
        feed_out = net.state_pred
        out = sess.run(feed_out, feed_in)
        preds = out.reshape(args.batch_size, 2*args.seq_length, args.state_dim)

        # # Now run through again to get next set of predictions
        # feed_in = {}
        # feed_in[net.z1] = z1
        # feed_in[net.generative] = True
        # feed_in[net.u] = u[:, (args.seq_length-1):-1]

        # feed_out = net.state_pred
        # out = sess.run(feed_out, feed_in)
        # x_pred_new = out.reshape(args.batch_size, args.seq_length, args.state_dim)[:, 1:]
        # preds = np.concatenate((x_pred, x_pred_new), axis=1)

        states_pred[i] = preds
        bar.update(i)
    bar.finish()
    f = h5py.File('../results/bf_predictions_acrobot_new.h5', 'w')
    f['states_pred'] = states_pred
    f.close()

# Plot predictions against true time evolution
def visualize_rnn_predictions(args, sess, net, replay_memory, env, e=0):
    # Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, 2*args.seq_length, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x_test
    u[:] = replay_memory.u_test

    # Construct inputs for network
    feed_in = {}
    feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
    feed_in[net.u] = u
    feed_out = net.state_pred
    out = sess.run(feed_out, feed_in)
    preds = out.reshape(args.batch_size, 2*args.seq_length, args.state_dim)[:, :-1]

    
    # # Now run through again to get next set of predictions
    # feed_in = {}
    # feed_in[net.z1] = z1
    # feed_in[net.u] = u[:, (args.seq_length-1):-1]
    # feed_in[net.rnn_init_state] = state

    # feed_out = net.state_pred
    # out = sess.run(feed_out, feed_in)
    # x_pred_new = out.reshape(args.batch_size, args.seq_length, args.state_dim)[:, 1:]
    # preds = np.concatenate((x_pred, x_pred_new), axis=1)

    # Find mean, max, and min of predictions
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    pred_min = np.amin(preds, axis=0)
    pred_max = np.amax(preds, axis=0)

    diffs = np.linalg.norm((preds[:, :args.seq_length] - sess.run(net.shift))/sess.run(net.scale) - x[0, :args.seq_length], axis=(1, 2))
    best_pred = np.argmin(diffs)
    worst_pred = np.argmax(diffs)
        
    # Plot different quantities
    x = x*sess.run(net.scale) + sess.run(net.shift)

    # Find indices for random predicted trajectories to plot
    ind0 = best_pred
    ind1 = worst_pred

    # Plot values
    hi_idx = 2*args.seq_length
    lo_idx = 0
    plt.close()
    f, axs = plt.subplots(args.state_dim, sharex=True, figsize=(15, 15))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for i in range(args.state_dim):
        axs[i].plot(range(1, hi_idx-lo_idx), x[0, lo_idx:-1, i], 'k')
        axs[i].plot(range(1, hi_idx-lo_idx), preds[ind0, lo_idx:, i], 'r')
        axs[i].plot(range(1, hi_idx-lo_idx), preds[ind1, lo_idx:, i], 'g')
        axs[i].plot(range(1, hi_idx-lo_idx), pred_mean[lo_idx:, i], 'b')
        axs[i].fill_between(range(1, hi_idx-lo_idx), pred_min[lo_idx:, i], pred_max[lo_idx:, i], facecolor='blue', alpha=0.5)
        # axs[i].set_ylabel(r"$\cos(\theta)$")
        axs[i].set_ylim([np.amin(x[0, lo_idx:, i])-0.2, np.amax(x[0, lo_idx:, i]) + 0.2])

    plt.xlabel('Time Step')
    plt.xlim([1, hi_idx-1-lo_idx])
    plt.savefig('rnn_predictions/predictions_' + str(e) + '.png')



# Function to generate predictions with trained network
def generate_rnn_predictions(args, sess, net):
    # Load data from file
    if args.domain_name == 'Pendulum-v0':
        filename = '../results/pendulum_data_64.h5'
    elif args.domain_name == 'CartPole-v1':
        filename = '../results/cartpole_data_64.h5'
    elif args.domain_name == 'Acrobot-v1':
        filename = '../results/acrobot_data_64.h5'
    f = h5py.File(filename, 'r')
    x_true = f['states'][()].reshape(-1, 2*args.seq_length, args.state_dim)
    u_true = f['actions'][()].reshape(-1, 2*args.seq_length-1, args.action_dim)

    # Normalize data
    x_true = (x_true - sess.run(net.shift))/sess.run(net.scale)
    u_true = (u_true - sess.run(net.shift_u))/sess.run(net.scale_u)

    # Initialize array to store predictions
    n_trials = len(x_true)
    states_pred = np.zeros((n_trials, args.batch_size, 2*args.seq_length-1, args.state_dim))

    # Define progress bar
    bar = progressbar.ProgressBar(maxval=n_trials).start()

    # Loop through each trajectory and generate predictions
    for i in range(len(x_true)):
        # Get inputs (test trajectory that is twice the size of a standard sequence)
        x = np.zeros((args.batch_size, 2*args.seq_length, args.state_dim), dtype=np.float32)
        u = np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32)
        x[:] = x_true[i]
        u[:] = u_true[i]

        # Construct inputs for network
        feed_in = {}
        feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
        feed_in[net.u] = u
        feed_out = net.state_pred
        out = sess.run(feed_out, feed_in)
        preds = out.reshape(args.batch_size, 2*args.seq_length, args.state_dim)[:, :-1]
        # z1 = out[1].reshape(args.batch_size, args.seq_length, args.code_dim)[:, -1]
        # state = out[2]

        
        # # Now run through again to get next set of predictions
        # feed_in = {}
        # feed_in[net.z1] = z1
        # feed_in[net.u] = u[:, (args.seq_length-1):-1]
        # feed_in[net.rnn_init_state] = state

        # feed_out = net.state_pred
        # out = sess.run(feed_out, feed_in)
        # x_pred_new = out.reshape(args.batch_size, args.seq_length, args.state_dim)[:, 1:]
        # preds = np.concatenate((x_pred, x_pred_new), axis=1)

        states_pred[i] = preds
        bar.update(i)
    bar.finish()
    f = h5py.File('../results/rnn_predictions_cartpole_future.h5', 'w')
    f['states_pred'] = states_pred
    f.close()

# Function to generate predictions with trained network
def generate_mlp_predictions(args, sess, net, saver):
    # Load data from file
    if args.domain_name == 'Pendulum-v0':
        filename = '../results/pendulum_data_128.h5'
        model_names = ['mlp_pendulum_1.ckpt-28',
                            'mlp_pendulum_2.ckpt-27',
                            'mlp_pendulum_3.ckpt-32',
                            'mlp_pendulum_4.ckpt-39',
                            'mlp_pendulum_5.ckpt-39',
                            'mlp_pendulum_6.ckpt-33',
                            'mlp_pendulum_7.ckpt-28',
                            'mlp_pendulum_8.ckpt-39',
                            'mlp_pendulum_9.ckpt-34',
                            'mlp_pendulum_10.ckpt-25']
    elif args.domain_name == 'CartPole-v1':
        filename = '../results/cartpole_data_128.h5'
        model_names = ['mlp_cartpole_1.ckpt-29',
                            'mlp_cartpole_2.ckpt-27',
                            'mlp_cartpole_3.ckpt-29',
                            'mlp_cartpole_4.ckpt-26',
                            'mlp_cartpole_5.ckpt-27',
                            'mlp_cartpole_6.ckpt-28',
                            'mlp_cartpole_7.ckpt-33',
                            'mlp_cartpole_8.ckpt-29',
                            'mlp_cartpole_9.ckpt-30',
                            'mlp_cartpole_10.ckpt-35']
    elif args.domain_name == 'Acrobot-v1':
        filename = '../results/acrobot_data_128.h5'
        model_names = ['mlp_acrobot_1.ckpt-36',
                            'mlp_acrobot_2.ckpt-31',
                            'mlp_acrobot_3.ckpt-34',
                            'mlp_acrobot_4.ckpt-37',
                            'mlp_acrobot_5.ckpt-48',
                            'mlp_acrobot_6.ckpt-38',
                            'mlp_acrobot_7.ckpt-44',
                            'mlp_acrobot_8.ckpt-39',
                            'mlp_acrobot_9.ckpt-40',
                            'mlp_acrobot_10.ckpt-49']
    f = h5py.File(filename, 'r')
    x_true = f['states'][()]
    u_true = f['actions'][()]

    # Reshape data
    x_dims = x_true.shape
    tsteps = x_dims[2]
    x_true = x_true.reshape(-1, tsteps, args.state_dim)
    u_true = u_true.reshape(-1, tsteps-1, args.action_dim)

    # Only use second half of data (we are only making predictions)
    x_true = x_true[:, tsteps//2:]
    u_true = u_true[:, tsteps//2:]

    # Normalize data
    # x_true = (x_true - sess.run(net.shift))/sess.run(net.scale)
    # u_true = (u_true - sess.run(net.shift_u))/sess.run(net.scale_u)

    # Define prediction horizon
    H = 128

    # Initialize array to store predictions
    n_trials = len(x_true)
    n_models = len(model_names)
    states_pred = np.zeros((n_trials, n_models, H, args.state_dim))

    # Find number of passes required to evaluate all scenarios
    n_passes = n_trials//(args.seq_length-1)//args.batch_size + 1
    n = (args.seq_length-1)*args.batch_size # Define since it will be used often
    # pdb.set_trace()

    # Construct array of initial inputs to the network
    x_in = np.zeros((n_passes*n, args.state_dim))
    u_in = np.zeros((n_passes*n, tsteps//2-1, args.action_dim))

    # Loop through each model and generate predictions
    for i in range(n_models):
        saver.restore(sess, os.path.join(args.save_dir, model_names[i]))
        print('restored model', i)

        x_in[:n_trials] = (x_true[:, 0] - sess.run(net.shift))/sess.run(net.scale)
        u_in[:n_trials] = (u_true - sess.run(net.shift_u))/sess.run(net.scale_u)

        # Initialize array to hold predictions
        preds = np.zeros((n_passes*n, H, args.state_dim))
        preds[:, 0] = x_in*sess.run(net.scale) + sess.run(net.shift)

        # Loop through and make predictions across all trials
        for j in range(n_passes):
            for t in range(H-1):
                # Construct inputs for network
                if t == 0:
                    x_out = x_in[j*n:(j+1)*n]
                feed_in = {}
                feed_in[net.x] = np.concatenate((x_out.reshape(args.batch_size, args.seq_length-1, args.state_dim), np.zeros((args.batch_size, 1, args.state_dim))), axis=1)
                feed_in[net.u] = u_in[j*n:(j+1)*n, t].reshape(args.batch_size, args.seq_length-1, args.action_dim)

                # Find loss and perform training operation
                feed_out = net.preds
                out = sess.run(feed_out, feed_in)
                x_out = (out - sess.run(net.shift))/sess.run(net.scale)
                preds[j*n:(j+1)*n, t+1] = out.reshape(-1, args.state_dim)

        # Save predictions to array
        preds = preds[:n_trials]
        states_pred[:, i] = preds

    f = h5py.File('../results/mlp_predictions_cartpole_128.h5', 'w')
    f['states_pred'] = states_pred
    f.close()
