import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import h5py
import tensorflow as tf
import numpy as np
import time
from mlp import MLP
from replay_memory import ReplayMemory
import random
from utils import generate_mlp_predictions

def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           type=str,   default='./mlp_checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',           type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',          type= str,  default='',         help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name',          type= str,  default='mlp_model', help='name of checkpoint files for saving')
    parser.add_argument('--domain_name',        type= str,  default='Pendulum-v0', help='environment name')

    parser.add_argument('--seq_length',         type=int,   default= 16,        help='sequence length for training')
    parser.add_argument('--batch_size',         type=int,   default= 2,         help='minibatch size')
    parser.add_argument('--noise_dim',          type=int,   default= 4,         help='dimensionality of noise vector')

    parser.add_argument('--num_epochs',         type=int,   default= 50,        help='number of epochs')
    parser.add_argument('--learning_rate',      type=float, default= 0.00075,   help='learning rate')
    parser.add_argument('--decay_rate',         type=float, default= 0.75,      help='decay rate for learning rate')
    parser.add_argument('--grad_clip',          type=float, default= 5.0,       help='clip gradients at this value')
    parser.add_argument('--reg_weight',         type=float, default= 1e-4,      help='weight applied to regularization losses')

    parser.add_argument('--n_trials',           type=int,   default= 100,       help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len',          type=int,   default= 128,       help='number of steps in each trial')
    parser.add_argument('--n_subseq',           type=int,   default= 8,         help='number of subsequences to divide each sequence into')
    parser.add_argument('--predict_evolution', type=bool,  default= False,     help='whether to train to predict future evolution')
    parser.add_argument('--network_size', nargs='+', type=int, default=[32],    help='hidden layer sizes in feature extractor/decoder')


    args = parser.parse_args()

    # Set random seed
    random.seed(1)

    # Create environment
    env = gym.make(args.domain_name)

    # Find state and action dimensionality from environment
    args.state_dim = env.observation_space.shape[0]
    if args.domain_name == 'CartPole-v1': args.state_dim += 1 # taking sine and cosine of theta
    args.action_dim = env.action_space.shape[0]
    args.action_max = env.action_space.high[0]

    # Construct model
    net = MLP(args)
    train(args, net, env)

# Train network
def train(args, net, env):
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Load data
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        shift_u = sess.run(net.shift_u)
        scale_u = sess.run(net.scale_u)
        
        generate_mlp_predictions(args, sess, net, saver)
        # replay_memory = ReplayMemory(args, shift, scale, shift_u, scale_u, env, net, sess)
        
        # Store normalization parameters
        sess.run(tf.assign(net.shift, replay_memory.shift_x))
        sess.run(tf.assign(net.scale, replay_memory.scale_x))
        sess.run(tf.assign(net.shift_u, replay_memory.shift_u))
        sess.run(tf.assign(net.scale_u, replay_memory.scale_u))

        #Function to evaluate loss on validation set
        def val_loss():
            replay_memory.reset_batchptr_val()
            loss = 0.0
            for b in range(replay_memory.n_batches_val):
                # Get inputs
                batch_dict = replay_memory.next_batch_val()
                x = batch_dict["states"]
                u = batch_dict['inputs']

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = x
                feed_in[net.u] = u

                # Find loss
                feed_out = net.cost
                cost = sess.run(feed_out, feed_in)
                loss += cost

            return loss/replay_memory.n_batches_val

        # Initialize variable to track validation score over time
        old_score = 1e9
        count_decay = 0
        decay_epochs = []

        # Set initial learning rate and weight on kl divergence
        print('setting learning rate to ', args.learning_rate)
        sess.run(tf.assign(net.learning_rate, args.learning_rate))

        # Loop over epochs
        for e in range(args.num_epochs):
            # visualize_rnn_predictions(args, sess, net, replay_memory, env, e)

            # Initialize loss
            loss = 0.0

            # Evaluate loss on validation set
            score = val_loss()
            print('Validation Loss: {0:f}'.format(score))

            # Set learning rate
            if old_score < score:
                count_decay += 1
                decay_epochs.append(e)
                if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                print('setting learning rate to ', args.learning_rate * (args.decay_rate ** count_decay))
                sess.run(tf.assign(net.learning_rate, args.learning_rate * (args.decay_rate ** count_decay)))
                if args.learning_rate * (args.decay_rate ** count_decay) < 1e-5: break
            old_score = score
            replay_memory.reset_batchptr_train()

            print('learning rate is set to ', args.learning_rate * (args.decay_rate ** count_decay))

            # Loop over batches
            for b in range(replay_memory.n_batches_train):
                start = time.time()

                # Get inputs
                batch_dict = replay_memory.next_batch_train()
                x = batch_dict["states"]
                u = batch_dict['inputs']

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = x
                feed_in[net.u] = u

                # Find loss and perform training operation
                feed_out = [net.cost, net.train]
                out = sess.run(feed_out, feed_in)

                # Update and display cumulative losses
                loss += out[0]

                end = time.time()

                # Print loss
                if (e * replay_memory.n_batches_train + b) % 100 == 0 and b > 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                              e, loss/100., end - start))
                    print('')
                    loss = 0.0

            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, args.save_name + '.ckpt')
            saver.save(sess, checkpoint_path, global_step = e)
            print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    main()
