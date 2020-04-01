import argparse
import gym
import numpy as np
import os
import progressbar
import random
import tensorflow as tf
import time

from variational_koopman import ReplayMemory, VariationalKoopman, visualize_predictions, perform_rollouts


def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',       type=str,   default='./checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',       type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',      type= str,  default='',         help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name',      type= str,  default='koopman_model', help='name of checkpoint files for saving')
    parser.add_argument('--domain_name',    type= str,  default='Pendulum-v0', help='environment name')

    parser.add_argument('--seq_length',     type=int,   default= 32,        help='sequence length for training')
    parser.add_argument('--mpc_horizon',    type=int,   default= 16,        help='horizon to consider for MPC')
    parser.add_argument('--batch_size',     type=int,   default= 16,        help='minibatch size')
    parser.add_argument('--latent_dim',     type=int,   default= 4,         help='dimensionality of code')

    parser.add_argument('--num_epochs',     type=int,   default= 100,       help='number of epochs')
    parser.add_argument('--learning_rate',  type=float, default= 0.0005,    help='learning rate')
    parser.add_argument('--decay_rate',     type=float, default= 0.5,       help='decay rate for learning rate')
    parser.add_argument('--l2_regularizer', type=float, default= 0.1,       help='regularization for least squares')
    parser.add_argument('--grad_clip',      type=float, default= 5.0,       help='clip gradients at this value')

    parser.add_argument('--n_trials',       type=int,   default= 1000,       help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len',      type=int,   default= 256,       help='number of steps in each trial')
    parser.add_argument('--n_subseq',       type=int,   default= 12,        help='number of subsequences to divide each sequence into')
    parser.add_argument('--kl_weight',      type=float, default= 1.0,       help='weight applied to kl-divergence loss')

    ######################################
    #          Network Params            #
    ######################################
    parser.add_argument('--extractor_size', nargs='+', type=int, default=[32],    help='hidden layer sizes in feature extractor/decoder')
    parser.add_argument('--inference_size', nargs='+', type=int, default=[32],    help='hidden layer sizes in feature inference network')
    parser.add_argument('--prior_size',     nargs='+', type=int, default=[32],    help='hidden layer sizes in prior network')
    parser.add_argument('--rnn_size',       type=int,   default= 64,        help='size of RNN layers')
    parser.add_argument('--transform_size', type=int,   default= 64,        help='size of transform layers')
    parser.add_argument('--reg_weight',     type=float, default= 1e-4,      help='weight applied to regularization losses')
    parser.add_argument('--seed',           type=int,   default= 1,         help='random seed for sampling operations')

    #####################################
    #       Addtitional Options         #
    #####################################
    parser.add_argument('--ilqr',           type=bool,  default= False,     help='whether to perform ilqr with the trained model')
    parser.add_argument('--evaluate',       type=bool,  default= False,     help='whether to evaluate trained network')
    parser.add_argument('--perform_mpc',    type=bool,  default= False,     help='whether to perform MPC instead of training')
    parser.add_argument('--worst_case',     type=bool,  default= False,     help='whether to optimize for worst-case cost')
    parser.add_argument('--gamma',          type=float, default= 1.0,       help='discount factor')
    parser.add_argument('--num_models',     type=int,   default= 5,         help='number of models to use in MPC')

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

    # Define model
    model = VariationalKoopman(args)

    # Begin training
    train(args, model, env)

# Train network
def train(args, model, env):
    # Define train and validation losses
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    pred_loss = tf.keras.metrics.Mean(name='pred_loss')
    kl_loss = tf.keras.metrics.Mean(name='kl_loss')
    val_score = tf.keras.metrics.Mean(name='val_loss')

    # Load from previous save
    if len(args.ckpt_name) > 0:
        model.load_weights(os.path.join(args.save_dir, args.ckpt_name))

    # Generate training data
    replay_memory = ReplayMemory(args, env, model)

    # Assign normalization params to model
    model.shift_x.assign(replay_memory.shift_x)
    model.scale_x.assign(replay_memory.scale_x)
    model.shift_u.assign(replay_memory.shift_u)
    model.scale_u.assign(replay_memory.scale_u)

    #Function to evaluate loss on validation set
    def val_loss(kl_weight):
        # Loop through validation examples and compute loss
        val_score.reset_states()
        for x, u in replay_memory.val_ds:
            loss, _, _ = model.compute_loss(x, u, kl_weight)
            val_score(loss)

        return val_score.result()

    # Set initial learning rate
    print('setting learning rate to ', args.learning_rate)
    lr = args.learning_rate
    optimizer = tf.keras.optimizers.Adam(lr)
    count_decay = 0

    # Define parameters for annealing kl_weight
    anneal_time = 5
    T = anneal_time*replay_memory.n_batches_train
    anneal_count = 0

    # Specify number of times to loop through training procedure
    n_loops = 10 if args.ilqr else 1
    for n in range(1, n_loops):
        # Initialize variable to track validation score over time
        old_score = 1e20

        # Loop through epochs
        for e in range(args.num_epochs):
            visualize_predictions(model, replay_memory, env, e)

            # Count number of batches and reset running total
            b = 0
            train_loss.reset_states()
            pred_loss.reset_states()
            kl_loss.reset_states()
            
            # Loop over batches
            for x, u in replay_memory.train_ds:
                start = time.time()

                # Update kl_weight
                if e < 3 and n == 0:
                    kl_weight = 1e-6
                else:
                    anneal_count += 1
                    kl_weight = min(args.kl_weight, 1e-6 + args.kl_weight*anneal_count/float(T))

                # Perform forward pass and compute gradients
                loss_b, pred_loss_b, kl_loss_b = model.compute_apply_gradients(x, u, optimizer, tf.convert_to_tensor(kl_weight, dtype=tf.float32))
                train_loss(loss_b)
                pred_loss(pred_loss_b)
                kl_loss(kl_loss_b)

                end = time.time()
                b += 1

                # Print loss
                if (e * replay_memory.n_batches_train + b) % 100 == 0 and b > 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                              e, train_loss.result(), end - start))
                    print("{}/{} (epoch {}), pred_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                              e, pred_loss.result(), end - start))
                    print("{}/{} (epoch {}), kl_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                              e, kl_loss.result(), end - start))

                    print('')
                    train_loss.reset_states()
                    pred_loss.reset_states()
                    kl_loss.reset_states()

            # Evaluate loss on validation set
            score = val_loss(kl_weight)
            print('Validation Loss: {0:f}'.format(score))

            # Set learning rate
            if (old_score - score) < -0.01 and (e >= 8 or n > 0):
                count_decay += 1
                lr = args.learning_rate * (args.decay_rate ** count_decay)
                if lr < 2e-5: break
                print('setting learning rate to ', lr)
                optimizer.learning_rate = lr

            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, args.save_name + '.ckpt-' + str(e))
            model.save_weights(checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
        
            old_score = score

            
        # Run trials to evaluate models and generate new training data
        if args.ilqr:
            avg_reward = perform_rollouts(model, env, replay_memory)
            print("Average reward: ", avg_reward)
            print('Number of training batches now is:', replay_memory.n_batches_train)
            if avg_reward > 190.0: break
            
            # Reset learning rate
            count_decay = 2
            lr = args.learning_rate * (args.decay_rate ** count_decay)
            optimizer.learning_rate = lr


if __name__ == '__main__':
    main()
