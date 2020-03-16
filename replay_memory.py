import h5py
import math
import numpy as np
import random
import progressbar
import pdb
import tensorflow as tf

# Class to load and preprocess data
class ReplayMemory():
    def __init__(self, args, env, model):
        """Constructs object to hold and update training/validation data.
        Args:
            args: Various arguments and specifications
            shift: Shift of state values for normalization
            scale: Scaling of state values for normalization
            shift_u: Shift of action values for normalization
            scale_u: Scaling of action values for normalization
            env: Simulation environment
            net: Neural network dynamics model
            sess: TensorFlow session
        """
        self.batch_size = args.batch_size
        self.seq_length = 2*args.seq_length
        self.val_frac = args.val_frac
        self.shift_x = np.array(model.shift_x.value())
        self.scale_x = np.array(model.scale_x.value())
        self.shift_u = np.array(model.shift_u.value())
        self.scale_u = np.array(model.scale_u.value())
        self.env = env
        self.model = model

        print('validation fraction: ', self.val_frac)

        print("generating data...")
        self._generate_data(args)

    def _generate_data(self, args):
        """Load data from environment
        Args:
            args: Various arguments and specifications
        """
        # Initialize array to hold states and actions
        x = np.zeros((args.n_trials, args.n_subseq, self.seq_length, args.state_dim), dtype=np.float32)
        u = np.zeros((args.n_trials, args.n_subseq, self.seq_length-1, args.action_dim), dtype=np.float32)

        trial_states = np.zeros((args.n_trials, args.trial_len, args.state_dim))
        trial_actions = np.zeros((args.n_trials, args.trial_len-1, args.action_dim))

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=args.n_trials).start()

        # Define array for dividing trials into subsequences
        stagger = (args.trial_len - self.seq_length)/args.n_subseq
        self.start_idxs = np.linspace(0, stagger*args.n_subseq, args.n_subseq)

        # Loop through episodes
        for i in range(args.n_trials):
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args.trial_len, args.state_dim), dtype=np.float32)
            u_trial = np.zeros((args.trial_len-1, args.action_dim), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[0] = self.env.reset()
            for t in range(1, args.trial_len):
                action = self.env.action_space.sample()  
                u_trial[t-1] = action
                step_info = self.env.step(action)
                x_trial[t] = np.squeeze(step_info[0])

            trial_states[i] = x_trial
            trial_actions[i] = u_trial

            # Divide into subsequences
            for j in range(args.n_subseq):
                x[i, j] = x_trial[int(self.start_idxs[j]):(int(self.start_idxs[j])+self.seq_length)]
                u[i, j] = u_trial[int(self.start_idxs[j]):(int(self.start_idxs[j])+self.seq_length-1)]
            bar.update(i)
        bar.finish()


        # Generate test scenario that is double the length of standard sequences
        self.x_test = np.zeros((2*args.seq_length, args.state_dim), dtype=np.float32)
        self.u_test = np.zeros((2*args.seq_length-1, args.action_dim), dtype=np.float32)
        self.x_test[0] = self.env.reset()
        for t in range(1, 2*args.seq_length):
            action = self.env.action_space.sample()
            self.u_test[t-1] = action
            step_info = self.env.step(action)
            self.x_test[t] = np.squeeze(step_info[0])

        # Reshape and trim data sets
        x = x.reshape(-1, self.seq_length, args.state_dim)
        u = u.reshape(-1, self.seq_length-1, args.action_dim)
        len_x = int(np.floor(len(x)/args.batch_size)*args.batch_size)
        x = x[:len_x]
        u = u[:len_x]

        # Shuffle data before splitting into train/val
        print('shuffling...')
        p = np.random.permutation(len(x))
        x = x[p]
        u = u[p]

        print('creating splits...')
        # Compute number of batches
        self.n_batches = len(x)//self.batch_size
        self.n_batches_val = int(math.floor(self.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print('num training batches: ', self.n_batches_train)
        print('num validation batches: ', self.n_batches_val)

        # Divide into train and validation datasets
        x_val = x[self.n_batches_train*self.batch_size:]
        u_val = u[self.n_batches_train*self.batch_size:]
        x = x[:self.n_batches_train*self.batch_size]
        u = u[:self.n_batches_train*self.batch_size]

        # Create train and val datasets
        self.train_ds = tf.data.Dataset.from_tensor_slices(
            (x, u)).shuffle(self.n_batches_train, reshuffle_each_iteration=True).batch(self.batch_size)

        self.val_ds = tf.data.Dataset.from_tensor_slices(
            (x_val, u_val)).shuffle(self.n_batches_val, reshuffle_each_iteration=True).batch(self.batch_size)

        print('shifting/scaling data...')
        # Find means and std if not initialized to anything
        if np.sum(self.scale_x) == 0.0:
            self.shift_x = np.mean(x[:self.n_batches_train], axis=(0, 1))
            self.scale_x = np.std(x[:self.n_batches_train], axis=(0, 1))
            self.shift_u = np.mean(u[:self.n_batches_train], axis=(0, 1))
            self.scale_u = np.std(u[:self.n_batches_train], axis=(0, 1))

            # Remove very small scale values
            self.scale_x[self.scale_x < 1e-6] = 1.0

            # Set u norm params to be 0, 1 for pendulum environment
            if args.domain_name == 'Pendulum-v0':
                self.shift_u = np.zeros_like(self.shift_u)
                self.scale_u = np.ones_like(self.scale_u)

    def update_data(self, x_new, u_new):
        """Update training/validation data
        Args:
            x_new: New state values
            u_new: New control inputs
        """
        # First permute data
        p = np.random.permutation(len(x_new))
        x_new = x_new[p]
        u_new = u_new[p]

        # Divide new data into training and validation components
        n_seq_val = max(int(math.floor(self.val_frac * len(x_new))), 1)
        n_seq_train = len(x_new) - n_seq_val
        x_new_val = x_new[n_seq_train:]
        u_new_val = u_new[n_seq_train:]
        x_new = x_new[:n_seq_train]
        u_new = u_new[:n_seq_train]

        # Create datasets for new data
        new_train_ds = tf.data.Dataset.from_tensor_slices(
            (x_new, u_new)).shuffle(n_seq_train, reshuffle_each_iteration=True).batch(self.batch_size)

        new_val_ds = tf.data.Dataset.from_tensor_slices(
            (x_new_val, u_new_val)).shuffle(self.n_batches_val, reshuffle_each_iteration=True).batch(self.batch_size)

        # Now update training and validation data
        self.train_ds.concatenate(new_train_ds)
        self.val_ds.concatenate(new_val_ds)

        # Update sizes of train and val sets
        self.n_batches_train += len(x_new)//self.batch_size
        self.n_batches_val += len(x_new_val)//self.batch_size


