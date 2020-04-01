import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.layers import Bidirectional, Dense, GRU, GRUCell, InputLayer, LSTM
from tensorflow.keras import Model, Sequential

class VariationalKoopman(Model):
    def __init__(self, args):
        """Constructs Deep Variational Koopman Model 
        Args:
            args: Various arguments and specifications
        """
        super(VariationalKoopman, self).__init__()

        # Define parameters
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length
        self.latent_dim = args.latent_dim
        self.l2_regularizer = args.l2_regularizer
        self.grad_clip = args.grad_clip
        self.domain_name = args.domain_name
        self.action_max = args.action_max
        self.mpc_horizon = args.mpc_horizon
        self.n_trials = args.n_trials
        self.trial_len = args.trial_len
        self.n_subseq = args.n_subseq
        self.gamma = args.gamma
        self.num_models = args.num_models
        self.worst_case = args.worst_case

        # Normalization parameters to be stored
        self.shift_x = tf.Variable(np.zeros(args.state_dim), trainable=False, dtype=tf.float32)
        self.scale_x = tf.Variable(np.zeros(args.state_dim), trainable=False, dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros(args.action_dim), trainable=False, dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros(args.action_dim), trainable=False, dtype=tf.float32)

        # Create networks
        self._create_extractor_network(args)
        self._create_temporal_encoder_network(args)
        self._create_init_inference_network(args)
        self._create_observation_encoder_network(args)
        self._create_inference_network(args)
        self._create_prior_network(args)
        self._create_decoder_network(args)

    def _create_extractor_network(self, args):
        """Create feature extractor
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.extractor_net = Sequential()

        # Add input
        self.extractor_net.add(InputLayer(input_shape=(self.state_dim,)))

        # Loop through elements of feature extractor and define layers
        for es in args.extractor_size:
            self.extractor_net.add(Dense(es, activation='relu'))

        # Last set of weights to map to output
        self.extractor_net.add(Dense(self.latent_dim))

    def _create_temporal_encoder_network(self, args):
        """Create temporal encoder network
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.temporal_encoder_net = Sequential()

        # Add input
        self.temporal_encoder_net.add(InputLayer(input_shape=(self.seq_length, self.latent_dim + self.action_dim)))

        #  Add bidirectional LSTM layer
        bi_lstm = Bidirectional(LSTM(args.rnn_size), merge_mode='concat')
        self.temporal_encoder_net.add(bi_lstm)

        # Add dense layers to transform to temporal encoding
        self.temporal_encoder_net.add(Dense(args.transform_size, activation='relu'))
        self.temporal_encoder_net.add(Dense(self.latent_dim))

    def _create_init_inference_network(self, args):
        """Create inference network for initial observation
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.init_inference_net = Sequential()

        # Add input
        self.init_inference_net.add(InputLayer(input_shape=(2*self.latent_dim,)))

        # Add transformation to initial observation distribution
        self.init_inference_net.add(Dense(args.transform_size, activation='relu'))
        self.init_inference_net.add(Dense(2*self.latent_dim))

    def _create_observation_encoder_network(self, args):
        """Create observation encoder network
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.observation_encoder_net = Sequential()

        # Add input
        self.observation_encoder_net.add(InputLayer(input_shape=(1, self.latent_dim,)))

        # Initialize single-layer GRU network to create observation encodings
        self.observation_encoder_net.add(LSTM(args.rnn_size))

        # Add in dense layers
        self.observation_encoder_net.add(Dense(args.transform_size, activation='relu'))
        self.observation_encoder_net.add(Dense(self.latent_dim))

    def _create_inference_network(self, args):
        """Create inference network
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.inference_net = Sequential()

        # Add input
        self.inference_net.add(InputLayer(input_shape=(3*self.latent_dim + self.action_dim,)))

        # Loop through elements of inference network and add layers
        for ins in args.inference_size:
            self.inference_net.add(Dense(ins, activation='relu'))

        # Last set of weights to map to output
        self.inference_net.add(Dense(2*self.latent_dim))

    def _create_prior_network(self, args):
        """Construct network and generate paramaters for conditional prior distributions
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.prior_net = Sequential()

        # Add inputs
        self.prior_net.add(InputLayer(input_shape=(self.latent_dim + self.action_dim,)))

        # Add dense layers
        for ps in args.prior_size:
            self.prior_net.add(Dense(ps, activation='relu'))

        # Final affine transform to dist params
        self.prior_net.add(Dense(2*self.latent_dim))
    
    def _create_decoder_network(self, args):
        """Create decoder network
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.decoder_net = Sequential()

        # Add input
        self.decoder_net.add(InputLayer(input_shape=(args.latent_dim,)))

        # Loop through elements of decoder network and define layers
        for i in range(len(args.extractor_size)-1, -1, -1):
            self.decoder_net.add(Dense(args.extractor_size[i], activation='tanh'))

        # Last set of weights to map to output
        self.decoder_net.add(Dense(self.state_dim))

    def _gen_sample(self, dist_params):
        """Function to generate samples given distribution parameters
        Args:
            dist_params: Mean and logstd of distribution [batch_size, 2*latent_dim]
        Returns:
            g: Sampled g-value [batch_size, latent_dim]
        """
        g_mean, g_logstd = tf.split(dist_params, [self.latent_dim, self.latent_dim], axis=1)

        # Make standard deviation estimates better conditioned, otherwise could be problem early in training
        g_std = tf.minimum(tf.exp(g_logstd) + 1e-6, 10.0) 
        samples = tf.random.normal([self.batch_size, self.latent_dim])
        g = samples*g_std + g_mean
        return g

    @tf.function
    def _get_features(self, x, u):
        """Find features and temporal encoding given sequence of states and actions
        Args:
            x: State values [2*batch_size*seq_length, state_dim]
            
        Returns:
            features: Feature values [batch_size, 2*seq_length, latent_dim]
            temporal_encoding: Temporal encoding [batch_size, latent_dim]
        """
        # Get fetures from feature extractor and reshape
        features = self.extractor_net(x)
        features = tf.reshape(features, [self.batch_size, 2*self.seq_length, self.latent_dim])

        # Construct input to temporal encoder and get temporal encoding
        padded_u = tf.concat([tf.zeros([self.batch_size, 1, self.action_dim]), u[:, :(self.seq_length-1)]], axis=1)
        temp_enc_input = tf.concat([features[:, :self.seq_length], padded_u], axis=2)
        temporal_encoding = self.temporal_encoder_net(temp_enc_input)

        return features, temporal_encoding

    @tf.function
    def _get_observations(self, features, temporal_encoding, u):
        """Finds inferred distributions over g-values and samples from them
        Args:
            features: Feature values [batch_size, 2*seq_length, latent_dim]
            temporal_encoding: Temporal encoding [batch_size, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            g_vals: Sampled observations [batch_size, seq_length, latent_dim]
            g_dists: Inferred distributions over g-values [batch_size*seq_length, 2*latent_dim]
        """
        # Get inferred distribution over initial observation
        init_obs_input = tf.concat([temporal_encoding, features[:, 0]], axis=1)
        g1_dist = self.init_inference_net(init_obs_input)

        # Infer remaining observations
        g_t = self._gen_sample(g1_dist)

        # Start list of g-distributions and sampled values
        g_vals = tf.TensorArray(tf.float32, self.seq_length)
        g_vals = g_vals.write(0, g_t)
        g_dists = tf.TensorArray(tf.float32, self.seq_length)
        g_dists = g_dists.write(0, g1_dist)


        # Initialize internal state of observation encoder
        self.observation_encoder_net.reset_states()

        # Loop through time
        for t in tf.range(1, self.seq_length):
            # Update observation encoding
            g_enc = self.observation_encoder_net(tf.expand_dims(g_t, axis=1))

            # Get distribution and sample new observations
            g_dist = self.inference_net(tf.concat([features[:, t], u[:, t-1], temporal_encoding, g_enc], axis=1))
            g_t = self._gen_sample(g_dist)

            # Append values to list
            g_vals = g_vals.write(t, g_t)
            g_dists = g_dists.write(t, g_dist)

        # Finally, stack inferred observations
        g_vals = tf.transpose(g_vals.stack(), [1, 0, 2])
        g_dists = tf.reshape(tf.transpose(g_dists.stack(), [1, 0, 2]), [self.batch_size*self.seq_length, 2*self.latent_dim])

        return g_vals, g_dists

    @tf.function
    def _get_prior_dists(self, g_vals, u):
        """Derive conditional prior distributions over observations
        Args:
            g_vals: Sampled observations [batch_size, seq_length, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            g_prior_dists: Prior distributions over g-values [batch_size*seq_length, 2*latent_dim]
        """
        # Construct diagonal unit Gaussian prior params for g1
        g1_prior = tf.concat([tf.zeros([self.batch_size, 1, self.latent_dim]), tf.ones([self.batch_size, 1, self.latent_dim])], axis=2)

        # Construct input to prior network
        gvals_reshape = tf.reshape(g_vals[:, :-1], [self.batch_size*(self.seq_length-1), self.latent_dim])
        u_reshape = tf.reshape(u[:, :(self.seq_length-1)], [self.batch_size*(self.seq_length-1), self.action_dim])
        prior_input = tf.concat([gvals_reshape, u_reshape], axis=1)

        # Find parameters to prior distributions
        prior_params = tf.reshape(self.prior_net(prior_input), [self.batch_size, self.seq_length-1, 2*self.latent_dim])

        # Combine and reshape to get full set of prior distribution parameter values
        g_prior_dists = tf.concat([g1_prior, prior_params], axis=1)

        # Combine and reshape to get full set of prior distribution parameter values
        g_prior_dists = tf.reshape(g_prior_dists, [self.batch_size*self.seq_length, 2*self.latent_dim])

        return g_prior_dists

    @tf.function
    def _derive_dynamics(self, g_vals, u):
        """Perform least squares to get A- and B-matrices and propagate
        Args:
            g_vals: Sampled observations [batch_size, seq_length, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            z_vals: Observations obtained by simulating dynamics [batch_size, seq_length, latent_dim]
            A: Dynamics A-matrix [batch_size, latent_dim, latent_dim]
            B: Dynamics B-matrix [batch_size, latent_dim, action_dim]
        """
        # Define X- and Y-matrices
        X = tf.concat([g_vals[:, :-1], u[:, :(self.seq_length-1)]], axis=2)
        Y = g_vals[:, 1:]

        # Solve for A and B using least-squares
        K = tf.linalg.lstsq(X, Y, l2_regularizer=self.l2_regularizer)
        A = K[:, :self.latent_dim]
        B = K[:, self.latent_dim:]

        # Perform least squares to find A-inverse
        A_inv = tf.linalg.lstsq(Y - tf.matmul(u[:, :(self.seq_length-1)], B), g_vals[:, :-1], l2_regularizer=self.l2_regularizer)
        
        # Get predicted code at final time step
        z_t = g_vals[:, -1]

        # Create recursive predictions for z
        z_t = tf.expand_dims(z_t, axis=1)
        z_vals = tf.TensorArray(tf.float32, self.seq_length)
        z_vals = z_vals.write(self.seq_length-1, z_t)
        for t in tf.range(self.seq_length-2, -1, -1):
            u_t = u[:, t]
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.matmul(z_t - tf.matmul(u_t, B), A_inv)
            z_vals = z_vals.write(t, z_t)

        # Concatenate and flip order
        z_vals = tf.transpose(tf.squeeze(z_vals.stack()), [1, 0, 2])

        return z_vals, A, B

    @tf.function
    def _generate_predictions(self, z1, u, A, B):
        """Generate predictions for how system will evolve given z1, A, and B
        Args:
            z1: Initial latent state [batch_size, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
            A: Dynamics A-matrix [batch_size, latent_dim, latent_dim]
            B: Dynamics B-matrix [batch_size, latent_dim, action_dim]
        Returns:
            z_pred: Observations obtained by simulating dynamics [batch_size, seq_length, latent_dim]
        """
        # Create predictions, simulating forward in time
        z_t = tf.expand_dims(z1, axis=1)
        z_pred = tf.TensorArray(tf.float32, self.seq_length)
        for t in tf.range(self.seq_length, 2*self.seq_length):
            u_t = u[:, t-1]
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.matmul(z_t, A) + tf.matmul(u_t, B)
            z_pred = z_pred.write(t-self.seq_length, z_t)
        z_pred = tf.transpose(tf.squeeze(z_pred.stack()), [1, 0, 2])
        
        return z_pred

    @tf.function
    def _get_cost(self, z_u_t):
        """Get cost associated with a set of states and actions
        Args:
            z_u_t: Latent state and control input at given time step [state_dim+action_dim]
        Returns:
            Scalar-valued cost
        """
        z_t = z_u_t[:, :self.latent_dim]
        u_t = z_u_t[:, self.latent_dim:]
        states = self.decoder_net(z_t)
        if self.domain_name == 'Pendulum-v0':
            return tf.math.square(tf.math.atan2(states[:, 1], states[:, 0])) + 0.1*tf.math.square(states[:, 2]) + 0.001*tf.math.square(tf.squeeze(u_t))
        else:
            raise NotImplementedError

    @tf.function
    def _get_grad_hessian(self, z_u_t):
        """ Get cost, gradient, and Hessian for latent states and actions
        Args:
            z_u_t: Combined latent state and action [batch_size, state_dim+action_dim]
        Returns:
            cost: cost values [batch_size]
            grads: Gradients of cost [batch_size, state_dim+action_dim]
            hessians: Hessians of cost [batch_size, state_dim+action_dim, state_dim+action_dim]
        """ 
        with tf.GradientTape(persistent=True) as hess_tape:
            hess_tape.watch(z_u_t)
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(z_u_t)
                cost = self._get_cost(z_u_t)
            grads = grad_tape.gradient(cost, z_u_t)
        hessians = hess_tape.batch_jacobian(grads, z_u_t)
        return cost, grads, hessians

    @tf.function
    def decode(self, z):
        """ Map latent state to full state through decoder
        Args:
            z: Latent state [Any, latent_dim]
        Returns:
            x: Full state [Any, state_dim]
        """
        return self.decoder_net(z)

    @tf.function
    def forward_pass(self, x, u):
        """Perform forward pass, mapping states and actions to network outputs
        Args:
            x: State values [batch_size, 2*seq_length, state_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            g_prior_dists: Prior distributions over g-values [batch_size*seq_length, 2*latent_dim]
            g_dists: Inferred distributions over g-values [batch_size*seq_length, 2*latent_dim]
            x_pred: State predictions [2*batch_size*seq_length, state_dim]
        """
        # Normalize inputs
        x = (x - self.shift_x)/self.scale_x
        x = tf.reshape(x, [2*self.batch_size*self.seq_length, self.state_dim])
        u = (u - self.shift_u)/self.scale_u

        # Find features and temporal encoding
        features, temporal_encoding = self._get_features(x, u)

        # Get inferred observation distributions and sampled g-values
        g_vals, g_dists = self._get_observations(features, temporal_encoding, u)

        # Derive dynamics and simulate latent dynamics with model
        z_vals, A, B = self._derive_dynamics(g_vals, u)

        # Simulate dynamics forward to get predicted observations
        z_pred = self._generate_predictions(z_vals[:, -1], u, A, B)

        # Concatenate z-values and reshape
        z_vals = tf.concat([z_vals, z_pred], axis=1)
        z_vals = tf.reshape(z_vals, [2*self.batch_size*self.seq_length, self.latent_dim])

        # Find prior distribution over g-values
        g_prior_dists = self._get_prior_dists(g_vals, u)

        # Run z-values through decoder to get state values
        x_pred = self.decoder_net(z_vals)
        x_pred = tf.reshape(x_pred, [self.batch_size, 2*self.seq_length, self.state_dim])

        return g_dists, g_prior_dists, x_pred

    @tf.function
    def compute_loss(self, x, u, kl_weight=1.0):
        """ Compute loss given states, actions, and weight to apply to KL-divergence loss
        Args:
            x: State values [batch_size, 2*seq_length, state_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
            kl_weight: Scalar weight to be applied to KL-divergence component of loss
        Returns:
            loss: Full loss
            pref_loss: Prediction (modeling) component of loss
            kl_loss: KL-divergence component of loss
        """
        g_dists, g_prior_dists, x_pred = self.forward_pass(x, u)

        # First extract mean and std for prior dists, dist over g, and dist over x
        g_prior_mean, g_prior_logstd = tf.split(g_prior_dists, [self.latent_dim, self.latent_dim], axis=1)
        g_prior_std = tf.exp(g_prior_logstd) + 1e-6
        g_mean, g_logstd = tf.split(g_dists, [self.latent_dim, self.latent_dim], axis=1)
        g_std = tf.exp(g_logstd) + 1e-6

        # First component of loss: NLL of observed states
        x_norm = (x - self.shift_x)/self.scale_x
        x_pred_norm = (x_pred - self.shift_x)/self.scale_x
        pred_loss = tf.math.reduce_sum(tf.math.square(x_norm - x_pred_norm))
        pred_loss += 20.0*tf.math.reduce_sum(tf.math.square(x_norm[:, self.seq_length-1] - x_pred_norm[:, self.seq_length-1]))

        # Second component of loss: KLD between approximate posterior and prior
        g_prior_dist = tfp.distributions.Normal(loc=g_prior_mean, scale=g_prior_std)
        g_dist = tfp.distributions.Normal(loc=g_mean, scale=g_std)
        kl_loss = tf.math.reduce_sum(tfp.distributions.kl_divergence(g_dist, g_prior_dist))

        # Sum with regularization losses to form total cost
        loss = pred_loss + kl_weight*kl_loss
        return loss, pred_loss, kl_loss

    @tf.function
    def compute_apply_gradients(self, x, u, optimizer, kl_weight=1.0):
        """ Compute and apply gradients to a model
        Args:
            x: State values [batch_size, 2*seq_length, state_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
            optimizer: TensorFlow optmizer
            kl_weight: Scalar weight to be applied to KL-divergence component of loss
        """
        # Compute loss
        with tf.GradientTape() as tape:
            loss, pred_loss, kl_loss = self.compute_loss(x, u, kl_weight)

        # Find and apply gradients
        grads = tape.gradient(loss, self.trainable_variables)
        grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(self.trainable_variables, grads)]
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss, pred_loss, kl_loss

    @tf.function
    def get_latent_dynamics(self, x, u):
        """Perform forward pass, mapping states and actions to network outputs
        Args:
            x: State values [batch_size, 2*seq_length, state_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            zT: Latent state at time T [batch_size, latent_dim]
            A: Dynamics A-matrix [batch_size, latent_dim, latent_dim]
            B: Dynamics B-matrix [batch_size, latent_dim, action_dim]
        """
        # Normalize inputs
        x = (x - self.shift_x)/self.scale_x
        x = tf.reshape(x, [2*self.batch_size*self.seq_length, self.state_dim])
        u = (u - self.shift_u)/self.scale_u

        # Find features and temporal encoding
        features, temporal_encoding = self._get_features(x, u)

        # Get inferred observation distributions and sampled g-values
        g_vals, g_dists = self._get_observations(features, temporal_encoding, u)

        # Define X- and Y-matrices
        X = tf.concat([g_vals[:, :-1], u[:, :(self.seq_length-1)]], axis=2)
        Y = g_vals[:, 1:]

        # Solve for A and B using least-squares
        K = tf.linalg.lstsq(X, Y, l2_regularizer=self.l2_regularizer)
        A = K[:, :self.latent_dim]
        B = K[:, self.latent_dim:]

        # Extract zT
        zT = g_vals[:, -1]

        return zT, A, B

    @tf.function
    def find_ilqr_params(self, z1, A, B, u_ilqr):
        """Find necessary params to perform iLQR
        Args:
            z1: Initial latent state [batch_size, latent_dim]
            A: Set of A-matrices [batch_size, latent_dim, latent_dim]
            B: Set of B-matrices [batch_size, latent_dim, action_dim]
            u_ilqr: iLQR candidate action sequence [batch_size, mpc_horizon-1, action_dim]
        """
        # Initialize state
        z_t = z1

        # Initialize lists to hold quantities
        L = tf.TensorArray(tf.float32, self.mpc_horizon+1)
        L_x = tf.TensorArray(tf.float32, self.mpc_horizon+1)
        L_u = tf.TensorArray(tf.float32, self.mpc_horizon)
        L_xx = tf.TensorArray(tf.float32, self.mpc_horizon+1)
        L_ux = tf.TensorArray(tf.float32, self.mpc_horizon)
        L_uu = tf.TensorArray(tf.float32, self.mpc_horizon)
        z_pred = tf.TensorArray(tf.float32, self.mpc_horizon+1)
        z_pred = z_pred.write(0, z_t)

        # Loop through time
        for t in tf.range(self.mpc_horizon):
            # Find cost for current state
            z_u_t = tf.concat([z_t, u_ilqr[:, t]], axis=1)
            
            # Find cost, gradient, and Hessian
            l_t, grads, hessians = self._get_grad_hessian(z_u_t)  

            # Separate into individual components
            l_x = grads[:, :self.latent_dim]
            l_u = grads[:, self.latent_dim:]
            l_xx = hessians[:, :self.latent_dim, :self.latent_dim]
            l_ux = hessians[:, self.latent_dim:, :self.latent_dim]
            l_uu = hessians[:, self.latent_dim:, self.latent_dim:]

            # Append to lists
            L = L.write(t, l_t)
            L_x = L_x.write(t, l_x)
            L_u = L_u.write(t, l_u)
            L_xx = L_xx.write(t, l_xx)
            L_ux = L_ux.write(t, l_ux)
            L_uu = L_uu.write(t, l_uu)

            # Find action by passing it through tanh
            u_t = self.action_max*tf.math.tanh(u_ilqr[:, t])
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.squeeze(tf.matmul(tf.expand_dims(z_t, axis=1), A) + tf.matmul(u_t, B))
            z_pred = z_pred.write(t+1, z_t)

        # Find cost and gradients at last time step
        z_u_t = tf.concat([z_t, tf.zeros_like(u_ilqr[:, -1])], axis=1)
        l_T, grads, hessians = self._get_grad_hessian(z_u_t)   
        L = L.write(self.mpc_horizon, l_T)
        L_x = L_x.write(self.mpc_horizon, grads[:, :self.latent_dim])
        L_xx = L_xx.write(self.mpc_horizon, hessians[:, :self.latent_dim, :self.latent_dim])

        # Finally stack into tensors
        L = tf.transpose(L.stack(), [1, 0])
        L_x = tf.transpose(L_x.stack(), [1, 0, 2])
        L_u = tf.transpose(L_u.stack(), [1, 0, 2])
        L_xx = tf.transpose(L_xx.stack(), [1, 0, 2, 3])
        L_ux = tf.transpose(L_ux.stack(), [1, 0, 2, 3])
        L_uu = tf.transpose(L_uu.stack(), [1, 0, 2, 3])
        z_pred = tf.transpose(z_pred.stack(), [1, 0, 2])

        # Return all values
        return L, L_x, L_u, L_xx, L_ux, L_uu, z_pred


