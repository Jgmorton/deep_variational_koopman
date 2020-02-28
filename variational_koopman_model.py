import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Bidirectional, Dense, GRU, GRUCell, InputLayer, LSTM
from tensorflow.keras import Model, Sequential

class VariationalKoopman(Model):
    def __init__(self, args):
        """Constructs Deep Variational Koopman Model 
        Args:
            args: Various arguments and specifications
        """
        super(VariationalKoopman, self).__init__()

        # Normalization parameters to be stored
        self.shift_x = tf.Variable(np.zeros(args.state_dim), trainable=False, name="state_shift", dtype=tf.float32)
        self.scale_x = tf.Variable(np.zeros(args.state_dim), trainable=False, name="state_scale", dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_shift", dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_scale", dtype=tf.float32)

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
        self.extractor_net.add(InputLayer(input_shape=(args.state_dim,)))

        # Loop through elements of feature extractor and define layers
        for es in args.extractor_size:
            self.extractor_net.add(Dense(es, activation='relu'))

        # Last set of weights to map to output
        self.extractor_net.add(Dense(args.latent_dim))

    def _create_temporal_encoder_network(self, args):
        """Create temporal encoder network
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.temporal_encoder_net = Sequential()

        # Add input
        self.temporal_encoder_net.add(InputLayer(input_shape=(args.seq_length, args.latent_dim + args.action_dim)))

        #  Add bidirectional LSTM layer
        bi_lstm = Bidirectional(LSTM(args.rnn_size), merge_mode='concat')
        self.temporal_encoder_net.add(bi_lstm)

        # Add dense layers to transform to temporal encoding
        self.temporal_encoder_net.add(Dense(args.transform_size, activation='relu'))
        self.temporal_encoder_net.add(Dense(args.latent_dim))

    def _create_init_inference_network(self, args):
        """Create inference network for initial observation
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.init_inference_net = Sequential()

        # Add input
        self.init_inference_net.add(InputLayer(input_shape=(2*args.latent_dim,)))

        # Add transformation to initial observation distribution
        self.init_inference_net.add(Dense(args.transform_size, activation='relu'))
        self.init_inference_net.add(Dense(2*args.latent_dim))

    def _create_observation_encoder_network(self, args):
        """Create observation encoder network
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.observation_encoder_net = Sequential()

        # Add input
        self.observation_encoder_net.add(InputLayer(input_shape=(1, args.latent_dim,)))

        # Initialize single-layer GRU network to create observation encodings
        self.observation_encoder_net.add(LSTM(args.rnn_size, input_shape=(1, args.latent_dim)))

        # Add in dense layers
        self.observation_encoder_net.add(Dense(args.transform_size, activation='relu'))
        self.observation_encoder_net.add(Dense(args.latent_dim))

    def _create_inference_network(self, args):
        """Create inference network
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.inference_net = Sequential()

        # Add input
        self.inference_net.add(InputLayer(input_shape=(3*args.latent_dim + args.action_dim,)))

        # Loop through elements of inference network and add layers
        for ins in args.inference_size:
            self.inference_net.add(Dense(ins, activation='relu'))

        # Last set of weights to map to output
        self.inference_net.add(Dense(2*args.latent_dim))

    def _create_prior_network(self, args):
        """Construct network and generate paramaters for conditional prior distributions
        Args:
            args: Various arguments and specifications
        """
        # Initialize network
        self.prior_net = Sequential()

        # Add inputs
        self.prior_net.add(InputLayer(input_shape=(args.latent_dim + args.action_dim,)))

        # Add dense layers
        for ps in args.prior_size:
            self.prior_net.add(Dense(ps, activation='relu'))

        # Final affine transform to dist params
        self.prior_net.add(Dense(2*args.latent_dim))
    
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
        self.decoder_net.add(Dense(args.state_dim))

    def _gen_sample(self, args, dist_params):
        """Function to generate samples given distribution parameters
        Args:
            args: Various arguments and specifications
            dist_params: Mean and logstd of distribution [batch_size, 2*latent_dim]
        Returns:
            g: Sampled g-value [batch_size, latent_dim]
        """
        g_mean, g_logstd = tf.split(dist_params, [args.latent_dim, args.latent_dim], axis=1)

        # Make standard deviation estimates better conditioned, otherwise could be problem early in training
        g_std = tf.minimum(tf.exp(g_logstd) + 1e-6, 10.0) 
        samples = tf.random.normal([args.batch_size, args.latent_dim], seed=args.seed)
        g = samples*g_std + g_mean
        return g

    def _get_features(self, args, x, u):
        """Find features and temporal encoding given sequence of states and actions
        Args:
            args: Various arguments and specifications
            x: State values [2*batch_size*seq_length, state_dim]
            
        Returns:
            features: Feature values [batch_size, 2*seq_length, latent_dim]
            temporal_encoding: Temporal encoding [batch_size, latent_dim]
        """
        # Get fetures from feature extractor and reshape
        features = self.extractor_net(x)
        features = tf.reshape(features, [args.batch_size, 2*args.seq_length, args.latent_dim])

        # Construct input to temporal encoder and get temporal encoding
        padded_u = tf.concat([tf.zeros([args.batch_size, 1, args.action_dim]), u[:, :(args.seq_length-1)]], axis=1)
        temp_enc_input = tf.concat([features[:, :args.seq_length], padded_u], axis=2)
        temporal_encoding = self.temporal_encoder_net(temp_enc_input)

        return features, temporal_encoding

    def _get_observations(self, args, features, temporal_encoding, u):
        """Finds inferred distributions over g-values and samples from them
        Args:
            args: Various arguments and specifications
            features: Feature values [batch_size, 2*seq_length, latent_dim]
            temporal_encoding: Temporal encoding [batch_size, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            g_vals: Sampled observations [batch_size, seq_length, latent_dim]
            g_dists: Inferred distributions over g-values [batch_size*seq_length, 2*latent_dim]
        """
        # Get inferred distribution over initial observation
        init_obs_input = g_input = tf.concat([temporal_encoding, features[:, 0]], axis=1)
        g1_dist = self.init_inference_net(init_obs_input)

        # Infer remaining observations
        g_t = self._gen_sample(args, g1_dist)

        # Start list of g-distributions and sampled values
        g_vals = [tf.expand_dims(g_t, axis=1)]
        g_dists = [tf.expand_dims(g1_dist, axis=1)]

        # Initialize internal state of observation encoder
        self.observation_encoder_net.reset_states()

        # Loop through time
        for t in range(1, args.seq_length):
            # Update observation encoding
            g_enc = self.observation_encoder_net(tf.expand_dims(g_t, axis=1))

            # Get distribution and sample new observations
            g_dist = self.inference_net(tf.concat([features[:, t], u[:, t-1], temporal_encoding, g_enc], axis=1))
            g_t = self._gen_sample(args, g_dist)

            # Append values to list
            g_vals.append(tf.expand_dims(g_t, axis=1))
            g_dists.append(tf.expand_dims(g_dist, axis=1))

        # Finally, stack inferred observations
        g_vals = tf.reshape(tf.stack(g_vals, axis=1), [args.batch_size, args.seq_length, args.latent_dim])
        g_dists = tf.reshape(tf.stack(g_dists, axis=1), [args.batch_size*args.seq_length, 2*args.latent_dim])

        return g_vals, g_dists

    def _get_prior_dists(self, args, g_vals, u):
        """Derive conditional prior distributions over observations
        Args:
            args: Various arguments and specifications
            g_vals: Sampled observations [batch_size, seq_length, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            g_prior_dists: Prior distributions over g-values [batch_size*seq_length, 2*latent_dim]
        """
        # Construct diagonal unit Gaussian prior params for g1
        g1_prior = tf.concat([tf.zeros([args.batch_size, 1, args.latent_dim]), tf.ones([args.batch_size, 1, args.latent_dim])], axis=2)

        # Construct input to prior network
        gvals_reshape = tf.reshape(g_vals[:, :-1], [args.batch_size*(args.seq_length-1), args.latent_dim])
        u_reshape = tf.reshape(u[:, :(args.seq_length-1)], [args.batch_size*(args.seq_length-1), args.action_dim])
        prior_input = tf.concat([gvals_reshape, u_reshape], axis=1)

        # Find parameters to prior distributions
        prior_params = tf.reshape(self.prior_net(prior_input), [args.batch_size, args.seq_length-1, 2*args.latent_dim])

        # Combine and reshape to get full set of prior distribution parameter values
        g_prior_dists = tf.concat([g1_prior, prior_params], axis=1)

        # Combine and reshape to get full set of prior distribution parameter values
        g_prior_dists = tf.reshape(g_prior_dists, [args.batch_size*args.seq_length, 2*args.latent_dim])

        return g_prior_dists

    def _derive_dynamics(self, args, g_vals, u):
        """Perform least squares to get A- and B-matrices and propagate
        Args:
            args: Various arguments and specifications
            g_vals: Sampled observations [batch_size, seq_length, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            z_vals: Observations obtained by simulating dynamics [batch_size, seq_length, latent_dim]
        """
        # Define X- and Y-matrices
        X = tf.concat([g_vals[:, :-1], u[:, :(args.seq_length-1)]], axis=2)
        Y = g_vals[:, 1:]

        # Solve for A and B using least-squares
        K = tf.linalg.lstsq(X, Y, l2_regularizer=args.l2_regularizer)
        self.A = K[:, :args.latent_dim]
        self.B = K[:, args.latent_dim:]

        # Perform least squares to find A-inverse
        A_inv = tf.linalg.lstsq(Y - tf.matmul(u[:, :(args.seq_length-1)], self.B), g_vals[:, :-1], l2_regularizer=args.l2_regularizer)
        
        # Get predicted code at final time step
        z_t = g_vals[:, -1]

        # Create recursive predictions for z
        z_t = tf.expand_dims(z_t, axis=1)
        z_vals = [z_t]
        for t in range(args.seq_length-2, -1, -1):
            u_t = u[:, t]
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.matmul(z_t - tf.matmul(u_t, self.B), A_inv)
            z_vals.append(z_t) 
        z_vals_reshape = tf.stack(z_vals, axis=1)

        # Flip order
        z_vals_reshape = tf.squeeze(tf.reverse(z_vals_reshape, [1]))

        # Reshape predicted z-values
        z_vals = tf.reshape(z_vals_reshape, [args.batch_size, args.seq_length, args.latent_dim])

        return z_vals

    def _generate_predictions(self, args, z1, u):
        """Generate predictions for how system will evolve given z1, A, and B
        Args:
            args: Various arguments and specifications
            z1: Initial latent state [batch_size, latent_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            z_pred: Observations obtained by simulating dynamics [batch_size, seq_length, latent_dim]
        """
        # Create predictions, simulating forward in time
        z_t = tf.expand_dims(z1, axis=1)
        z_pred = [z_t]
        for t in range(args.seq_length, 2*args.seq_length):
            u_t = u[:, t-1]
            u_t = tf.expand_dims(u_t, axis=1)
            z_t = tf.matmul(z_t, self.A) + tf.matmul(u_t, self.B)
            z_pred.append(z_t) 
        z_pred = tf.stack(z_pred, axis=1)

        # Reshape predicted z-values
        z_pred_reshape = z_pred[:, 1:]
        z_pred = tf.reshape(z_pred[:, 1:], [args.batch_size, args.seq_length, args.latent_dim]) 
        
        return z_pred

    def forward_pass(self, args, x, u):
        """Perform forward pass, mapping states and actions to network outputs
        Args:
            args: Various arguments and specifications
            x: State values [batch_size, 2*seq_length, state_dim]
            u: Action values [batch_size, 2*seq_length-1, action_dim]
        Returns:
            g_prior_dists: Prior distributions over g-values [batch_size*seq_length, 2*latent_dim]
            g_dists: Inferred distributions over g-values [batch_size*seq_length, 2*latent_dim]
            x_pred: State predictions [2*]
        """
        # Normalize inputs
        x = (x - self.shift_x)/self.scale_x
        x = tf.reshape(x, [2*args.batch_size*args.seq_length, args.state_dim])
        u = (u - self.shift_u)/self.scale_u

        # Find features and temporal encoding
        features, temporal_encoding = self._get_features(args, x, u)

        # Get inferred observation distributions and sampled g-values
        g_vals, g_dists = self._get_observations(args, features, temporal_encoding, u)

        # Find prior distribution over g-values
        g_prior_dists = self._get_prior_dists(args, g_vals, u)

        # Derive dynamics and simulate latent dynamics with model
        z_vals = self._derive_dynamics(args, g_vals, u)

        # Simulate dynamics forward to get predicted observations
        z_pred = self._generate_predictions(args, z_vals[:, 0], u)

        # Concatenate z-values and reshape
        z_vals = tf.concat([z_vals, z_pred], axis=1)
        z_vals = tf.reshape(z_vals, [2*args.batch_size*args.seq_length, args.latent_dim])

        # Run z-values through decoder to get state values
        x_pred = self.decoder_net(z_vals)

        return g_dists, g_prior_dists, x_pred



    # def _get_cost(self, args, z_u_t):
    #     """Get cost associated with a set of states and actions
    #     Args:
    #         args: Various arguments and specifications
    #         z_u_t: Latent state and control input at given time step [batch_size, state_dim+action_dim]
    #     Returns:
    #         Cost [batch_size]
    #     """
    #     z_t = z_u_t[:, :args.latent_dim]
    #     u_t = z_u_t[:, args.latent_dim:]
    #     states = self._get_decoder_output(args, z_t)*self.scale + self.shift
    #     if args.domain_name == 'Pendulum-v0':
    #         return tf.square(tf.atan2(states[:, 1], states[:, 0])) + 0.1*tf.square(states[:, 2]) + 0.001*tf.square(tf.squeeze(u_t))
    #     else:
    #         raise NotImplementedError

    # def _find_ilqr_params(self, args):
    #     """Find necessary params to perform iLQR
    #     Args:
    #         args: Various arguments and specifications
    #     """
    #     # Initialize state
    #     z_t = self.z1

    #     # Initialize lists to hold quantities
    #     L = []
    #     L_x = []
    #     L_u = []
    #     L_xx = []
    #     L_ux = []
    #     L_uu = [] 
    #     z_vals = [z_t]

    #     # Loop through time
    #     for t in range(args.mpc_horizon):
    #         # Find cost for current state
    #         z_u_t = tf.concat([z_t, self.u_ilqr[:, t]], axis=1)
    #         l_t = args.gamma**t*self._get_cost(args, z_u_t)

    #         # Find gradients and Hessians (think you need to compute Hessians this way because it handles 3d tensors weirdly)
    #         grads = tf.gradients(l_t, z_u_t)[0]
    #         hessians = tf.reduce_sum(tf.hessians(l_t, z_u_t)[0], axis=2)              

    #         # Separate into individual components
    #         l_x = grads[:, :args.latent_dim]
    #         l_u = grads[:, args.latent_dim:]
    #         l_xx = hessians[:, :args.latent_dim, :args.latent_dim]
    #         l_ux = hessians[:, args.latent_dim:, :args.latent_dim]
    #         l_uu = hessians[:, args.latent_dim:, args.latent_dim:]

    #         # Append to lists
    #         L.append(l_t)
    #         L_x.append(l_x)
    #         L_u.append(l_u)
    #         L_xx.append(l_xx)
    #         L_ux.append(l_ux)
    #         L_uu.append(l_uu)

    #         # Find action by passing it through tanh
    #         u_t = args.action_max*tf.nn.tanh(self.u_ilqr[:, t])
    #         u_t = tf.expand_dims(u_t, axis=1)
    #         z_t = tf.squeeze(tf.matmul(tf.expand_dims(z_t, axis=1), self.A) + tf.matmul(u_t, self.B))
    #         z_vals.append(z_t)

    #     # Find cost and gradients at last time step
    #     z_u_t = tf.concat([z_t, tf.zeros_like(self.u_ilqr[:, -1])], axis=1)
    #     l_T = args.gamma**args.seq_length*self._get_cost(args, z_u_t)
    #     grads = tf.gradients(l_T, z_u_t)[0]
    #     hessians = tf.reduce_sum(tf.hessians(l_T, z_u_t)[0], axis=2)  
    #     L.append(l_T)
    #     L_x.append(grads[:, :args.latent_dim])
    #     L_xx.append(hessians[:, :args.latent_dim, :args.latent_dim])

    #     # Finally stack into tensors
    #     self.L = tf.stack(L, axis=1)
    #     self.L_x = tf.stack(L_x, axis=1)
    #     self.L_u = tf.stack(L_u, axis=1)
    #     self.L_xx = tf.stack(L_xx, axis=1)
    #     self.L_ux = tf.stack(L_ux, axis=1)
    #     self.L_uu = tf.stack(L_uu, axis=1)
    #     self.xs = tf.stack(z_vals, axis=1)
    #     states_pred = self._get_decoder_output(args, tf.reshape(self.xs, [-1, args.latent_dim]))*self.scale + self.shift
    #     self.states_pred = tf.reshape(states_pred, [args.batch_size, -1, args.state_dim])

    # def _create_optimizer(self, args):
    #     """Create optimizer to minimize loss
    #     Args:
    #         args: Various arguments and specifications
    #     """
    #     # First extract mean and std for prior dists, dist over g, and dist over x
    #     g_prior_mean, g_prior_logstd = tf.split(self.g_prior, [args.latent_dim, args.latent_dim], axis=1)
    #     g_prior_std = tf.exp(g_prior_logstd) + 1e-6
    #     g_mean, g_logstd = tf.split(self.g_dists, [args.latent_dim, args.latent_dim], axis=1)
    #     g_std = tf.exp(g_logstd) + 1e-6

    #     # Get predictions for x and reconstructions
    #     self.x_pred_norm = self._get_decoder_output(args, self.z_vals)
    #     self.x_pred = self.x_pred_norm*self.scale + self.shift

    #     # First component of loss: NLL of observed states
    #     x_reshape = tf.reshape(self.x, [args.batch_size, 2*args.seq_length, args.state_dim])
    #     x_pred_reshape = tf.reshape(self.x_pred_norm, [args.batch_size, args.seq_length, args.state_dim])
    #     self.x_pred_init = x_pred_reshape*self.scale + self.shift # needed for ilqr

    #     # Add in predictions for how system will evolve
    #     self.x_pred_reshape = tf.concat([x_pred_reshape, self.x_future_norm], axis=1)
    #     self.x_pred_reshape_unnorm = self.x_pred_reshape*self.scale + self.shift

    #     # Prediction loss
    #     self.pred_loss = tf.reduce_sum(tf.square(x_reshape - self.x_pred_reshape))
    
    #     # Weight loss at t = T more heavily
    #     self.pred_loss += 20.0*tf.reduce_sum(tf.square(x_reshape[:, args.seq_length-1]\
    #                                                          - x_pred_reshape[:, args.seq_length-1]))

    #     # Define reconstructed state needed for ilqr
    #     self.rec_state = self._get_decoder_output(args, self.z1)*self.scale + self.shift

    #     # Second component of loss: KLD between approximate posterior and prior
    #     g_prior_dist = tf.distributions.Normal(loc=g_prior_mean, scale=g_prior_std)
    #     g_dist = tf.distributions.Normal(loc=g_mean, scale=g_std)
    #     self.kl_loss = tf.reduce_sum(tf.distributions.kl_divergence(g_dist, g_prior_dist))

    #     # Sum with regularization losses to form total cost
    #     self.cost = self.pred_loss + self.kl_weight*self.kl_loss + tf.reduce_sum(tf.losses.get_regularization_losses())  

    #     # Perform parameter update
    #     optimizer = tf.train.AdamOptimizer(self.learning_rate)
    #     tvars = [v for v in tf.trainable_variables()]
    #     self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
    #     self.train = optimizer.apply_gradients(zip(self.grads, tvars))

