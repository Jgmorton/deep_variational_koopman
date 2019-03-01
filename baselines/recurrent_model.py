import numpy as np
import tensorflow as tf

class RecurrentModel():
    def __init__(self, args):

        # Placeholder for data -- inputs are number of elements x pts in mesh x dimensionality of data for each point
        self.x = tf.Variable(np.zeros((2*args.batch_size*args.seq_length, args.state_dim), dtype=np.float32), trainable=False, name="state_values")
        self.u = tf.Variable(np.zeros((args.batch_size, 2*args.seq_length-1, args.action_dim), dtype=np.float32), trainable=False, name="action_values")

        # Parameters to be set externally
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")
        self.kl_weight = tf.Variable(0.0, trainable=False, name="kl_weight")

        # Normalization parameters to be stored
        self.shift = tf.Variable(np.zeros(args.state_dim), trainable=False, name="input_shift", dtype=tf.float32)
        self.scale = tf.Variable(np.zeros(args.state_dim), trainable=False, name="input_scale", dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_shift", dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_scale", dtype=tf.float32)
        
        # Create the computational graph
        self._create_feature_extractor_params(args)
        self._create_feature_extractor(args)
        self._create_initial_generator(args)
        self._propagate_solution(args)
        self._create_decoder_params(args)
        self._create_optimizer(args)

    # Create parameters to comprise feature extractor network
    def _create_feature_extractor_params(self, args):
        self.extractor_w = []
        self.extractor_b = []

        # Loop through elements of decoder network and define parameters
        for i in range(len(args.extractor_size)):
            if i == 0:
                prev_size = args.state_dim
            else:
                prev_size = args.extractor_size[i-1]
            self.extractor_w.append(tf.get_variable("extractor_w"+str(i), [prev_size, args.extractor_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.extractor_b.append(tf.get_variable("extractor_b"+str(i), [args.extractor_size[i]]))

        # Last set of weights to map to output
        self.extractor_w.append(tf.get_variable("extractor_w_end", [args.extractor_size[-1], args.code_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.extractor_b.append(tf.get_variable("extractor_b_end", [args.code_dim]))

    # Function to run inputs through extractor
    def _get_extractor_output(self, args, states):
        extractor_input = states
        for i in range(len(args.extractor_size)):
            extractor_input = tf.nn.relu(tf.nn.xw_plus_b(extractor_input, self.extractor_w[i], self.extractor_b[i]))
        output = tf.nn.xw_plus_b(extractor_input, self.extractor_w[-1], self.extractor_b[-1])
        return output

    # Create feature extractor (maps state -> features, assumes feature same dimensionality as latent states)
    def _create_feature_extractor(self, args):
        features = self._get_extractor_output(args, self.x)
        self.features = tf.reshape(features, [args.batch_size, 2*args.seq_length, args.code_dim])

    # Function to generate samples given distribution parameters
    def _gen_sample(self, args, dist_params):
        z_mean, z_logstd = tf.split(dist_params, [args.code_dim, args.code_dim], axis=1)
        z_std = tf.minimum(tf.exp(z_logstd) + 1e-6, 10.0)
        samples = tf.random_normal([args.batch_size, args.code_dim])
        z = samples*z_std + z_mean
        return z

    # Bidirectional LSTM to generate initial sample of z1
    def _create_initial_generator(self, args):
        fwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        bwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        
        # Get outputs from rnn and concatenate
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, self.features[:, :args.seq_length], dtype=tf.float32)
        output_fw, output_bw = outputs
        output = tf.concat([output_fw[:, -1], output_bw[:, -1]], axis=1)

        # Single affine transformation into z1 distribution params
        hidden = tf.layers.dense(output, 
                                args.transform_size, 
                                activation=tf.nn.relu,
                                name='to_hidden_w1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.z1_dist = tf.layers.dense(hidden, 
                                2*args.noise_dim, 
                                name='to_w1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.z1 = self._gen_sample(args, self.z1_dist)

    # Now use various params/networks to propagate solution forward in time
    def _propagate_solution(self, args):
        # Start list of z-values
        z_t = self.z1
        self.z_vals = [tf.expand_dims(self.z1, axis=1)]

        # Create parameters for transformation to be performed at output of GRU
        W_z_out = tf.get_variable("w_z_out", [args.rnn_size, args.transform_size], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        b_z_out = tf.get_variable("b_z_out", [args.transform_size])
        W_to_z_enc = tf.get_variable("w_to_z_enc", [args.transform_size, args.code_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        b_to_z_enc = tf.get_variable("b_to_z_enc", [args.code_dim])

        # Initialize single-layer GRU network to create observation encodings
        cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        self.rnn_init_state = cell.zero_state(args.batch_size, tf.float32)
        self.rnn_state = self.rnn_init_state

        # Loop through time steps
        for t in range(1, 2*args.seq_length):
            # Generate temporal encoding
            self.rnn_output, self.rnn_state = cell(tf.concat([z_t, self.u[:, t-1]], axis=1), self.rnn_state)
            hidden = tf.nn.relu(tf.nn.xw_plus_b(self.rnn_output, W_z_out, b_z_out))
            z_t = tf.nn.xw_plus_b(hidden, W_to_z_enc, b_to_z_enc)

            # Append values to list
            self.z_vals.append(tf.expand_dims(z_t, axis=1))

        # Finally, stack inferred observations and distributions and flip order
        self.z_pred = tf.reshape(tf.stack(self.z_vals, axis=1), [2*args.batch_size*args.seq_length, args.code_dim])
        
    # Create parameters to comprise decoder network
    def _create_decoder_params(self, args):
        self.decoder_w = []
        self.decoder_b = []

        # Loop through elements of decoder network and define parameters
        for i in range(len(args.extractor_size)-1, -1, -1):
            if i == len(args.extractor_size)-1:
                prev_size = args.code_dim
            else:
                prev_size = args.extractor_size[i+1]
            self.decoder_w.append(tf.get_variable("decoder_w"+str(len(args.extractor_size)-i), [prev_size, args.extractor_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.decoder_b.append(tf.get_variable("decoder_b"+str(len(args.extractor_size)-i), [args.extractor_size[i]]))

        # Last set of weights to map to output
        self.decoder_w.append(tf.get_variable("decoder_w_end", [args.extractor_size[0], args.state_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.decoder_b.append(tf.get_variable("decoder_b_end", [args.state_dim]))

    # Function to run inputs through decoder
    def _get_decoder_output(self, args, encodings):
        decoder_input = encodings
        for i in range(len(args.extractor_size)):
            decoder_input = tf.nn.relu(tf.nn.xw_plus_b(decoder_input, self.decoder_w[i], self.decoder_b[i]))
        output = tf.nn.xw_plus_b(decoder_input, self.decoder_w[-1], self.decoder_b[-1])
        return output

    # Create optimizer to minimize loss
    def _create_optimizer(self, args):
        # Find reconstruction loss
        self.rec_sol = self._get_decoder_output(args, self.z_pred)
        x_reshape = tf.reshape(self.x, [args.batch_size, 2*args.seq_length, args.state_dim])
        rec_sol_reshape = tf.reshape(self.rec_sol, [args.batch_size, 2*args.seq_length, args.state_dim])
        self.loss_reconstruction = tf.reduce_sum(tf.square(x_reshape[:, :args.seq_length] - rec_sol_reshape[:, :args.seq_length]))

        # Find state predictions by undoing data normalization
        self.state_pred = self.rec_sol*self.scale + self.shift

        # Find KL-divergence component of loss
        z1_mean, z1_logstd = tf.split(self.z1_dist, [args.code_dim, args.code_dim], axis=1)
        z1_std = tf.exp(z1_logstd) + 1e-6

        # Define distribution and prior objects
        z1_dist = tf.distributions.Normal(loc=z1_mean, scale=z1_std)
        prior_dist = tf.distributions.Normal(loc=tf.zeros_like(z1_mean), scale=tf.ones_like(z1_std))
        self.kl_loss = tf.reduce_sum(tf.distributions.kl_divergence(z1_dist, prior_dist))

        # Sum with regularization losses to form total cost
        self.cost = self.loss_reconstruction + tf.reduce_sum(tf.losses.get_regularization_losses()) + self.kl_weight*self.kl_loss

        # Perform parameter update
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars))




