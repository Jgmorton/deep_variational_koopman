import numpy as np
import tensorflow as tf

class MLP():
    def __init__(self, args):

        # Placeholder for data -- inputs are number of elements x pts in mesh x dimensionality of data for each point
        self.x = tf.Variable(np.zeros((args.batch_size, args.seq_length, args.state_dim), dtype=np.float32), trainable=False, name="state_values")
        self.u = tf.Variable(np.zeros((args.batch_size, args.seq_length-1, args.action_dim), dtype=np.float32), trainable=False, name="action_values")

        # Parameters to be set externally
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")

        # Normalization parameters to be stored
        self.shift = tf.Variable(np.zeros(args.state_dim), trainable=False, name="input_shift", dtype=tf.float32)
        self.scale = tf.Variable(np.zeros(args.state_dim), trainable=False, name="input_scale", dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_shift", dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros(args.action_dim), trainable=False, name="action_scale", dtype=tf.float32)
        
        # Create the computational graph
        self._create_network_params(args)
        self._create_network(args)
        self._create_optimizer(args)

    # Create parameters to comprise network
    def _create_network_params(self, args):
        self.network_w = []
        self.network_b = []

        # Loop through elements of network and define parameters
        for i in range(len(args.network_size)):
            if i == 0:
                prev_size = args.state_dim + args.action_dim
            else:
                prev_size = args.network_size[i-1]
            self.network_w.append(tf.get_variable("network_w"+str(i), [prev_size, args.network_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.network_b.append(tf.get_variable("network_b"+str(i), [args.network_size[i]]))

        # Last set of weights to map to output
        self.network_w.append(tf.get_variable("network_w_end", [args.network_size[-1], args.state_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.network_b.append(tf.get_variable("network_b_end", [args.state_dim]))

    # Function to run inputs through network
    def _get_network_output(self, args, states, actions):
        network_input = tf.concat([states[:, :-1], actions], axis=2)
        network_input = tf.reshape(network_input, [-1, args.state_dim + args.action_dim])
        for i in range(len(args.network_size)):
            network_input = tf.nn.relu(tf.nn.xw_plus_b(network_input, self.network_w[i], self.network_b[i]))
        output = tf.nn.xw_plus_b(network_input, self.network_w[-1], self.network_b[-1])
        return output

    # Create network to predict state evolution
    def _create_network(self, args):
        preds = self._get_network_output(args, self.x, self.u)
        self.preds = tf.reshape(preds, [args.batch_size, args.seq_length-1, args.state_dim])

    # Create optimizer to minimize loss
    def _create_optimizer(self, args):
        # Find reconstruction loss
        self.loss_reconstruction = tf.reduce_sum(tf.square(self.x[:, 1:]*self.scale + self.shift - self.preds))

        # Sum with regularization losses to form total cost
        self.cost = self.loss_reconstruction + tf.reduce_sum(tf.losses.get_regularization_losses())

        # Perform parameter update
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars))




