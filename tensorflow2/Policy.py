import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


class PPOPolicy:
    def __init__(self, pi_network, v_network, optimizer, clip_range, value_coeff, obs_dim, action_dim, initial_std=1.0, max_grad_norm=0.5) -> None:
        (
            self.pi_network,
            self.v_network,
            self.optimizer,
            self.clip_range,
            self.value_coeff,
            self.obs_dim,
            self.action_dim,
            self.max_grad_norm,
        ) = (
            pi_network,
            v_network,
            optimizer,
            clip_range,
            value_coeff,
            obs_dim,
            action_dim,
            max_grad_norm
        )

        # Gaussian policy will be used. So, log standard deviation is created as trainable variables
        self.log_std = tf.Variable(
            initial_value=tf.math.log(initial_std) *
            tf.ones(shape=self.action_dim),
            dtype=tf.float32,
            trainable=True
        )

        # Add Normal distribution layer at the output of pi_network
        input = keras.layers.Input(
            shape=(self.obs_dim,)
        )
        pi_out = self.pi_network(input)
        dist_out = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda t: tfp.distributions.MultivariateNormalDiag(
                t, tf.exp(self.log_std)),
        )(pi_out)

        self.policy_dist = keras.Model(inputs=input, outputs=dist_out)

        # Parameter from policy networks and value networks are optimized simultaneously along with `self.log_std` variable
        self.trainable_params = self.pi_network.trainable_variables + \
            self.v_network.trainable_variables + [self.log_std]

    @tf.function
    def get_action(self, obs):
        """
        Sample action based on current policy
        """
        obs_tf = tf.expand_dims(tf.cast(obs, dtype=tf.float32), 0)
        dist = self.policy_dist(obs_tf, training=False)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        action = action[0]
        log_prob = tf.squeeze(log_prob)
        values = tf.squeeze(self.v_network(obs_tf, training=False))

        return action, log_prob, values

    @tf.function
    def evaluate_action(self, obs_batch, action_batch, training):
        """
        Evaluate taken action.
        """
        dist = self.policy_dist(obs_batch, training=training)
        log_prob = dist.log_prob(action_batch)
        log_prob = tf.expand_dims(log_prob, 1)

        values = self.v_network(obs_batch, training=training)

        return log_prob, values

    @tf.function
    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch):
        """
        Performs one step gradient update of policy and value network.
        """

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_params)
            new_log_prob, values = self.evaluate_action(
                obs_batch, action_batch, training=True)

            ratio = tf.exp(new_log_prob-log_prob_batch)
            clipped_ratio = tf.clip_by_value(
                ratio,
                1-self.clip_range,
                1+self.clip_range,
            )

            surr1 = ratio * advantage_batch
            surr2 = clipped_ratio * advantage_batch
            pi_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            value_loss = self.value_coeff * \
                tf.reduce_mean(tf.square(values - return_batch))

            total_loss = pi_loss + value_loss

        grads = tape.gradient(total_loss, self.trainable_params)
        grads = [(tf.clip_by_norm(grad, clip_norm=self.max_grad_norm))
                 for grad in grads]
        self.optimizer.apply_gradients(
            zip(
                grads,
                self.trainable_params
            )
        )

        return pi_loss, value_loss, total_loss, tf.reduce_mean((ratio - 1) - tf.math.log(ratio)), tf.exp(self.log_std)
