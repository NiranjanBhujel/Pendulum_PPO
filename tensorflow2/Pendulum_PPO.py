import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
from Policy import PPOPolicy
from PPOBuffer import PPOBuffer
import matplotlib.pyplot as plt


NUM_STEPS = 2048                    # Number of timesteps data to collect before updating
BATCH_SIZE = 64                     # Batch size of training data
TOTAL_TIMESTEPS = NUM_STEPS * 500   # Total timesteps to run
GAMMA = 0.99                        # Discount factor
GAE_LAM = 0.95                      # Lambda value for generalized advantage estimation
NUM_EPOCHS = 10                     # Number of epochs to train


def get_pi_network(obs_dim, action_dim, lower_bound, upper_bound):
    """
    Function to create actor network.
    """
    input = keras.layers.Input(shape=(obs_dim))
    output = keras.layers.Dense(
        64,
        activation="tanh",
        kernel_initializer=keras.initializers.orthogonal(gain=np.sqrt(2)))(input)
    output = keras.layers.Dense(
        64,
        activation="tanh",
        kernel_initializer=keras.initializers.orthogonal(gain=np.sqrt(2)))(output)

    action_out = keras.layers.Dense(
        action_dim,
        kernel_initializer=keras.initializers.orthogonal(gain=0.01))(output)
    action_out = (action_out + 1)*(upper_bound - lower_bound)/2+lower_bound

    model = keras.Model(inputs=input, outputs=action_out)
    return model


def get_v_network(obs_dim):
    """
    Function to create value network.
    """
    state_input = keras.layers.Input(shape=(obs_dim))
    state_out = keras.layers.Dense(
        64,
        activation="tanh",
        kernel_initializer=keras.initializers.orthogonal(gain=np.sqrt(2)))(state_input)
    q_out = keras.layers.Dense(
        64,
        activation="tanh",
        kernel_initializer=keras.initializers.orthogonal(gain=np.sqrt(2)))(state_out)
    q_out = keras.layers.Dense(
        1,
        kernel_initializer=keras.initializers.orthogonal(gain=1.0))(q_out)

    model = keras.Model(inputs=state_input, outputs=q_out)
    return model


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    lower_bound = env.action_space.low
    upper_bound = env.action_space.high
    
    train_test = "train" if len(sys.argv)==1 else sys.argv[1]
    if train_test=="train":
        pi_network = get_pi_network(obs_dim, action_dim, lower_bound, upper_bound)
        v_network = get_v_network(obs_dim)

        optimizer = keras.optimizers.Adam(learning_rate=3e-4)

        buffer = PPOBuffer(obs_dim, action_dim, NUM_STEPS)
        policy = PPOPolicy(
            pi_network,
            v_network,
            optimizer,
            clip_range=0.2,
            value_coeff=0.5,
            obs_dim=obs_dim,
            action_dim=action_dim,
            initial_std=1.0
        )

        ep_reward = 0.0
        ep_count = 0
        season_count = 0

        pi_losses, v_losses, total_losses, approx_kls, stds = [], [], [], [], []
        mean_rewards = []

        obs = env.reset()
        for t in range(TOTAL_TIMESTEPS):
            action, log_prob, values = policy.get_action(obs)
            clipped_action = np.clip(action, lower_bound, upper_bound)
            next_obs, reward, done, _ = env.step(clipped_action)

            ep_reward += reward

            # Add to buffer
            buffer.record(obs, action, reward, values, log_prob)
            obs = next_obs

            # Calculate advantage and returns if it is the end of episode or its time to update
            if done or (t+1) % NUM_STEPS==0:
                if done:
                    ep_count += 1
                # Value of last time-step
                last_value = tf.squeeze(
                    policy.v_network(tf.expand_dims(obs, 0), training=False))

                # Compute returns and advantage and store in buffer
                buffer.process_trajectory(
                    gamma=GAMMA,
                    gae_lam=GAE_LAM,
                    is_last_terminal=done,
                    last_v=last_value)
                obs = env.reset()

            if (t+1) % NUM_STEPS==0:
                season_count += 1
                # Update for epochs
                for ep in range(NUM_EPOCHS):
                    batch_data = buffer.get_mini_batch(BATCH_SIZE)
                    num_grads = len(batch_data)

                    # Iterate over minibatch of data
                    for k in range(num_grads):
                        (
                            obs_batch,
                            action_batch,
                            log_prob_batch,
                            advantage_batch,
                            return_batch,
                        ) = (
                            batch_data[k]['obs'],
                            batch_data[k]['action'],
                            batch_data[k]['log_prob'],
                            batch_data[k]['advantage'],
                            batch_data[k]['return'],
                        )

                        # Normalize advantage
                        advantage_batch = (
                            advantage_batch -
                            np.squeeze(np.mean(advantage_batch, axis=0))
                        ) / (np.squeeze(np.std(advantage_batch, axis=0)) + 1e-8)

                        # Update the networks on minibatch of data
                        (
                            pi_loss,
                            v_loss,
                            total_loss,
                            approx_kl,
                            std,
                        ) = policy.update(obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch)

                        pi_losses.append(pi_loss.numpy())
                        v_losses.append(v_loss.numpy())
                        total_losses.append(total_loss.numpy())
                        approx_kls.append(approx_kl.numpy())
                        stds.append(std.numpy())
                
                buffer.clear()

                mean_ep_reward = ep_reward / ep_count
                ep_reward, ep_count = 0.0, 0

                print(f"Season={season_count} --> mean_ep_reward={mean_ep_reward}, pi_loss={np.mean(pi_losses)}, v_loss={np.mean(v_losses)}, total_loss={np.mean(total_losses)}, approx_kl={np.mean(approx_kls)}, avg_std={np.mean(stds)}")

                mean_rewards.append(mean_ep_reward)
                pi_losses, v_losses, total_losses, approx_kls, stds = [], [], [], [], []

        # Save policy and value network
        pi_network.save('saved_network/pi_network.h5')
        v_network.save('saved_network/v_network.h5')

        # Plot episodic reward
        _, ax = plt.subplots(1, 1, figsize=(5, 4), constrained_layout=True)
        ax.plot(range(season_count), mean_rewards)
        ax.set_xlabel("season")
        ax.set_ylabel("episodic reward")
        ax.grid(True)
        plt.savefig("season_reward.png")

    elif train_test=="eval" or train_test=="test": 
        # Evaluate trained network
        # Load saved policy network
        pi_network = keras.models.load_model('saved_network/pi_network.h5')
        obs = env.reset()
        for _ in range(500):
            obs_tf = tf.expand_dims(tf.cast(obs, dtype=tf.float32), 0)
            action = pi_network(obs_tf, training=False)
            clipped_action = np.clip(action[0], lower_bound, upper_bound)

            env.render(mode="human")
            obs, reward, done, _ = env.step(clipped_action)
        env.close()

    else:
        print("Please specify whether to train or evaluate!!!")
        sys.exit()
    
