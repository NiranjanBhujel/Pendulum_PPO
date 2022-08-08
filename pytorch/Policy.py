import torch
from torch import nn
from torch.distributions import Normal


class PPOPolicy(nn.Module):
    def __init__(self, pi_network, v_network, learning_rate, clip_range, value_coeff, obs_dim, action_dim, initial_std=1.0, max_grad_norm=0.5) -> None:
        super().__init__()

        (
            self.pi_network,
            self.v_network,
            self.learning_rate,
            self.clip_range,
            self.value_coeff,
            self.obs_dim,
            self.action_dim,
            self.max_grad_norm,
        ) = (
            pi_network,
            v_network,
            learning_rate,
            clip_range,
            value_coeff,
            obs_dim,
            action_dim,
            max_grad_norm
        )

        # Gaussian policy will be used. So, log standard deviation is created as trainable variables
        self.log_std =  nn.Parameter(torch.ones(self.action_dim) * torch.log(torch.tensor(initial_std)), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, obs):
        pi_out = self.pi_network(obs)

        # Add Normal distribution layer at the output of pi_network
        dist_out = Normal(pi_out, torch.exp(self.log_std))

        v_out = self.v_network(obs)

        return dist_out, v_out

    def get_action(self, obs):
        """
        Sample action based on current policy
        """
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)
        dist, values = self.forward(obs_torch)
        action = dist.sample()
        log_prob = torch.sum(dist.log_prob(action), dim=1)

        return action[0].detach().numpy(), torch.squeeze(log_prob).detach().numpy(), torch.squeeze(values).detach().numpy()

    def get_values(self, obs):
        """
        Function  to return value of the state
        """
        obs_torch = torch.unsqueeze(torch.tensor(obs, dtype=torch.float32), 0)

        _, values = self.forward(obs_torch)

        return torch.squeeze(values).detach().numpy()

    def evaluate_action(self, obs_batch, action_batch, training):
        """
        Evaluate taken action.
        """     
  
        obs_torch = obs_batch.clone().detach()
        action_torch = action_batch.clone().detach()
        dist, values = self.forward(obs_torch)
        log_prob = dist.log_prob(action_torch)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return log_prob, values

    def update(self, obs_batch, action_batch, log_prob_batch, advantage_batch, return_batch):
        """
        Performs one step gradient update of policy and value network.
        """

        new_log_prob, values = self.evaluate_action(obs_batch, action_batch, training=True)

        ratio = torch.exp(new_log_prob-log_prob_batch)
        clipped_ratio = torch.clip(
            ratio,
            1-self.clip_range,
            1+self.clip_range,
        )

        surr1 = ratio * advantage_batch
        surr2 = clipped_ratio * advantage_batch
        pi_loss = -torch.mean(torch.min(surr1, surr2))
        value_loss = self.value_coeff * torch.mean((values - return_batch)**2)
        total_loss = pi_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return (
            pi_loss.detach(), 
            value_loss.detach(), 
            total_loss.detach(), 
            (torch.mean((ratio - 1) - torch.log(ratio))).detach(), 
            torch.exp(self.log_std).detach()
        )
