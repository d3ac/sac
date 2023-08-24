import parl
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from parl.utils.utils import check_model_method
from copy import deepcopy
import numpy as np
from torch.distributions import Categorical

__all__ = ['SAC']


class SAC(parl.Algorithm):
    def __init__(self, model, gamma=None, tau=None, alpha=None, actor_lr=None, critic_lr=None):
        """ SAC algorithm
            Args:
                model(parl.Model): forward network of actor and critic.
                gamma(float): discounted factor for reward computation
                tau (float): decay coefficient when updating the weights of self.target_model with self.model
                alpha (float): Temperature parameter determines the relative importance of the entropy against the reward
                actor_lr (float): learning rate of the actor model
                critic_lr (float): learning rate of the critic model
        """
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        device = torch.device("cpu")
        self.model = model.to(device)
        self.target_model = deepcopy(self.model)
        self.actor_optimizer = torch.optim.Adam(self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.get_critic_params(), lr=critic_lr)

    def predict(self, obs):
        logits = self.model.policy(obs)
        act = np.zeros((self.model.n_clusters, len(self.model.action_dim)), dtype=np.int64)
        for i in range(self.model.n_clusters):
            for j in range(len(self.model.action_dim)):
                act[i][j] = torch.argmax(logits[i][j]).numpy()
        return act

    def sample(self, obs):
        logits = self.model.policy(obs)
        action = torch.zeros(size=(self.model.n_clusters, len(self.model.action_dim), self.model.batch_size), dtype=torch.int64)
        action_log_probs = torch.zeros(size=(self.model.n_clusters, len(self.model.action_dim), self.model.batch_size))
        for i in range(self.model.n_clusters):
            for j in range(len(self.model.action_dim)):
                dist = Categorical(logits=logits[i][j])
                action[i][j] = dist.sample()
                action_log_probs[i][j] = dist.log_prob(action[i][j])
        loss = action_log_probs.mean()
        loss.backward()
        return action, action_log_probs



    def learn(self, obs, action, reward, next_obs, terminal):
        critic_loss = self._critic_learn(obs, action, reward, next_obs, terminal)
        actor_loss = self._actor_learn(obs)

        self.sync_target()
        return critic_loss, actor_loss

    def _critic_learn(self, obs, action, reward, next_obs, terminal):
        with torch.no_grad():
            next_action, next_log_pro = self.sample(next_obs)
            q1_next, q2_next = self.target_model.value(next_obs, next_action)
            target_Q = torch.min(q1_next, q2_next) - self.alpha * next_log_pro.transpose(2, 1)
            target_Q = reward.unsqueeze(2) + self.gamma * (1. - terminal.unsqueeze(2)) * target_Q
        cur_q1, cur_q2 = self.model.value(obs, action.transpose(2, 1))

        critic_loss = F.mse_loss(cur_q1, target_Q) + F.mse_loss(cur_q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, obs):
        act, log_pi = self.sample(obs)
        q1_pi, q2_pi = self.model.value(obs, act)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi.transpose(1, 2)).mean()
        # actor_loss = ((self.alpha) - min_q_pi.transpose(1, 2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data + decay * target_param.data)