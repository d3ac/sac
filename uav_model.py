import parl
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0


class uavModel(parl.Model):
    def __init__(self, obs_dim, action_dim, n_clusters, batch_size):
        super(uavModel, self).__init__()
        self.n_clusters = n_clusters
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.actor_model = Actor(obs_dim, action_dim, n_clusters)
        self.critic_model = Critic(obs_dim, action_dim, n_clusters)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

class ActorBase(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(ActorBase, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.ModuleList([nn.Linear(64, act_shape[i]) for i in range(len(act_shape))])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, obs):
        obs = obs.to(torch.float32)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = [self.fc_pi[i](x) for i in range(len(self.fc_pi))]
        return logits

class Actor(parl.Model):
    """
    输入state, 输出action
    """
    def __init__(self, obs_dim, action_dim, n_clusters):
        super(Actor, self).__init__()
        self.n_clusters = n_clusters
        self.net = nn.ModuleList([ActorBase(obs_dim, action_dim) for i in range(n_clusters)])

    def forward(self, obs):
        logits = [self.net[i](obs[i].reshape(-1, obs.shape[-1])) for i in range(len(self.net))]
        return logits

class CriticBase(nn.Module):
    def __init__(self, obs_shape, act_shape):
        super(CriticBase, self).__init__()
        # Q1 network
        self.fc1 = nn.Linear(obs_shape[0] + len(act_shape), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(act_shape))
        # Q2 network
        self.fc4 = nn.Linear(obs_shape[0] + len(act_shape), 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, len(act_shape))
    
    def forward(self, obs, action):
        x = torch.cat([obs, action.transpose(0, 1)/10], dim=-1) #TODO 注意这里的action是直接除10的
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        q1 = self.fc3(x1)
        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        q2 = self.fc6(x2)
        return q1, q2

class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim, n_clusters):
        super(Critic, self).__init__()
        self.n_clusters = n_clusters
        self.net = nn.ModuleList([CriticBase(obs_dim, action_dim) for i in range(n_clusters)])

    def forward(self, obs, action):
        q1, q2 = [], []
        for i in range(self.n_clusters):
            a, b = self.net[i](obs[i], action[i])
            q1.append(a)
            q2.append(b)
        return torch.stack(q1, dim=0), torch.stack(q2, dim=0)