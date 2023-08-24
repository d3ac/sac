#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import parl
import torch
import numpy as np
from torch.distributions import Categorical

class Agent(parl.Agent):
    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm)
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        action = self.alg.predict(obs)
        return action

    def sample(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        logits = self.alg.model.policy(obs)
        action = torch.zeros(size=(self.alg.model.n_clusters, len(self.alg.model.action_dim)), dtype=torch.int64)
        for i in range(self.alg.model.n_clusters):
            for j in range(len(self.alg.model.action_dim)):
                dist = Categorical(logits=logits[i][j])
                action[i][j] = dist.sample()
        return action

    def learn(self, obs, action, reward, next_obs, terminal):
        obs = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)

        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs, terminal)
        return critic_loss, actor_loss