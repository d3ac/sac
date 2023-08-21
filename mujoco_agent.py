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


class MujocoAgent(parl.Agent):
    def __init__(self, algorithm):
        super(MujocoAgent, self).__init__(algorithm)

        self.alg.sync_target(decay=0)

    def predict(self, obs):
        obs = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
        action = self.alg.predict(obs)
        action_numpy = action.cpu().detach().numpy()[0]
        return action_numpy

    def sample(self, obs):
        obs = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
        action, _ = self.alg.sample(obs)
        action_numpy = action.cpu().detach().numpy()[0]
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        terminal = torch.tensor(terminal, dtype=torch.float32)
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs, terminal)
        return critic_loss, actor_loss