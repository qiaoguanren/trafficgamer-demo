# BSD 3-Clause License

# Copyright (c) 2024

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from algorithm.mappo import MAPPO
from typing import Optional
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Laplace
from utils.dual_variable import DualVariable
import numpy as np
from tqdm import tqdm
from utils.normalizing_flow import MADE, BatchNormFlow, Reverse, FlowSequential
from utils.utils import compute_advantage

class CCE_MAPPO(MAPPO):
    def __init__(self, state_dim: int, agent_number: int, config, device):
        super(CCE_MAPPO, self).__init__(state_dim, agent_number, config, device)
        self.beta_coef = config["beta_coef"]
        self.magnet = config['is_magnet']
        self.eta_coef1 = config["eta_coef1"]

    def sample(self, transition, agent_index):

        s = []
        o = []
        a = []
        r = []
        c = []
        s_next = []
        o_next = []
        done = []
        magnet = []                    
        with torch.no_grad():

            for row in range(0, self.batch_size):
                for i in range(self.agent_number):
                    s += transition[row]["observations"][i]

                    s_next += transition[row]["next_observations"][i]
                a += transition[row]["actions"][agent_index]
                r += transition[row]["rewards"][agent_index] 
                c += transition[row]["costs"][agent_index]
                o += transition[row]["observations"][agent_index]
                o_next += transition[row]["next_observations"][agent_index]
                magnet += transition[row]["magnet"][agent_index]

                done += transition[row]["dones"]
            return s, o, a, r, c, s_next, o_next, done, magnet

    def update(self, transition, agent_index):

        logs = []
        
        states, observations, actions, rewards, costs, next_states, next_observations, dones, magnet = (
            self.sample(transition, agent_index)
        )

        dones = torch.stack(dones, dim=0).view(-1, 1)
        states = (
            torch.stack(states, dim=0)
            .reshape(-1, self.agent_number * self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        next_states = (
            torch.stack(next_states, dim=0)
            .reshape(-1, self.agent_number * self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        observations = (
            torch.stack(observations, dim=0)
            .reshape(-1, self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        next_observations = (
            torch.stack(next_observations, dim=0)
            .reshape(-1, self.state_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )

        rewards = torch.stack(rewards, dim=0).to(self.device)
        costs = torch.stack(costs, dim=0).to(self.device)

        actions = torch.stack(actions, dim=0).to(self.device)
        
            
        if  self.magnet:
                magnet_signal = torch.tensor(
                    [magnet[i].log_prob(actions[i]) for i in range(len(actions))]
                )
                rewards = rewards + self.eta_coef1 * magnet_signal[:, None].to(self.device)

        with torch.no_grad():
            
            # get ratio
            old_policy = self.get_action_dist(observations)
            old_log_probs = old_policy.log_prob(actions)
            
            next_state_value = self.value(next_states)
            
            td_target = rewards + self.gamma * next_state_value * (1 - dones)
        
            td_value = self.value(states)
            td_delta = td_target - td_value

            # calculate GAE
            reward_advantages = compute_advantage(
                self.gae, td_delta, self.device, dones, self.gamma, self.lamda)
            # advantage_normalization
            reward_advantages = (reward_advantages - reward_advantages.mean()) / (
                reward_advantages.std() + 1e-8
            )

            returns = reward_advantages + td_value
        
        temp_pi_old = torch.ones_like(old_log_probs)    
        for epoch in tqdm(range(self.epochs)):
            log = {}
            log["magnet_signal"] = magnet_signal.mean().item()
            
            # for target_param, local_param in zip(self.cost_value_net_target.parameters(), self.cost_value_net_local.parameters()):
            #         target_param.data.copy_(
            #             self.tau_update * local_param.data + (1.0 - self.tau_update) * target_param.data)
            
            new_policy = self.get_action_dist(observations)
            log_probs = new_policy.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)

            # clipping
            ratio = ratio.flatten(start_dim=1)
            surr1 = ratio * reward_advantages
            surr2 = reward_advantages * torch.clamp(ratio, 1 - self.eps, 1 + self.eps)

            value = self.value(states)

            # pi and value loss
            pi_loss = torch.mean(-torch.min(surr1, surr2))
            value_loss = torch.mean(F.mse_loss(value, returns.detach()))
            entropy = new_policy.entropy()

            log["entropy"] = entropy.mean().item()
            total_loss = (
                pi_loss + 0.5 * value_loss  - entropy * self.entropy_coef
            ).mean()
            log["total_loss"] = total_loss.mean().item()
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 0.5)
            self.optimizer.step()

            logs.append(log)

        return logs
        
