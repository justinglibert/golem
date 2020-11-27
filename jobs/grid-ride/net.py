from nle import nethack
from torch.nn import functional as F
from torch import nn
import torch
import argparse
import logging
import os
import pprint
import threading
import time
import timeit
import traceback

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def init_no_bias(module, weight_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    return module

class GridNet(nn.Module):
    def __init__(self, observation_shape, num_actions, embedding_size, use_lstm=False):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ELU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ELU()
        )
        n = observation_shape.shape[0]
        m = observation_shape.shape[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.use_lstm = use_lstm
        self.num_actions = num_actions

        self.embedding_size = embedding_size

        self.precore = nn.Sequential(nn.Linear(self.image_embedding_size, self.embedding_size), nn.ReLU())

        if self.use_lstm:
            self.core = nn.LSTM(self.embedding_size, self.embedding_size, num_layers=1)


        self.policy = nn.Linear(self.embedding_size, num_actions)
        
        self.baseline = nn.Linear(self.embedding_size, 1)


    def initial_state(self, batch_size=1, device=torch.device('cpu')):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size,
                        self.core.hidden_size, device=device)
            for _ in range(2)
        )

    def forward(self, env_outputs, core_state):
        frame = env_outputs["frame"]
        T, B, *_ = frame.shape
        frame = torch.flatten(frame, 0, 1)  # Merge time and batch.
        x = frame.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        assert x.shape[0] == T * B
        x = x.view(T * B, -1)
        x = self.precore(x)
        if self.use_lstm:
            core_input = x.view(T, B, -1)
            core_output_list = []
            notdone = (~env_outputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = x


        # -- [B x A]
        policy_logits = self.policy(core_output)
        # -- [B x A]
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(
                F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )

class GridStateEmbeddingNet(nn.Module):
    
    def __init__(self, observation_space):
        super(GridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_space.shape

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        self.image_conv = nn.Sequential(
            init_(nn.Conv2d(3, 32, (3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 128, (3, 3), stride=2, padding=1)),
            nn.ELU()
        )
        
    def forward(self, frame):
        # -- [unroll_length x batch_size x height x width x channels]
        x = frame
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.image_conv(x)

        state_embedding = x.view(T, B, -1)

        return state_embedding

class MinigridInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions 
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * 128, 256)), 
            nn.ReLU(),  
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))

        
    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits
    

class MinigridForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions 

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
    
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(128 + self.num_actions, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.fd_out = init_(nn.Linear(256, 128))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb

