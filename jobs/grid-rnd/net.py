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

