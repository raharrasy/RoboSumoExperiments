import argparse
import torch
import time
import os
import numpy as np
import gym
from gym.spaces import Box, Discrete
from torch.autograd import Variable
from GaussianNoise import GaussianNoise
import robosumo.envs
#from env_wrappers import SubprocVecEnv, DummyVecEnv
from Network import MLP
from misc import *
import torch.nn as nn
from ReplayBuffer import *
import random

USE_CUDA = False  # torch.cuda.is_available()
from torch.optim import Adam


def numpy_random_disc_action(numActions):
    actionArr = np.array([0]*numActions)
    actionId = random.randint(0, numActions-1)
    actionArr[actionId] = 1
    return actionArr


def step_with_exp(obs_collection, nets_collection, noise=None):
    actions = [model(obs)[0] for obs, model in zip(obs_collection, nets_collection)]
    if noise is not None:
         actions = [action + torch.Tensor(noise.noise(batch_dim=action.size(0))).clamp(-0.5, 0.5) for action in actions]
         actions = [action.clamp(-1.0, 1.0) for action in actions]

    return actions


def getQValues(critic_network, joint_state, joint_act):
    return critic_network(torch.cat((joint_state, joint_act), dim=-1))


def criticUpdate(critic_nets, critic_optim, target_actor_nets, target_critic_net, states, all_next_obs_agents, acts,
                 rewards, dones, kwargs, agent_id):
    critic_optim.zero_grad()
    # joint_states = torch.cat(states, dim=-1)
    # joint_act = torch.cat(acts, dim=-1)
    # joint_next_obs = torch.cat(all_next_obs_agents, dim=-1)

    predicted_q_val_1 = getQValues(critic_nets[0], states[agent_id], acts[agent_id])
    predicted_q_val_2 = getQValues(critic_nets[1], states[agent_id], acts[agent_id])
    noise = GaussianNoise(8, sigma=0.2)

    next_acts = step_with_exp([all_next_obs_agents[agent_id]], [target_actor_nets[agent_id]], noise=noise)[0]
    next_input = torch.cat((all_next_obs_agents[agent_id], next_acts), dim=-1)
    next_vals_1 = target_critic_net[0](next_input).detach()
    next_vals_2 = target_critic_net[1](next_input).detach()
    next_vals = torch.min(next_vals_1, next_vals_2)
    gamma = kwargs['gamma']

    target_vals = rewards + gamma * (1-dones) * next_vals
    loss_func = nn.MSELoss()
    loss = loss_func(predicted_q_val_1, target_vals) + loss_func(predicted_q_val_2, target_vals)
    loss.backward()
    print('TD LOSS', loss)
    nn.utils.clip_grad_norm_(critic_nets[0].parameters(), kwargs['clip_grad'])
    nn.utils.clip_grad_norm_(critic_nets[1].parameters(), kwargs['clip_grad'])
    critic_optim.step()


def actorUpdate(agentIdx, states, actor_net, critic_net, actor_optim, acts, kwargs):
    actor_optim.zero_grad()
    agent_obs = states[agentIdx]
    # predicted_action = step_with_exp([agent_obs], [actor_net], noise=None)[0]
    predicted_action = actor_net(agent_obs)[0]
    qVal = critic_net(torch.cat((agent_obs, predicted_action), dim=-1))
    loss = -torch.mean(qVal)
    print('POLICY LOSS', loss)
    loss.backward()
    nn.utils.clip_grad_norm_(actor_net.parameters(), kwargs['clip_grad'])
    actor_optim.step()


def update(actor_nets, critic_nets1, critic_nets2, target_actor_nets, target_critic_nets1, target_critic_nets2,
           exp_replay, actor_optimizers, critic_optimizers, kwargs, timestep):
    if exp_replay.__len__() > kwargs['batch_size']:
        idx = 0
        obs, act, rews, next_obs, dones = exp_replay.sample(kwargs['batch_size'], norm_rews=False)
        for actor_network, critic_network1, critic_network2, actor_optim, critic_optim, target_actor_network, \
            target_critic_network1, target_critic_network2 in \
                zip(actor_nets, critic_nets1, critic_nets2, actor_optimizers, critic_optimizers, target_actor_nets,
                    target_critic_nets1, target_critic_nets2):

            criticUpdate([critic_network1, critic_network2], critic_optim, target_actor_nets,
                         [target_critic_network1, target_critic_network2], obs, next_obs, act,
                         rews[idx], dones[idx], kwargs, idx)
            if timestep % 2 == 0:
                actorUpdate(idx, obs, actor_network, critic_network1, actor_optim, act, kwargs)

            idx += 1


def main(**kwargs):

    environment = gym.make("RoboSumo-Ant-vs-Ant-v0")
    observation_space = environment.observation_space
    action_space = environment.action_space
    noise = GaussianNoise(8, sigma=0.1)

    action_dim = [8, 8]
    obs_dim = [120, 120]

    actor_nets = [MLP(obs_dim[i], [400, 300], action_dim[i]) for i in range(2)]
    critic_nets1 = [MLP((obs_dim[i] + action_dim[i]), [400, 300], 1, net_type="Critic") for
                   i in range(2)]
    critic_nets2 = [MLP((obs_dim[i] + action_dim[i]), [400, 300], 1, net_type="Critic") for
                   i in range(2)]
    target_actor_nets = [MLP(obs_dim[i], [400, 300], action_dim[i]) for i in range(2)]
    target_critic_nets1 = [MLP((obs_dim[i] + action_dim[i]),
                               [400, 300], 1, net_type="Critic") for i in range(2)]
    target_critic_nets2 = [MLP((obs_dim[i] + action_dim[i]),
                               [400, 300], 1, net_type="Critic") for i in range(2)]

    actor_optimizers = [Adam(actor_net.parameters(), lr=kwargs['actor_lr']) for actor_net in actor_nets]
    critic_optimizers = [Adam(list(c1.parameters()) + list(c2.parameters()), lr=kwargs['critic_lr'])
                         for c1, c2 in zip(critic_nets1, critic_nets2)]

    for actor_network, target_network in zip(actor_nets, target_actor_nets):
        hard_update(target_network, actor_network)

    for critic_network, target_network in zip(critic_nets1, target_critic_nets1):
        hard_update(target_network, critic_network)
    for critic_network, target_network in zip(critic_nets2, target_critic_nets2):
        hard_update(target_network, critic_network)

    exp_replay = ReplayBuffer(int(1e6), 2, obs_dim, action_dim)
    totalSteps = 0
    totalUpdates = 0
    for epsNum in range(kwargs['num_episodes']):
        numSteps = 0
        observation = environment.reset()
        observation = [[observation[0], observation[1]]]
        dones = [[False, False]]
        
        total_rewards = [0.0, 0.0]

        while dones[0][0] is False and dones[0][1] is False:
            tens_obs = [Variable(torch.Tensor([elem]), requires_grad=False) for elem in observation[0]]
            if totalSteps >= 1000:
                actions = step_with_exp(tens_obs, actor_nets, noise=noise)
            else:
                actions = environment.action_space.sample()
                actions = [torch.Tensor([a]) for a in actions]
            print(actions)
            actions = [a.detach() for a in actions]
            action_numpy = [a[0].data.numpy() for a in actions]
            nextObs, rewards, dones, info = environment.step(action_numpy)
            numSteps += 1
            next_obs_given = [[nextObs[0], nextObs[1]]]
            dones = [[dones[0], dones[1]]]

            rewards = [[rewards[0], rewards[1]]]
            numSteps += 1
            totalSteps += 1

            total_rewards[0] += rewards[0][0]
            total_rewards[1] += rewards[0][1]
            environment.render(mode='human')
            exp_replay.push(np.asarray(observation), action_numpy, np.asarray(rewards) * 0.01, np.asarray(next_obs_given),
                            np.asarray(dones))
            if totalSteps % kwargs['train_frequency'] == 0:
                update(actor_nets, critic_nets1, critic_nets2, target_actor_nets, target_critic_nets1,
                       target_critic_nets2, exp_replay, actor_optimizers, critic_optimizers, kwargs, totalSteps)
                for actor_network, target_network in zip(actor_nets, target_actor_nets):
                    soft_update(target_network, actor_network, kwargs['tau'])

                for critic_network, target_network in zip(critic_nets1, target_critic_nets1):
                    soft_update(target_network, critic_network, kwargs['tau'])
                for critic_network, target_network in zip(critic_nets2, target_critic_nets2):
                    soft_update(target_network, critic_network, kwargs['tau'])
                totalUpdates += 1

            observation = nextObs
            observation = [[observation[0], observation[1]]]

        print("Total Reward episode ", epsNum, " : ", total_rewards, "timesteps :", totalSteps)


if __name__ == "__main__":
    main(
        num_episodes=125000,
        gamma=0.95,
        train_frequency=1,
        batch_size=100,
        tau=5e-3,
        critic_lr=0.0001,
        actor_lr=0.0001,
        clip_grad=0.5
    )
