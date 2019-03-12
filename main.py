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
    actions = [model((obs)) for obs, model in zip(obs_collection, nets_collection)]
    if noise is not None:
         actions = [action + torch.Tensor(noise.noise()) for action in actions]
         actions = [action.clamp(-1.0,1.0) for action in actions]

    return actions


def getQValues(critic_network, joint_state, joint_act):
    return critic_network((Variable(torch.cat((joint_state, joint_act), dim=1))))


def criticUpdate(critic_net, critic_optim, target_actor_nets, target_critic_net, states, all_next_obs_agents, acts, rewards, dones, kwargs):
    joint_states = torch.cat((states), dim=1)
    joint_act = torch.cat((acts), dim=1)
    joint_next_obs = torch.cat((all_next_obs_agents), dim=1)

    predicted_q_val = getQValues(critic_net, joint_states, joint_act)
    
    next_acts = step_with_exp(all_next_obs_agents, target_actor_nets, noise=None)
    next_vals = target_critic_net((Variable(torch.cat((joint_next_obs, torch.cat((next_acts), dim=1)), dim=1)))).detach()
    
    gamma = kwargs['gamma']
    critic_optim.zero_grad()

    target_vals = rewards + gamma*(1-dones)*next_vals
    loss_func = nn.MSELoss()
    loss = loss_func(predicted_q_val, target_vals)
    loss.backward()

    nn.utils.clip_grad_norm_(critic_net.parameters(), kwargs['clip_grad'])
    critic_optim.step()


def actorUpdate(agentIdx, states, actor_net, critic_net, actor_optim, acts, kwargs):

    agent_obs = states[agentIdx]
    predicted_action = step_with_exp([agent_obs], [actor_net], noise=None)[0]
    allActs = []
    for k in range(len(acts)):
        if k != agentIdx:

            allActs.append(acts[k])
        else:
            allActs.append(predicted_action)
    
    joint_obs = torch.cat(states, dim=1)
    qVal = critic_net((torch.cat((joint_obs, torch.cat(allActs, dim=1)), dim=1)))
    loss = -torch.mean(qVal)
    
    actor_optim.zero_grad()
    loss.backward()
    
    nn.utils.clip_grad_norm_(actor_net.parameters(), kwargs['clip_grad'])
    actor_optim.step()


def update(actor_nets, critic_nets, target_actor_nets, target_critic_nets, exp_replay, actor_optimizers,
           critic_optimizers, batch_size, kwargs):
    if exp_replay.__len__() > batch_size:
        idx = 0
        for actor_network, critic_network, actor_optim, critic_optim, target_actor_network, target_critic_network in \
                zip(actor_nets, critic_nets, actor_optimizers, critic_optimizers, target_actor_nets, target_critic_nets):
            obs, act, rews, next_obs, dones = exp_replay.sample(batch_size, norm_rews=True)

            criticUpdate(critic_network, critic_optim, target_actor_nets, target_critic_network, obs, next_obs, act, rews[idx], dones[idx], kwargs)
            actorUpdate(idx, obs, actor_network, critic_network, actor_optim, act, kwargs)

            idx += 1


def main(**kwargs):

    environment = gym.make("RoboSumo-Ant-vs-Ant-v0")
    observation_space = environment.observation_space
    action_space = environment.action_space
    noise = GaussianNoise(8)

    action_dim = [8, 8]
    obs_dim = [120, 120]

    MAX_EPS_STEPS = 500
    NUM_EPISODES = 125000
    UPDATE_FREQ = 100
    COPY_FREQ = 100

    actor_nets = [MLP(obs_dim[i], [128, 128, 128], action_dim[i]) for i in range(2)]
    critic_nets = [MLP((obs_dim[0]+obs_dim[1]+action_dim[0]+action_dim[1]), [128, 128, 128], 1, net_type="Critic") for i in range(2)]
    target_actor_nets = [MLP(obs_dim[i], [128, 128, 128], action_dim[i]) for i in range(2)]
    target_critic_nets = [MLP((obs_dim[0]+obs_dim[1]+action_dim[0]+action_dim[1]), [128, 128, 128], 1, net_type="Critic") for i in range(2)]

    actor_optimizers = [Adam(actor_net.parameters(), lr=kwargs['actor_lr']) for actor_net in actor_nets]
    critic_optimizers = [Adam(critic_net.parameters(), lr=kwargs['critic_lr']) for critic_net in critic_nets]

    for actor_network, target_network in zip(actor_nets, target_actor_nets):
        hard_update(target_network, actor_network)

    for critic_network, target_network in zip(critic_nets, target_critic_nets):
        hard_update(target_network, critic_network)

    exp_replay = ReplayBuffer(int(1e6), 2, obs_dim, action_dim)
    totalSteps = 0
    totalUpdates = 0

    #environment.render()
    for epsNum in range(NUM_EPISODES):
        numSteps = 0
        observation = environment.reset()
        observation = [[observation[0], observation[1]]]
        dones = [[False, False]]
        
        total_rewards = [0.0, 0.0]
        epsilon = 0.0

        while numSteps < MAX_EPS_STEPS and dones[0][0]==False and dones[0][1]==False:
            actions = []
            tens_obs = [Variable(torch.Tensor([elem]), requires_grad=False) for elem in observation[0]]
            actions = step_with_exp(tens_obs, actor_nets, noise=noise)
            actions = [a.detach() for a in actions]
            print("Actions : ",actions)
            action_numpy = [a[0].data.numpy() for a in actions]

            nextObs, rewards, dones, info = environment.step(action_numpy)
            numSteps += 1
            next_obs_given = [[nextObs[0], nextObs[1]]]
            dones = [[dones[0], dones[1]]]
            if numSteps == MAX_EPS_STEPS - 1:
                dones[0][0] = True
                dones[0][1] = True
            rewards = [[rewards[0], rewards[1]]]
            numSteps += 1
            totalSteps += 1

            total_rewards[0] += rewards[0][0]
            total_rewards[1] += rewards[0][1]
            environment.render(mode='human')
            exp_replay.push(np.asarray(observation), action_numpy, np.asarray(rewards), np.asarray(next_obs_given), np.asarray(dones))

            if totalSteps % UPDATE_FREQ == 0:
                print(totalSteps)
                update(actor_nets, critic_nets, target_actor_nets, target_critic_nets, exp_replay, actor_optimizers, critic_optimizers, 124, kwargs)
                for actor_network, target_network in zip(actor_nets, target_actor_nets):
                    soft_update(target_network, actor_network,kwargs['tau'])

                for critic_network, target_network in zip(critic_nets, target_critic_nets):
                    soft_update(target_network, critic_network, kwargs['tau'])
                totalUpdates += 1

            

            observation = nextObs
            observation = [[observation[0], observation[1]]]

        print("Total Reward episode ", epsNum," : ", total_rewards)
            #environment.render()
            




if __name__ == "__main__":
    main(
        episode_length=25,
        num_tasks=4,
        num_processes=1,
        MAX_STEPS=25000 * 25,
        copyFrequency=4000,
        update_nets=25,
        gamma=0.99,
        train_frequency=100,
        batch_size=1024,
        tau=0.001,
        critic_lr=0.00001,
        actor_lr=0.00001,
        init_steps=1024,
        clip_grad=0.5
    )
