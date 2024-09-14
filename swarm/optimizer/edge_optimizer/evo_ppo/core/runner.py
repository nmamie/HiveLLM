
from swarm.optimizer.edge_optimizer.evo_ppo.core import utils as utils
import numpy as np
import torch
from copy import deepcopy


# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, env_constructor):
    
    env = env_constructor.make_env()
    np.random.seed(id) ###make sure the random seeds across learners are different
    
    edge_index = env.edge_index

    ###LOOP###
    while True:
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        net = model_bucket[identifier]

        fitness = 0.0
        total_frame = 0
        state, record = env.reset()
        rollout_trajectory = []
        # state = utils.to_tensor(state)
        while True:  # unless done
            sentence = record['question']
            if type == 'pg': action = net.noisy_action(state, edge_index, sentence)  # Choose an action from the policy network
            else: action = net.clean_action(state, edge_index, sentence)
            
            # action = utils.to_numpy(action)
            next_state, reward, done, info, record = env.step(action.flatten(), state)  # Simulate one step in environment
            
            # next_state = utils.to_tensor(next_state)
            fitness += reward
            
            # If storing transitions
            if store_data: #Skip for test set
                rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(next_state),
                                        np.float32(action), np.reshape(np.float32(np.array([reward])), (1, 1)),
                                           np.reshape(np.float32(np.array([float(done)])), (1, 1))])
            state = next_state
            total_frame += 1

            # DONE FLAG IS Received
            if done:
                break

        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame, rollout_trajectory])
