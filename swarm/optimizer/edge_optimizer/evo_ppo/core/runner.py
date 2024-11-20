
import random
import re
from sre_parse import State

from sklearn.metrics import euclidean_distances
from swarm.optimizer.edge_optimizer.evo_ppo.core import utils as utils
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy


# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, env_constructor):
    print(f"Rollout worker {id} started")
    if type == 'test':
        env, num_envs = env_constructor.make_env(test=True)
    else:
        env, num_envs = env_constructor.make_env(test=False)
        
    np.random.seed(id); torch.manual_seed(id); random.seed(id) ###make sure the random seeds across learners are different
    
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    
    # edge_index = env.edge_index

    ###LOOP###
    action_history = {}
    reward_hist = {}
    pruned_nodes = []
    while True:
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        net = model_bucket[identifier]
        
        fitness = 0.0
        total_frame = 0        
        rollout_trajectory = []
        # state = utils.to_tensor(state)
        for i in range(num_envs):
            state, edge_index, init_active_node_idx, record = env.reset()
            record, sentence_emb = record
            terminate = False
            truncate = False
            done = False
            steps = 0
            active_node_idx = init_active_node_idx
            sentence_emb_traj = sentence_emb.cpu().numpy()
            
            while True:  # unless done    
                # sentence = record['question']
                                
                # # edge index is not changed
                # edge_index = deepcopy(init_edge_index)
                                
                # if type != 'test':
                
                # # remove pruned nodes from edge_index
                # if len(pruned_nodes) > 0:
                #     edge_index = utils.remove_pruned_nodes(edge_index, pruned_nodes)
                            
                            
                    # # randomly delete some edges in the edge_index
                    # for i in range(len(edge_index[0])):
                    #     if random.random() < 0.1:
                    #         edge_index[0][i] = -1
                    #         edge_index[1][i] = -1
                
                # state[active_node_idx] += net.state_indicator_fc(state[active_node_idx])
                           
                # ---- CALL GAT NETWORK FOR ACTION SELECTION ----
                if type == 'pg': action, x, attention, logits = net.noisy_action(state, edge_index, active_node_idx, sentence_emb, return_only_action=False, step=steps, pruned_nodes=[])  # Choose an action from the policy network
                else: action, x, attention, logits = net.clean_action(state, edge_index, active_node_idx, sentence_emb, return_only_action=False, step=steps, pruned_nodes=[])
                
                # # softmax
                # print("Action logits:", logits)
                # action_logits_soft = F.softmax(logits, dim=1)
                # print("Action logits soft:", action_logits_soft)
                
                # print(f"Action {id}: {action.flatten()}")
                
                # Simulate one step in environment
                next_state, next_active_node_idx, reward, terminate, truncate, info = env.step(action.flatten(), record, edge_index)
                
                done = terminate or truncate
                
                fitness += reward
                
                # ---- STORE TRANSITIONS IF NEEDED ----
                if store_data:  # Skip for test set
                    # store the rollout trajectory
                    rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(next_state), np.array([sentence_emb_traj]),
                        np.float32(action), np.reshape(np.float32(np.array([reward])), (1, 1)), np.reshape(np.float32(np.array([active_node_idx])), (1, 1)), np.reshape(np.float32(np.array([next_active_node_idx])), (1, 1)),
                        np.reshape(np.float32(np.array([steps])), (1, 1)), np.array([edge_index]), np.reshape(np.float32(np.array([float(terminate)])), (1, 1))
                    ])
                
                state = deepcopy(next_state)
                active_node_idx = deepcopy(next_active_node_idx)
                total_frame += 1
                steps += 1

                # # Display the attention and logits
                # print(f"Attention {id}: ", attention)
                # print(f"Logits {id}: ", logits)
                
                if steps == 1:                    
                    # Record the actions
                    if init_active_node_idx not in action_history:
                        action_history[init_active_node_idx] = 0
                    action_history[init_active_node_idx] += 1
                    
                    # Record the rewards
                    if init_active_node_idx not in reward_hist:
                        reward_hist[init_active_node_idx] = 0.0
                    reward_hist[init_active_node_idx] += reward
                
                # Record the actions
                if action.item() not in action_history:
                    action_history[action.item()] = 0
                action_history[action.item()] += 1
                
                # Record the rewards
                if action.item() not in reward_hist:
                    reward_hist[action.item()] = 0.0
                reward_hist[action.item()] += reward
                
                # # Record the attention
                # attention_history.append(attention[1])

                # DONE FLAG IS Received
                if done:
                    state = None
                    next_state = None
                    active_node_idx = None
                    break
                
                # clean up cuda memory
                torch.cuda.empty_cache()
        
        fitness /= num_envs # average fitness over the number of environments
        fitness *= 10.0 # scale the fitness to reward scheme
        
        # sort action and reward history by key
        action_dist = dict(sorted(action_history.items()))
        # rewards multiplied by 100 and averaged to make it more readable
        rewards_dist = {k: v / action_dist[k] * 10 for k, v in dict(sorted(reward_hist.items())).items()}
        
        print(f"Action distribution {id}: ", action_dist)
        print(f"Reward distribution {id}: ", rewards_dist)
        # print(f"Attention distribution {id}: ", attention_dist)
        
        # nodes that do not perform well should be pruned from starting nodes
        # if store_data:
        # changes = False
        # pruned_nodes = []
        # for node in rewards_dist:
        #     if rewards_dist[node] <= -1.0 and action_dist[node] > 100: # adjust this threshold
        #         if node not in pruned_nodes and len(pruned_nodes) < len(rewards_dist) - 1:
        #             pruned_nodes.append(node)
        #             changes = True
        # print(f"Pruned nodes {id}: ", pruned_nodes)
        
        # if changes is True:
        #     env.prune(pruned_nodes)
        # pruned_nodes = [] # TODO: to prune or not to prune, that is the question
        
        # clean up cuda memory
        torch.cuda.empty_cache()
        
        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame, pruned_nodes, rollout_trajectory])