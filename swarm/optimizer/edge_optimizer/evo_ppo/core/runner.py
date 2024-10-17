
from sre_parse import State
from swarm.optimizer.edge_optimizer.evo_ppo.core import utils as utils
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy


# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, env_constructor):
    print(f"Rollout worker {id} started")
    env = env_constructor.make_env()
    np.random.seed(id) ###make sure the random seeds across learners are different
    
    # edge_index = env.edge_index

    ###LOOP###
    reward_history = {}
    action_history = {}
    pruned_nodes = []
    while True:
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        net = model_bucket[identifier]
        
        fitness = 0.0
        total_frame = 0
        state, edge_index, active_node_idx, records = env.reset()
        
        rollout_trajectory = []
        # state = utils.to_tensor(state)
        for record in records:
            done = False
            steps = 0
            while True:  # unless done    
                sentence = record['question']
                
                # state[active_node_idx] += net.state_indicator_fc(state[active_node_idx])
                           
                # ---- CALL GAT NETWORK FOR ACTION SELECTION ----
                if type == 'pg':
                    action, x, attention, logits = net.noisy_action(state, edge_index, active_node_idx, sentence, return_only_action=False, step=steps, pruned_nodes=pruned_nodes)  # Choose an action from the policy network
                else:
                    action, x, attention, logits = net.clean_action(state, edge_index, active_node_idx, sentence, return_only_action=False, step=steps, pruned_nodes=pruned_nodes)

                # softmax
                action_logits_soft = F.softmax(logits, dim=1)
                print("Action logits:", action_logits_soft)
                
                # Simulate one step in environment
                next_state, active_node_idx, reward, done, info = env.step(action.flatten(), record, state, edge_index)
                
                fitness += reward

                # ---- STORE TRANSITIONS IF NEEDED ----
                if store_data:  # Skip for test set
                    rollout_trajectory.append([
                        np.array([state]), np.array([next_state]),
                        np.float32(action), np.float32(np.array([reward])), np.float32(np.array([active_node_idx])),
                        np.float32(np.array([steps])), np.array([edge_index]), np.float32(np.array([float(done)]))
                    ])
                
                state = next_state
                total_frame += 1
                steps += 1

                # # Display the attention and logits
                # print(f"Attention {id}: ", attention)
                # print(f"Logits {id}: ", logits)

                # Record the rewards for each node
                if action.item() not in reward_history:
                    reward_history[action.item()] = []
                reward_history[action.item()].append(reward)

                # Record the actions
                if action.item() not in action_history:
                    action_history[action.item()] = 0
                action_history[action.item()] += 1

                # DONE FLAG IS Received
                if done:
                    break
        
        fitness /= len(records)
        
        # sort the reward and action history by key
        reward_history = dict(sorted(reward_history.items()))
        action_history = dict(sorted(action_history.items()))
        
        # package reward history and action history for return
        rewards_dist = {k: sum(v) / len(v) for k, v in reward_history.items()}
        action_dist = action_history
        
        print(f"Averaged reward per node {id}: ", rewards_dist)
        print(f"Action distribution {id}: ", action_dist)
        
        # nodes that do not perform well should be pruned
        if store_data:
            for node in rewards_dist:
                if rewards_dist[node] < -3.0 and action_dist[node] > 100: # adjust this threshold
                    if node not in pruned_nodes and len(pruned_nodes) < len(rewards_dist) - 1:
                        pruned_nodes.append(node)
            print(f"Pruned nodes {id}: ", pruned_nodes)
        
        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame, rollout_trajectory])