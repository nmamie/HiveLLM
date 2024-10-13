
from sre_parse import State
from swarm.optimizer.edge_optimizer.evo_ppo.core import utils as utils
import numpy as np
import torch
from copy import deepcopy


# Rollout evaluate an agent in a complete game
@torch.no_grad()
def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, env_constructor):
    print(f"Rollout worker {id} started")
    env = env_constructor.make_env()
    np.random.seed(id) ###make sure the random seeds across learners are different
    
    # edge_index = env.edge_index

    ###LOOP###
    attention_history = []
    reward_history = {}
    action_history = {}
    while True:
        identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
        if identifier == 'TERMINATE': exit(0) #Exit

        # Get the requisite network
        net = model_bucket[identifier]
        
        # # realize graph
        # edges = net.potential_connections
        
        # edge_logits = torch.zeros(
        #     len(edges),
        #     requires_grad=False)
        # realized_graph, log_probs = env.swarm.connection_dist.realize_particle(env.swarm.composite_graph, edge_logits)
        # env.env = realized_graph
        # build edge_index
        # # Unique nodes
        # nodes = list(set([node for edge in edges for node in edge]))
        # # Create a mapping from node labels to indices
        # node_to_index = {node: i for i, node in enumerate(nodes)}
        # # Convert the edges to index format
        # edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges], dtype=torch.long, requires_grad=False).t()
        fitness = 0.0
        total_frame = 0
        state, edge_index, active_node_idx, records = env.reset()
        
        # ---- ADD POSITIONAL ENCODING AND BINARY STATE INDICATOR ----        
        # Apply binary state indicator for the active node
        state_indicator = torch.zeros(state.size(0), 1)
        state_indicator[active_node_idx] = 1  # Mark active node
        state += net.state_indicator_fc(state_indicator)  # Incorporate binary state indicator

        # Apply learnable positional encoding to the active node
        state[active_node_idx] += net.positional_encoding.squeeze(0)
        
        rollout_trajectory = []
        # state = utils.to_tensor(state)
        for record in records:
            done = False
            while True:  # unless done    
                sentence = record['question']
                           
                # ---- CALL GAT NETWORK FOR ACTION SELECTION ----
                if type == 'pg':
                    action, next_state, attention, logits = net.noisy_action(state, edge_index, sentence, return_only_action=False)  # Choose an action from the policy network
                else:
                    action, next_state, attention, logits = net.clean_action(state, edge_index, sentence, return_only_action=False)

                # Simulate one step in environment
                next_state, active_node_idx, reward, done, info = env.step(action.flatten(), record, state, edge_index)
                
                # ---- ADD POSITIONAL ENCODING AND BINARY STATE INDICATOR ----
                # Apply binary state indicator for the new active node
                state_indicator = torch.zeros(next_state.size(0), 1)
                state_indicator[active_node_idx] = 1  # Mark active node
                next_state += net.state_indicator_fc(state_indicator)  # Incorporate binary state indicator
                
                # Apply learnable positional encoding to the active node
                next_state[active_node_idx] += net.positional_encoding.squeeze(0)
                
                fitness += reward

                # ---- STORE TRANSITIONS IF NEEDED ----
                if store_data:  # Skip for test set
                    rollout_trajectory.append([
                        utils.to_numpy(state), utils.to_numpy(next_state),
                        np.float32(action), np.float32(np.array([reward])),
                        utils.to_numpy(edge_index), np.float32(np.array([float(done)]))
                    ])
                
                state = next_state
                total_frame += 1

                # Record the attention
                attention_history.append(attention[1])

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
        attention_dist = torch.mean(torch.stack(attention_history), dim=0)
        
        print(f"Averaged reward per node {id}: ", rewards_dist)
        print(f"Action distribution {id}: ", action_dist)
        print(f"Averaged attention {id}: ", attention_dist)

        
        
        # Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([identifier, fitness, total_frame, rollout_trajectory, rewards_dist, action_dist, attention_dist])