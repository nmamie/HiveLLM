import os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.optim import Adam
from swarm.optimizer.edge_optimizer.evo_ppo.core.utils import soft_update, hard_update
from swarm.optimizer.edge_optimizer.evo_ppo.core import utils as utils



class DDQN(object):
    def __init__(self, args, model_constructor):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = model_constructor.make_model('CategoricalPolicy').to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.actor_target = model_constructor.make_model('CategoricalPolicy').to(device=self.device)
        hard_update(self.actor_target, self.actor)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_updates = 0

    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, active_node_batch, edge_index_batch, done_batch, batch_size, node_feature_size, num_nodes, num_edges):
        
        state_batch = torch.reshape(state_batch, (batch_size, num_nodes, node_feature_size))
        next_state_batch = torch.reshape(next_state_batch, (batch_size, num_nodes, node_feature_size))
        edge_index_batch = torch.reshape(edge_index_batch, (batch_size, 2, num_edges))
        
        state_batch = state_batch.to(self.device)
        next_state_batch=next_state_batch.to(self.device)
        action_batch=action_batch.to(self.device)
        reward_batch=reward_batch.to(self.device)
        active_node_batch=active_node_batch.to(self.device)
        edge_index_batch=edge_index_batch.to(self.device)
        done_batch=done_batch.to(self.device)
        action_batch = action_batch.long().unsqueeze(1)
        with torch.no_grad():            
            # Assuming each graph has its own node features and edge_index
            data_list = []
            for i in range(batch_size):
                # Example node features: 3 nodes, each with 64 features
                x = next_state_batch[i]
                # Example edge index for this graph
                edge_index = edge_index_batch[i].long()
                
                active_node = active_node_batch[i].long()

                # Create a Data object for each graph
                data = Data(x=x, edge_index=edge_index, active_node=active_node)
                data_list.append(data)

            # Batch the graphs
            batch = Batch.from_data_list(data_list)
                        
            na = self.actor.clean_action(batch.x, batch.edge_index, batch.active_node, "", return_only_action=True, batch_size=batch_size)
            _, _, _, ns_logits = self.actor_target.noisy_action(batch.x, batch.edge_index, batch.active_node, "", return_only_action=False, batch_size=batch_size)
            next_entropy = -(F.softmax(ns_logits, dim=1) * F.log_softmax(ns_logits, dim=1)).mean(1).unsqueeze(1)
            
            ns_logits = ns_logits.gather(1, na.unsqueeze(1))
            
            next_target = ns_logits + self.alpha * next_entropy
            
            next_q_value = reward_batch + (1-done_batch) * self.gamma * next_target

    
        # Assuming each graph has its own node features and edge_index
        data_list = []
        for i in range(batch_size):
            # Example node features: 3 nodes, each with 64 features
            x = state_batch[i]
            # Example edge index for this graph
            edge_index = edge_index_batch[i].long()
            
            active_node = active_node_batch[i].long()

            # Create a Data object for each graph
            data = Data(x=x, edge_index=edge_index, active_node=active_node)
            data_list.append(data)

        # Batch the graphs
        batch = Batch.from_data_list(data_list)

        _, _, _, logits  = self.actor.noisy_action(batch.x, batch.edge_index, batch.active_node, "", return_only_action=False, batch_size=batch_size)
        entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).mean(1).unsqueeze(1)
        q_val = logits.gather(1, action_batch)

        q_loss = (next_q_value - q_val)**2
        q_loss -= self.alpha*entropy
        q_loss = q_loss.mean()


        self.actor_optim.zero_grad()
        q_loss.backward()
        self.actor_optim.step()

        self.num_updates += 1
        soft_update(self.actor_target, self.actor, self.tau)