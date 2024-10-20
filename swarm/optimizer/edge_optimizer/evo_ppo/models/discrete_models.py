import torch
import random
import torch.nn as nn
from torch.distributions import Normal, RelaxedOneHotCategorical, Categorical
import torch.nn.functional as F
from torch.nn import Dropout, Linear

from torch_geometric.nn.conv import GATConv

from swarm.optimizer.edge_optimizer.graph_net.layers import GraphAttentionLayer

from transformers import BertModel, BertTokenizer


# Load pre-trained BERT model and tokenizer
# Using a distilled BERT model (smaller and faster with fewer parameters)
bert = BertModel.from_pretrained('distilbert-base-uncased').to('cuda')
tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

# Freeze BERT model parameters
for param in bert.parameters():
    param.requires_grad = False

class CategoricalGATPolicy(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, num_node_features, action_dim, hidden_channels, potential_connections, num_heads=4):
        super(CategoricalGATPolicy, self).__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.action_dim = action_dim

        self.potential_connections = potential_connections

        self.conv1 = GATConv(num_node_features, hidden_channels,
                            heads=num_heads, dropout=0.6, residual=True)
        self.conv2 = GATConv(hidden_channels * num_heads,
                            hidden_channels, dropout=0.6, heads=1, concat=False, residual=True)

        # self.positional_encoding = nn.Parameter(
        #     torch.randn(1, num_node_features), requires_grad=True)

        # # Adjust for concatenated state indicator
        # self.state_indicator_fc = nn.Linear(
        #     num_node_features, num_node_features)
        
        self.fc0 = nn.Linear(1, hidden_channels)
        self.fc1 = nn.Linear(768, hidden_channels) # 768 is the size of BERT embeddings

        # Action head for predicting next action/node
        self.action_fc = nn.Linear(hidden_channels, action_dim)
        
        # Normalization layer for embedding (optional)
        self.norm_layer1 = nn.LayerNorm(hidden_channels * num_heads)
        self.norm_layer2 = nn.LayerNorm(hidden_channels)


    def clean_action(self, x: torch.Tensor, edge_index: torch.Tensor, active_node_idx: int, sentence: str, return_only_action=True, step=0, pruned_nodes=[], batch_size=1):
        
        # Encode sentence with BERT
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to('cuda')
        with torch.no_grad():
            sentence_embedding = bert(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
            sentence_embedding = sentence_embedding.to(x.device)
        
        # merge different state embeddings to a single sentence embedding
        steps = step + 1
        if batch_size == 1:
            steps = torch.tensor([steps], dtype=torch.float32).to(x.device)
        steps = steps.unsqueeze(1)
        steps = self.fc0(steps * 1e-2)  # Scale down step to avoid exploding gradients
        sentence_embedding = self.fc1(sentence_embedding)
        sentence_embedding = sentence_embedding + steps

        # Pass through GAT layers
        # x_res = x
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = self.norm_layer1(x)
        # # Match dimensions for residual connection
        # x_res = self.proj_residual(x_res, x.size(1))
        # x = x + x_res

        # x_res = x
        x, attention = self.conv2(
            x, edge_index, return_attention_weights=True)
        
        x = self.norm_layer2(x)
        # x_res = self.proj_residual(x_res, x.size(1))
        # x = x + x_res

        # # Optional normalization
        # x = self.norm_layer(x)
        
        # Predict action logits
        active_node_features = x[active_node_idx].unsqueeze(0)
                
        # unsqueeze sentence embedding if not same dimension as active node features
        if sentence_embedding.dim() != active_node_features.dim():
            # sentence_embedding = sentence_embedding.repeat(batch_size, 1)
            sentence_embedding = sentence_embedding.unsqueeze(0)
        
        active_node_features = active_node_features + sentence_embedding
        
        # if batch_size > 1:   
        #     active_node_features = torch.cat((active_node_features, sentence_embedding), dim=2)
        # else:
        #     active_node_features = torch.cat((active_node_features, sentence_embedding), dim=1)
            
        # active_node_features = active_node_features + (step + 1) * 1e-3 # Add step as positional encoding
        # active_node_features = self.fc0(active_node_features)
        
        action_logits = self.action_fc(active_node_features)       
        
        # check dimension of action logits
        if action_logits.size(1) != self.action_dim:
            action_logits = action_logits.squeeze(0)
        
        # set logits of pruned nodes to very low value
        for node in pruned_nodes:
            action_logits[:, node] = -1e9

        # Apply argmax on the correct dimension
        action = action_logits.argmax(dim=1)

        if return_only_action:
                return action

        return action, x, attention, action_logits

    def noisy_action(self, x: torch.Tensor, edge_index: torch.Tensor, active_node_idx: int, sentence: str, return_only_action=True, step: int=0, pruned_nodes=[], batch_size=1):
        
        _, x, attention, logits = self.clean_action(
            x, edge_index, active_node_idx, sentence, return_only_action=False, step=step, pruned_nodes=pruned_nodes, batch_size=batch_size)

        dist = Categorical(logits=logits)
        action = dist.sample()
        action = action

        if return_only_action:
            return action

        return action, x, attention, logits

    # def proj_residual(self, x_res, target_dim):
    #     """Projects the residual connection to match the target dimension."""
    #     if x_res.size(1) != target_dim:
    #         x_res = nn.Linear(x_res.size(1), target_dim,
    #                           bias=False).to(x_res.device)(x_res)
    #     return x_res


class CategoricalPolicy(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim, hidden_size):
        super(CategoricalPolicy, self).__init__()
        self.action_dim = action_dim

        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.f1 = nn.Linear(state_dim, hidden_size)
        # self.q1ln1 = nn.LayerNorm(l1)

        # Hidden Layer 2
        self.f2 = nn.Linear(hidden_size, hidden_size)
        # self.q1ln2 = nn.LayerNorm(l2)

        # Value
        self.val = nn.Linear(hidden_size, 1)

        # Advantages
        self.adv = nn.Linear(hidden_size, action_dim)

    def clean_action(self, obs, return_only_action=True):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        ###### Feature ####
        info = torch.relu(self.f1(obs))
        info = torch.relu(self.f2(info))

        val = self.val(info)
        adv = self.adv(info)

        logits = val + adv - adv.mean()

        if return_only_action:
            return logits.argmax(1)

        return None, None, logits

    def noisy_action(self, obs, return_only_action=True):
        _, _, logits = self.clean_action(obs, return_only_action=False)

        dist = Categorical(logits=logits)
        action = dist.sample()
        action = action

        if return_only_action:
            return action

        return action, None, logits


class GumbelPolicy(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim, hidden_size, epsilon_start, epsilon_end, epsilon_decay_frames):
        super(GumbelPolicy, self).__init__()
        self.action_dim = action_dim

        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.f1 = nn.Linear(state_dim, hidden_size)
        # self.q1ln1 = nn.LayerNorm(l1)

        # Hidden Layer 2
        self.f2 = nn.Linear(hidden_size, hidden_size)
        # self.q1ln2 = nn.LayerNorm(l2)

        # Value
        self.val = nn.Linear(hidden_size, 1)

        # Advantages
        self.adv = nn.Linear(hidden_size, action_dim)

        # Temperature
        self.log_temp = torch.nn.Linear(hidden_size, 1)

        self.LOG_TEMP_MAX = 2
        self.LOG_TEMP_MIN = -10

    def clean_action(self, obs, return_only_action=True):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        ###### Feature ####
        info = torch.relu(self.f1(obs))
        info = torch.relu(self.f2(info))

        val = self.val(info)
        adv = self.adv(info)

        logits = val + adv - adv.mean()

        if return_only_action:
            return logits.argmax(1)
        else:
            log_temp = self.log_temp(info)
            log_temp = torch.clamp(
                log_temp, min=self.LOG_TEMP_MIN, max=self.LOG_TEMP_MAX)

            return logits.argmax(1), log_temp, logits

    def noisy_action(self, obs, return_only_action=True):
        _, log_temp, logits = self.clean_action(obs, return_only_action=False)

        temp = log_temp.exp()
        dist = RelaxedOneHotCategorical(
            temperature=temp, probs=F.softmax(logits, dim=1))
        action = dist.rsample()

        if return_only_action:
            return action.argmax(1)

        log_prob = dist.log_prob(action)
        log_prob = torch.diagonal(log_prob, offset=0).unsqueeze(1)

        return action.argmax(1), log_prob, logits
