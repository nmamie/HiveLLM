import torch
import random
import torch.nn as nn
from torch.distributions import Normal, RelaxedOneHotCategorical, Categorical
import torch.nn.functional as F
from torch.nn import Dropout, Linear

from torch_geometric.nn.conv import GATConv

from swarm.optimizer.edge_optimizer.graph_net.layers import GraphAttentionLayer

from transformers import BertModel, BertTokenizer


# # Load pre-trained BERT model and tokenizer
# # Using a distilled BERT model (smaller and faster with fewer parameters)
# bert = BertModel.from_pretrained('distilbert-base-uncased').to('cuda') # type: ignore
# tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

# # Freeze BERT model parameters
# for param in bert.parameters():
#     param.requires_grad = False


class CategoricalGATPolicy(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, num_node_features, action_dim, hidden_channels, potential_connections, num_heads=8):
        super(CategoricalGATPolicy, self).__init__()

        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.action_dim = action_dim

        self.potential_connections = potential_connections

        self.conv1 = GATConv(num_node_features, hidden_channels,
                            heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads,
                            hidden_channels, dropout=0.6, heads=1, concat=False)

        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_node_features), requires_grad=True)

        # Adjust for concatenated state indicator
        self.state_indicator_fc = nn.Linear(
            num_node_features + 1, num_node_features)

        # Action head for predicting next action/node
        self.action_fc = nn.Linear(hidden_channels, 1)

        # Normalization layer for embedding (optional)
        self.norm_layer = nn.LayerNorm(hidden_channels)  # Or use nn.BatchNorm1d


    def clean_action(self, x: torch.Tensor, edge_index: torch.Tensor, active_node_idx: int, sentence: str, return_only_action=True, batch_size=1):
        # Binary state indicator for the active node
        state_indicator = torch.zeros(x.size(0), 1).to(x.device)
        state_indicator[active_node_idx] = 1
        x = torch.cat([x, state_indicator], dim=1)  # Concatenate binary indicator

        x = self.state_indicator_fc(x)  # Apply state indicator linear layer

        # Apply learnable positional encoding (broadcast to all, enhance active node)
        x += self.positional_encoding
        # Enhance active node
        x[active_node_idx] += self.positional_encoding.squeeze(0)

        # Pass through GAT layers
        x_res = x
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Match dimensions for residual connection
        x_res = self.proj_residual(x_res, x.size(1))
        x = x + x_res

        x_res = x
        x, attention = self.conv2(
            x, edge_index, return_attention_weights=True)
        x_res = self.proj_residual(x_res, x.size(1))
        x = x + x_res

        # Optional normalization
        x = self.norm_layer(x)

        # Predict action logits
        action_logits = self.action_fc(x).view(batch_size, -1)

        # Apply argmax on the correct dimension
        action = action_logits.argmax(dim=1)

        if return_only_action:
                return action

        return action, x, attention, action_logits

    def noisy_action(self, x: torch.Tensor, edge_index: torch.Tensor, active_node_idx: int, sentence: str, return_only_action=True, batch_size=1):
        _, x, attention, logits = self.clean_action(
            x, edge_index, active_node_idx, sentence, return_only_action=False, batch_size=batch_size)

        dist = Categorical(logits=logits)
        action = dist.sample()
        action = action

        if return_only_action:
            return action

        return action, x, attention, logits

    def proj_residual(self, x_res, target_dim):
        """Projects the residual connection to match the target dimension."""
        if x_res.size(1) != target_dim:
            x_res = nn.Linear(x_res.size(1), target_dim,
                              bias=False).to(x_res.device)(x_res)
        return x_res


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
