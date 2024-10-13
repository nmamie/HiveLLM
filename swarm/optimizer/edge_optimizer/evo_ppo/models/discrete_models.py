import torch, random
import torch.nn as nn
from torch.distributions import  Normal, RelaxedOneHotCategorical, Categorical
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
        
        # initial_probability = 0.5
        # init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        # init_tensor = torch.ones(
        #     len(potential_connections),
        #     requires_grad=True) * init_logit
        # self.edge_logits = torch.nn.Parameter(init_tensor)
        
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, dropout=0.6, heads=1, concat=False)
        # self.fc0 = Linear(768, num_node_features)
        # self.fc1 = Linear(num_node_features * 2, num_node_features) 
         
        # Learnable positional encoding for the active node
        self.positional_encoding = nn.Parameter(torch.randn(1, num_node_features), requires_grad=True)
        
        # Binary state indicator
        self.state_indicator_fc = nn.Linear(1, num_node_features)
        
        # Action head (for predicting the next node/action)
        self.action_fc = nn.Linear(hidden_channels, 1)          
        
            
    def clean_action(self, x: torch.Tensor, edge_index: torch.Tensor, sentence: str, return_only_action=True, batch_size=1):
        # Pass through the GAT layers
        x_res = x  # Save initial embeddings for residual connection
        
        # conv1: GAT with multiple heads, output shape (num_nodes, hidden_channels * num_heads)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Project the original input (x_res) to the same dimensionality as the output of conv1
        x_res = self.proj_residual(x_res, x.size(1))  # Project to match hidden_channels * num_heads
        
        x = x + x_res  # Residual connection
        x_res = x  # Save intermediate embeddings for residual connection
        
        # conv2: GAT with single head (concat=False), output shape (num_nodes, hidden_channels)
        x, attention = self.conv2(x, edge_index, return_attention_weights=True)
        
        # Add residual connection
        x_res = self.proj_residual(x_res, x.size(1))  # Ensure dimensionality matches hidden_channels
        x = x + x_res
        
        # Apply normalization to prevent exploding logits
        x = F.normalize(x, p=2, dim=-1)

        # Action head to predict the next node/action
        action_logits = self.action_fc(x)  # Evaluate actions for the active node
        action_logits = action_logits.view(batch_size, -1)  # Reshape to (batch_size, num_nodes)
        
        # Apply argmax on the correct dimension
        action = action_logits.argmax(dim=1)
        
        if return_only_action:
            return action
        
        return action, x, attention, action_logits



    def noisy_action(self, x: torch.Tensor, edge_index: torch.Tensor, sentence: str, return_only_action=True, batch_size=1):
        _, x, attention, logits = self.clean_action(x, edge_index, sentence, return_only_action=False, batch_size=batch_size)

        dist = Categorical(logits=logits)
        action = dist.sample()
        action = action

        if return_only_action:
            return action

        return action, x, attention, logits
    
    def proj_residual(self, x_res, target_dim):
        """Projects the residual connection to match the target dimension."""
        if x_res.size(1) != target_dim:
            x_res = nn.Linear(x_res.size(1), target_dim, bias=False).to(x_res.device)(x_res)
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
        #self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(hidden_size, hidden_size)
        #self.q1ln2 = nn.LayerNorm(l2)

        #Value
        self.val = nn.Linear(hidden_size, 1)

        #Advantages
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
        #self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(hidden_size, hidden_size)
        #self.q1ln2 = nn.LayerNorm(l2)

        #Value
        self.val = nn.Linear(hidden_size, 1)

        #Advantages
        self.adv = nn.Linear(hidden_size, action_dim)

        #Temperature
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
            log_temp = torch.clamp(log_temp, min=self.LOG_TEMP_MIN, max=self.LOG_TEMP_MAX)

            return logits.argmax(1), log_temp, logits

    def noisy_action(self, obs, return_only_action=True):
        _, log_temp, logits = self.clean_action(obs, return_only_action=False)

        temp = log_temp.exp()
        dist = RelaxedOneHotCategorical(temperature=temp, probs=F.softmax(logits, dim=1))
        action = dist.rsample()

        if return_only_action:
            return action.argmax(1)

        log_prob = dist.log_prob(action)
        log_prob = torch.diagonal(log_prob, offset=0).unsqueeze(1)


        return action.argmax(1), log_prob, logits



