from platform import node
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear
from typing import Tuple

from torch_geometric.nn.conv import GATConv

from swarm.optimizer.edge_optimizer.graph_net.layers import GraphAttentionLayer

from transformers import BertModel, BertTokenizer

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
    
class EdgeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        """
        Args:
            input_dim (int): Number of input features.
                             For example, if you concatenate x_i, x_j, |x_i - x_j|, x_i * x_j,
                             and each node embedding has dimension d, then input_dim = 4*d.
            hidden_dim (int): Hidden layer size. You can experiment with this value.
            dropout (float): Dropout probability for regularization.
        """
        super(EdgeMLP, self).__init__()
        
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Optional: helps with training stability
        
        # Second hidden layer (reducing dimensionality)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Final output layer (predicts a single score)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        edge_score = self.fc3(x)
        return edge_score
    
    
class GATWithSentenceEmbedding(torch.nn.Module):
    def __init__(self, num_potential_edges, num_node_features, hidden_channels, sentence_embedding_dim, num_heads=1):
        super(GATWithSentenceEmbedding, self).__init__()
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_potential_edges = num_potential_edges
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.sentence_embedding_dim = sentence_embedding_dim
        
        initial_probability = 0.5
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            len(num_potential_edges)) * init_logit
        self.edge_logits = torch.nn.Parameter(init_tensor, requires_grad=True)
        
        self.conv1 = GATConv(sentence_embedding_dim, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, num_node_features, heads=1, dropout=0.6, concat=False)
        
        self.fc0 = Linear(768, sentence_embedding_dim)
        self.fc1 = Linear(num_node_features, sentence_embedding_dim)
        self.fc2 = Linear(2*len(num_potential_edges), len(num_potential_edges))
        self.fce = Linear(768, len(num_potential_edges))
        self.fcl = Linear(len(num_potential_edges), len(num_potential_edges))
        # self.fc3 = Linear(len(num_potential_edges), hidden_channels)
        # self.fc4 = Linear(hidden_channels, len(num_potential_edges)) # logistic regression
        self.edge_mlp = EdgeMLP(input_dim=4*num_node_features, hidden_dim=hidden_channels)
                
        # Load pre-trained BERT model and tokenizer
        # Using a distilled BERT model (smaller and faster with fewer parameters)
        self.bert = BertModel.from_pretrained('distilbert-base-uncased').to(self.device) # type: ignore
        self.tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT model parameters except for the final layer
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # # unfreeze the last layer
        # for param in self.bert.encoder.layer[-1].parameters():
        #     param.requires_grad = True
                        
        # self.dropout = Dropout(0.6).to(self.device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sentence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_index = edge_index.to(self.device)
        orig_edge_index = edge_index.clone()
        x = x.to(self.device)
        
        # print("Inputs:", x)
        
        # # normalize node features
        # x = F.normalize(x, p=2, dim=-1)
        
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            orig_sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
            orig_sentence_embedding = orig_sentence_embedding.squeeze(0)
            
        # print("Node features shape:", x.shape)
        # print("Edge index shape:", edge_index.shape)
        # print("Sentence embedding shape:", sentence_embedding.shape)
        
        sentence_embedding = self.fc0(orig_sentence_embedding)
        sentence_embedding = F.relu(sentence_embedding)
        
        orig_edge_logits = self.fcl(self.edge_logits)
        orig_edge_logits = F.relu(orig_edge_logits)
        print(f"Edge probs (init): {torch.sigmoid(orig_edge_logits)}")
        
        sentence_embedding_edge = self.fce(orig_sentence_embedding)
        sentence_embedding_edge = F.relu(sentence_embedding_edge)
        print(f"Sentence embedding edge: {sentence_embedding_edge}")
        orig_edge_logits = torch.cat([orig_edge_logits, sentence_embedding_edge], dim=0)
        orig_edge_logits = self.fc2(orig_edge_logits)
        edge_probs = torch.sigmoid(orig_edge_logits)
        
        print(f"Edge probs for graph: {edge_probs}")
                
        # construct edge index based on query and edge logits
        if self.training:
            print("Training mode")
            edge_index = torch.stack([edge_index[0], edge_index[1], edge_probs], dim=0)
            edge_index = edge_index[:, edge_index[2] > torch.rand(edge_index.shape[1]).to(self.device)]
            edge_index = edge_index[:2]
            edge_index = edge_index.to(torch.int64)
        else:
            print("Inference mode")
            edge_index = torch.stack([edge_index[0], edge_index[1], edge_probs], dim=0)
            edge_index = edge_index[:, edge_index[2] > torch.tensor(0.5).to(self.device)]
            edge_index = edge_index[:2]
            edge_index = edge_index.to(torch.int64)
            
        print(f"Edge index: {edge_index}")
            
        # sentence_embedding_repeat = sentence_embedding.repeat(x.shape[0], 1)
        # x = torch.cat([x, sentence_embedding_repeat], dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = x + sentence_embedding
        # x_reduced = self.fc1(x)
        # x_reconstructed = self.fc2(x_reduced)
        # print("Reduced node features:", x_reduced)
                
        # Pass through GAT layers as before
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x, attention = self.conv2(x, edge_index.to(self.device), return_attention_weights=True)
        
        print(f"Attention weights: {attention}")

        # Get node embeddings for each edge (source and target nodes)
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]

        # Create edge features by concatenating various combinations along with the attention weight
        edge_features = torch.cat([
            x_i, 
            x_j, 
            torch.abs(x_i - x_j), 
            x_i * x_j,  # Element-wise product
        ], dim=1)

        # Predict edge score with a learnable MLP
        edge_score = self.edge_mlp(edge_features)

        
        print(f"Edge score: {edge_score}")
        
        edge_logits = []
        j = 0
        for e in orig_edge_index[0]:
            if e == edge_index[0][j]:
                # if edge_score[j] <= 0.0:
                #     edge_logits.append(0.1)
                # else:
                edge_logits.append(edge_score[j])
                j = j + 1
                if j == len(edge_index[0]):
                    while len(edge_logits) < len(orig_edge_index[0]):
                        edge_logits.append(-2.5)
                    break
            else:
                edge_logits.append(-2.5)
                
        edge_logits = torch.tensor(edge_logits).to(self.device)
        print(f"Edge logits (pre-sigmoid): {edge_logits}")     

        # # Pass through neural network layers for edge prediction
        # edge_logits = self.fc3(edge_logits)
        # edge_logits = nn.ReLU()(edge_logits)

        # # Optionally apply Dropout for regularization (good during training)
        # edge_logits = nn.Dropout(0.2)(edge_logits)

        # # Final layer for binary classification
        # edge_logits = self.fc4(edge_logits)
        # print(f"Edge logits (post-sigmoid): {edge_logits}")

        # Apply sigmoid to get probabilities
        edge_probs = torch.sigmoid(edge_logits)  # This gives values between 0 and 1
        print(f"Edge probabilities (post-sigmoid): {edge_probs}")
                
        # # define edge probabilities as attention weights
        # attention_scores = []
        # # filter out self-loops using edge_index
        # for i, (src, dst) in enumerate(edge_index.t().tolist()):
        #     if src != dst:
        #         attention_scores.append(attention[1][i])
        
        # attention_scores = torch.stack(attention_scores)
        
        # # Min-max scaling to [0, 1]
        # min_val = attention_scores.min()
        # max_val = attention_scores.max()
        # if min_val < 0 or max_val > 1:
        #     attention_scores = (attention_scores - min_val) / (max_val - min_val + 1e-8)     
        
        return edge_probs, orig_edge_logits
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.fc = nn.Linear(in_channels * 2, 1) # logistic regression
        
    def forward(self, x_i, x_j):
        # Concatenate node embeddings
        edge_input = torch.cat([x_i, x_j], dim=1)
        
        # Pass through the MLP
        edge_logits = torch.sigmoid(self.fc(edge_input))
        print(f"Edge logits: {edge_logits}")
        
        return edge_logits