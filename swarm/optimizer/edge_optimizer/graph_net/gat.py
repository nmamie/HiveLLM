from platform import node
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear
from typing import Tuple
import numpy as np

from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import negative_sampling, dropout_adj, structured_negative_sampling

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
    
    
class GATWithSentenceEmbedding(nn.Module):
    def __init__(self, num_nodes, num_potential_edges, num_node_features, hidden_channels, sentence_embedding_dim, num_heads=1):
        super(GATWithSentenceEmbedding, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_nodes = num_nodes
        self.num_potential_edges = num_potential_edges
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.sentence_embedding_dim = sentence_embedding_dim
        
        # Define GAT layers
        self.conv1 = GATConv(sentence_embedding_dim * 2, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, num_node_features, heads=1, dropout=0.6, concat=False)
        
        # Fully Connected Layers
        self.fc0 = Linear(768, sentence_embedding_dim)  # Reduce BERT output to sentence embedding dimension
        self.fc1 = Linear(num_node_features, sentence_embedding_dim)
        self.fc2 = Linear(2 * num_node_features, hidden_channels)
        self.fc3 = Linear(hidden_channels, 1) # Edge scoring
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT except for the last layer
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True  # Unfreeze last transformer layer
        
        self.dropout = Dropout(0.6)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sentence: str) -> torch.Tensor:
        orig_edge_index = edge_index.to(self.device)
        x = x.to(self.device)     
        
        # Efficient edge dropout for training
        if self.training:
            edge_index, _ = dropout_adj(orig_edge_index, p=0.2, force_undirected=False, training=True)
        else:
            edge_index, _ = dropout_adj(orig_edge_index, p=0.2, force_undirected=False, training=True)
            # edge_index = orig_edge_index
        print(f"Edge index: {edge_index}")
        
        # Tokenize and encode sentence using BERT
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            orig_sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]  # CLS token embedding
        sentence_embedding = self.fc0(orig_sentence_embedding).squeeze(0)
        sentence_embedding = F.relu(sentence_embedding)
        
        # Repeat sentence embedding for each node
        sentence_embedding_repeat = sentence_embedding.repeat(x.shape[0], 1)
        
        # Transform node features and combine with sentence embeddings
        x = self.fc1(x)
        x = F.relu(x)
        x = torch.cat([x, sentence_embedding_repeat], dim=1)
        
        # Pass through GAT layers
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x, attention = self.conv2(x, edge_index, return_attention_weights=True)
        
        print(f"Node embeddings: {x}")
        print(f"Attention weights: {attention}")

        if isinstance(attention, tuple):  # Handle tuple output from GATConv
            attention = attention[1]
        
        # if self.training:
        #     # Generate negative edges
        #     neg_edge_index = negative_sampling(edge_index, num_nodes=x.size(0), num_neg_samples=edge_index.size(1))

        #     # Concatenate positive and negative edges
        #     edge_label_index = torch.cat([edge_index, neg_edge_index], dim=1)
        #     # edge_labels = torch.cat([
        #     #     torch.ones(edge_index.size(1)), 
        #     #     torch.zeros(neg_edge_index.size(1))
        #     # ]).to(self.device)
        # else:
        #     edge_label_index = edge_index
        #     # edge_labels = torch.ones(edge_index.size(1)).to(self.device)  # All edges are positive in inference mode
        
        # Compute edge logits
        edge_features = torch.cat([x[orig_edge_index[0]], x[orig_edge_index[1]]], dim=1)
        edge_logits = self.fc2(edge_features).relu().squeeze(-1)
        edge_logits = Dropout(0.6)(edge_logits)
        edge_logits = self.fc3(edge_logits)     
        print(f"Edge logits (pre-sigmoid): {edge_logits}") 
        
        return edge_logits
    
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