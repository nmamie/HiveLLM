import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear

from torch_geometric.nn import GATConv
from transformers import BertModel, BertTokenizer

from swarm.optimizer.edge_optimizer.graph_net.layers import GraphAttentionLayer

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
    
    
class GATWithSentenceEmbedding(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, sentence_embedding_dim, num_heads=1):
        super(GATWithSentenceEmbedding, self).__init__()
        # Set device
        self.device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
        
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.sentence_embedding_dim = sentence_embedding_dim
        
        self.conv1 = GATConv(sentence_embedding_dim * 2, hidden_channels, heads=num_heads, dropout=0.6, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * num_heads, num_node_features, heads=1, dropout=0.6, add_self_loops=False)
        # self.fc = Linear(768, sentence_embedding_dim)

        # Load pre-trained BERT model and tokenizer
        # Using a distilled BERT model (smaller and faster with fewer parameters)
        self.bert = BertModel.from_pretrained('distilbert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT model parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # self.dropout = Dropout(0.6).to(self.device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sentence: str):
        # Encode the sentence to obtain its embedding
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.device)
        edge_index = edge_index.to(self.device)
        x = x.to(self.device)
        
        print("Inputs:", x)
        
        # # normalize node features
        # x = F.normalize(x, p=2, dim=-1)
        
        with torch.no_grad():
            sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
            sentence_embedding = sentence_embedding.squeeze(0)
            
        print("Node features shape:", x.shape)
        print("Edge index shape:", edge_index.shape)
        print("Sentence embedding shape:", sentence_embedding.shape)
        
        sentence_embedding_repeat = sentence_embedding.repeat(x.shape[0], 1)
        x = torch.cat((x, sentence_embedding_repeat), dim=1)
                
        # Pass through the GAT layers
        x, attention = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        print(f"Attention first conv: {attention}")
        x, attention = self.conv2(x, edge_index.to(self.device), return_attention_weights=True)                
        
        print(f"Attention second conv: {attention}")

        
        print(f"Node embeddings: {x}")
        
        # x = F.normalize(x, p=2, dim=-1)
        
        # sentence_embedding_repeated = sentence_embedding.repeat(x.shape[0], 1)
        
        # x = torch.cat((x, sentence_embedding_repeated), dim=1)
        
        # print(f"Normalized node embeddings: {x}")
        
        return x
    
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        # self.fc = Linear(in_channels * 2, 1)  # Concatenating embeddings of node pairs

    def forward(self, x_i, x_j):
        # Compute the inner product between node embeddings
        edge_probabilities = torch.sum(x_i * x_j, dim=-1)
        edge_probabilities = torch.sigmoid(edge_probabilities)
        print(f"Edge probabilities: {edge_probabilities}")
        return edge_probabilities