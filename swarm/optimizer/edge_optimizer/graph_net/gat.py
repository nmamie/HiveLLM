import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from transformers import BertModel, BertTokenizer

from swarm.optimizer.edge_optimizer.graph_net.layers import GraphAttentionLayer

# Set device
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

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
        self.conv1 = GATConv(num_node_features + sentence_embedding_dim, hidden_channels, heads=num_heads).to(device)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads).to(device)
        self.fc = torch.nn.Linear(hidden_channels * num_heads * 2, 1).to(device)  # For edge prediction
        
        # Load pre-trained BERT model and tokenizer
        # Using a distilled BERT model (smaller and faster with fewer parameters)
        self.bert = BertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT model parameters
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sentence: str):
        # Encode the sentence to obtain its embedding
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(device)
        edge_index = edge_index.to(device)
        x = x.to(device)
        
        with torch.no_grad():
            sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
            sentence_embedding = sentence_embedding.squeeze(0).to(device)

        # Concatenate the sentence embedding to each node feature
        sentence_embedding_repeated = sentence_embedding.repeat(x.size(0), 1).to(device)
        x = torch.cat([x, sentence_embedding_repeated], dim=1).to(device)

        # Pass through the GAT layers
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        # Compute edge probabilities
        start, end = edge_index
        edge_features = torch.cat([x[start], x[end]], dim=1).to(device)
        edge_logits = self.fc(edge_features).to(device)
        edge_probs = torch.sigmoid(edge_logits).to(device)

        return edge_probs