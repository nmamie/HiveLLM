from platform import node
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear

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
    
    
class GATWithSentenceEmbedding(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, sentence_embedding_dim, num_heads=1):
        super(GATWithSentenceEmbedding, self).__init__()
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_node_features = num_node_features
        self.hidden_channels = hidden_channels
        self.sentence_embedding_dim = sentence_embedding_dim
        
        self.conv1 = GATConv(sentence_embedding_dim * 2, hidden_channels, heads=num_heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * num_heads, num_node_features, heads=num_heads, dropout=0.6, concat=False)
        self.fc0 = Linear(768, sentence_embedding_dim)
        # self.fc1 = Linear(sentence_embedding_dim * 2, num_node_features)
        self.fc2 = Linear(num_node_features * 2, num_node_features)
        self.fc3 = Linear(num_node_features, 1) # logistic regression
                
        # Load pre-trained BERT model and tokenizer
        # Using a distilled BERT model (smaller and faster with fewer parameters)
        self.bert = BertModel.from_pretrained('distilbert-base-uncased').to(self.device) # type: ignore
        self.tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Freeze BERT model parameters
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # self.dropout = Dropout(0.6).to(self.device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, sentence: str):
        
        edge_index = edge_index.to(self.device)
        x = x.to(self.device)
        
        print("Inputs:", x)
        
        # # normalize node features
        # x = F.normalize(x, p=2, dim=-1)
        
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
            sentence_embedding = sentence_embedding.squeeze(0)
            
        # print("Node features shape:", x.shape)
        # print("Edge index shape:", edge_index.shape)
        # print("Sentence embedding shape:", sentence_embedding.shape)
        
        sentence_embedding = self.fc0(sentence_embedding)
        sentence_embedding_repeat = sentence_embedding.repeat(x.shape[0], 1)
        x = torch.cat([x, sentence_embedding_repeat], dim=1)
        # x = x + sentence_embedding
        # x_reduced = self.fc1(x)
        # x_reconstructed = self.fc2(x_reduced)
        # print("Reduced node features:", x_reduced)
                
        # Pass through the GAT layers
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # print(f"Node embeddings after first conv: {x}")
        # print(f"Attention first conv: {attention}")
        x, attention = self.conv2(x, edge_index.to(self.device), return_attention_weights=True)            
        
        # print(f"Attention second conv: {attention}")

        
        print(f"Node embeddings: {x}")
        
        # average attention over all heads

        print(f"Attention: {attention[1]}")
        
        # Get node embeddings for each edge (source and target nodes)
        x_i, x_j = x[edge_index[0]], x[edge_index[1]]

        # Concatenate the node embeddings of source and target nodes for each edge
        edge_input = torch.cat([x_i, x_j], dim=1)  # This will result in shape (num_edges, 2 * embedding_dim)

        # Pass through neural network layers for edge prediction
        edge_logits = self.fc2(edge_input)  # Assuming fc2 is defined correctly
        edge_logits = nn.ReLU()(edge_logits)

        # Optionally apply Dropout for regularization (good during training)
        edge_logits = nn.Dropout(0.2)(edge_logits)  # Enable this if needed

        # Final layer for binary classification
        edge_logits = self.fc3(edge_logits)  # Assuming fc3 is defined correctly
        print(f"Edge logits (pre-sigmoid): {edge_logits}")

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
        
        return edge_probs
    
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