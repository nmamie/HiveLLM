#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib.metadata import requires
import pdb
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple
import random

from sklearn.metrics.pairwise import cosine_similarity

from swarm.graph.node import Node
from swarm.graph.graph import Graph
from swarm.graph.composite_graph import CompositeGraph

from swarm.optimizer.edge_optimizer.graph_net.gat import GATWithSentenceEmbedding, LinkPredictor

class ConnectDistribution(nn.Module):
    def __init__(self, potential_connections):
        super().__init__()
        self.potential_connections = potential_connections

    def realize(self, graph):
        raise NotImplemented


class MRFDist(ConnectDistribution):
    pass


class EdgeWiseDistribution(ConnectDistribution):
    def __init__(self,
                 potential_connections,
                 node_feature_size,
                 hidden_size,
                 initial_probability: float = 0.5,
                 ):
        super().__init__(potential_connections)
        # init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        # init_tensor = torch.ones(
        #     len(potential_connections)) * init_logit
        # self.edge_logits = torch.nn.Parameter(init_tensor)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        node_ids = set([x for pair in potential_connections for x in pair])
        self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
        self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
        order_tensor = torch.randn(len(node_ids), requires_grad=False)
        self.order_params = order_tensor
        # self.node_features = torch.randn(len(node_ids), 16, requires_grad=False)
        self.node_features = torch.randn(len(node_ids), node_feature_size, requires_grad=False)
        self.state_indicator = torch.randn(1, node_feature_size, requires_grad=False)
        # self.gat = GATWithSentenceEmbedding(num_node_features=node_feature_size, hidden_channels=hidden_size, sentence_embedding_dim=node_feature_size, num_heads=8).to(self.device)
        # self.link_predictor = LinkPredictor(in_channels=16).to(self.device)
        # edge index
        # edges = self.potential_connections
        # # Unique nodes
        # nodes = list(set([node for edge in edges for node in edge]))
        # # Create a mapping from node labels to indices
        # node_to_index = {node: i for i, node in enumerate(nodes)}
        # # Convert the edges to index format
        # self.edge_index = torch.tensor([[node_to_index[edge[0]], node_to_index[edge[1]]] for edge in edges], dtype=torch.long, requires_grad=False).t()
        self.edge_index = None
                
    def random_sample_num_edges(self, graph: CompositeGraph, num_edges: int) -> CompositeGraph:
        _graph = deepcopy(graph)
        while True:
            if _graph.num_edges >= num_edges:
                break
            potential_connection = random.sample(self.potential_connections, 1)[0]
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    def realize_ranks(self, graph, use_max: bool = False):
        log_probs = []
        ranks = {}
        in_degrees = {node.id: len(node.predecessors) for node in graph.nodes.values()}
        for i in range(len(self.order_params)):
            available_nodes = [node for node in graph.nodes if in_degrees[node] == 0]
            logits = []
            for node in available_nodes:
                logits.append(self.order_params[self.node_id2idx[node]])
            logits = torch.stack(logits).reshape(-1)
            if use_max:
                idx = torch.argmax(logits)
            else:
                idx = torch.distributions.Categorical(logits=logits).sample()
            log_probs.append(torch.log_softmax(logits, dim=0)[idx])

            ranks[available_nodes[idx]] = i
            in_degrees[available_nodes[idx]] = -1
            for successor in graph.nodes[available_nodes[idx]].successors:
                in_degrees[successor.id] -= 1
        return ranks, torch.sum(torch.stack(log_probs))

    def realize(self,
                graph: CompositeGraph,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = 0.0,
                use_learned_order: bool = False,
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        if use_learned_order:
            ranks, log_prob = self.realize_ranks(graph, threshold is not None)
            log_probs = [log_prob]
        else:
            log_probs = [torch.tensor(0.0)]
        _graph = deepcopy(graph)    

        for potential_connection, edge_logit in zip(
                self.potential_connections, self.edge_logits):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue
            
            addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
            addable_if_not_used_learned_order = (not use_learned_order) and (not _graph.check_cycle(in_node, {out_node}, set()))
            if addable_if_not_used_learned_order or addable_if_use_learned_order:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                # edge_prob = edge_logit
                
                if threshold > 0.0:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node)
                    # in_node.add_predecessor(out_node)
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        log_prob = torch.sum(torch.stack(log_probs))
        return _graph, log_prob
    
    def realize_particle(self,
                graph: CompositeGraph,
                edge_logits: torch.Tensor,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = 0.0,
                use_learned_order: bool = False,
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        if use_learned_order:
            ranks, log_prob = self.realize_ranks(graph, threshold is not None)
            log_probs = [log_prob]
        else:
            log_probs = []
        _graph = deepcopy(graph)    

        edge_logits = edge_logits.detach().cpu()
        for potential_connection, edge_logit in zip(
                self.potential_connections, edge_logits):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue
            
            addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
            addable_if_not_used_learned_order = (not use_learned_order) and (not _graph.check_cycle(in_node, {out_node}, set()))
            if addable_if_not_used_learned_order or addable_if_use_learned_order:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                # edge_prob = edge_logit
                
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node)
                    # in_node.add_predecessor(out_node)
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        log_prob = torch.sum(torch.stack(log_probs))
        return _graph, log_prob

    def realize_full(self, graph: CompositeGraph) -> CompositeGraph:
        _graph = deepcopy(graph)
        for i, potential_connection in enumerate(self.potential_connections):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
                
        def is_node_useful(node):
            if node in _graph.output_nodes:
                return True
            
            for successor in node.successors:
                if is_node_useful(successor):
                    return True
            return False
                
        useful_node_ids = [node_id for node_id, node in _graph.nodes.items() if is_node_useful(node)]
        in_degree = {node_id: len(_graph.nodes[node_id].predecessors) for node_id in useful_node_ids}
        # out_degree = {node_id: len(self.nodes[node_id].successors) for node_id in self.useful_node_ids}
        zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0 and node_id in useful_node_ids]
                
        edge_index = []
        while zero_in_degree_queue:
            node_id = zero_in_degree_queue.pop(0)
            node = _graph.nodes[node_id]
            for successor in node.successors:
                edge_index.append([self.node_id2idx[node_id], self.node_id2idx[successor.id]])
                in_degree[successor.id] -= 1
                if in_degree[successor.id] == 0:
                    zero_in_degree_queue.append(successor.id)
                    
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        print(f"Edge index: {self.edge_index}")
        
        
        return _graph

    def realize_mask(self, graph: CompositeGraph, edge_mask: torch.Tensor) -> CompositeGraph:
        _graph = deepcopy(graph)
        for i, (potential_connection, is_edge) in enumerate(zip(self.potential_connections, edge_mask)):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                if is_edge:
                    out_node.add_successor(in_node)
                    in_node.add_predecessor(out_node)
        return _graph
    
    def realize_gat(self, graph: CompositeGraph, record, node_features: torch.Tensor, temperature: float = 1.0, threshold: float = 0.0) -> Tuple[CompositeGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
        _graph = deepcopy(graph)
        # _graph_full = self.realize_full(_graph)
        # input sentence
        query = record['question']
        # adjacency matrix
        # edge_index = _graph_full.get_edge_index()
        # Encode the sentence to obtain its embedding
        
          
        x = torch.tensor(deepcopy(node_features), dtype=torch.float, requires_grad=True)
        print("Node features shape:", x.shape)
        edge_index = deepcopy(self.edge_index)
        # edge_probs = self.link_predictor(x[self.edge_index[0]], x[self.edge_index[1]], init=True)
        # print(f"Edge probs before: {edge_probs}")

        # edge_sim = torch.nn.functional.cosine_similarity(x[orig_edge_index[0]], x[orig_edge_index[1]], dim=1)
        # # Define a threshold to create edges based on similarity
        # edges = []
        # for i, prob in enumerate(edge_sim):
        #     if prob > torch.rand(1).to(self.device):
        #         edges.append(self.edge_index[:, i].tolist())
        #         edges.append(self.edge_index[:, i].tolist()[::-1])

        # # Convert edges to tensor format
        # edge_index = torch.tensor(edges, dtype=torch.long).t()
        # print(f"Edge index: {edge_index}")
        # print(f"ORIGINAL EDGE INDEX: {self.edge_index}")
        
        # if len(edge_index) == 0:
        #     return _graph, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
                
        # with torch.no_grad():
        edge_logits, _, _ = self.gat(x, edge_index, query)
        edge_logits = edge_logits.cpu()
        # edge_probs = self.link_predictor(node_embeddings[self.edge_index[0]], node_embeddings[self.edge_index[1]])
        # # calculate cosine similarity
        # node_embeddings = node_embeddings.detach().cpu()
        # edge_matrix = cosine_similarity(x[self.edge_index[0]], x[self.edge_index[1]])
        # edge_probs = []
        # for i, j in zip(self.edge_index[0], self.edge_index[1]):
        #     if i != j:
        #         edge_probs.append(abs(edge_matrix[i][j]))
        # print(f"Edge probs after: {edge_probs}")
        # edge_probs = torch.tensor(edge_probs)     
        # edge_mask = []
        # for prob in edge_probs:
        #     edge_mask.append(prob > torch.rand(1).to(self.device))
        # edge_mask = torch.stack(edge_mask)
        
        log_probs = [torch.tensor([0.0])]
        # reverse the sigmoid for edge probs, handle the case where edge_probs is 0 or 1
        edge_probs = torch.sigmoid(edge_logits)
        realized_edges = []
        node2idx = {}
        node_id = 0
        print(f"Edge probs after sigmoid: {edge_probs}")
        for i, (potential_connection, edge_logit) in enumerate(zip(self.potential_connections, edge_logits)):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            # if not _graph.check_cycle(in_node, {out_node}, set()):
            edge_prob = torch.sigmoid(edge_logit / temperature)
            # if is_edge:
            #     out_node.add_successor(in_node)
            #     in_node.add_predecessor(out_node)
            if threshold > 0.0:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node)
                realized_edges.append(potential_connection)
                node2idx[out_node.id] = node_id
                node_id += 1
                # in_node.add_predecessor(out_node)
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
        
        if threshold > 0.0:
            log_probs = torch.tensor([0.0])
        else:
            log_probs = torch.sum(torch.stack(log_probs))
        return _graph, edge_probs, log_probs, edge_logits, realized_edges, node2idx