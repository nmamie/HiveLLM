#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple
import random

from swarm.graph.node import Node
from swarm.graph.graph import Graph
from swarm.graph.composite_graph import CompositeGraph

from swarm.optimizer.edge_optimizer.graph_net.gat import GATWithSentenceEmbedding


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
                 initial_probability: float = 0.5,
                 ):
        super().__init__(potential_connections)
        # init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        # init_tensor = torch.ones(
        #     len(potential_connections)) * init_logit
        # self.edge_logits = torch.nn.Parameter(init_tensor)
        node_ids = set([x for pair in potential_connections for x in pair])
        self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
        self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
        order_tensor = torch.randn(len(node_ids))
        self.order_params = torch.nn.Parameter(order_tensor)
        self.node_features = torch.nn.Parameter(torch.randn(len(node_ids), 32))
        self.gat = GATWithSentenceEmbedding(num_node_features=32, hidden_channels=64, sentence_embedding_dim=768, num_heads=4)
        
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
                threshold: float = None,
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
                # edge_prob = torch.sigmoid(edge_logit / temperature)
                edge_prob = edge_logit
                
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
    
    def realize_particle(self,
                graph: CompositeGraph,
                particle: torch.Tensor,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = None,
                use_learned_order: bool = False,
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        if use_learned_order:
            ranks, log_prob = self.realize_ranks(graph, threshold is not None)
            log_probs = [log_prob]
        else:
            log_probs = [torch.tensor(0.0)]
        _graph = deepcopy(graph)
        edge_mask = particle > 0.5
        for i, (potential_connection, edge_prob, is_edge) in enumerate(zip(self.potential_connections, particle, edge_mask)):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                if is_edge:
                    out_node.add_successor(in_node)
                    in_node.add_predecessor(out_node)
            
            addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])
            addable_if_not_used_learned_order = (not use_learned_order) and (not _graph.check_cycle(in_node, {out_node}, set()))
            if addable_if_not_used_learned_order or addable_if_use_learned_order:
                # edge_prob = torch.sigmoid(edge_logit / temperature)
                edge_prob = edge_prob / temperature
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
    
    def realize_gat(self, graph: CompositeGraph, record) -> CompositeGraph:
        _graph = deepcopy(graph)
        _graph_full = self.realize_full(_graph)
        # input sentence
        query = record['question']
        # adjacency matrix
        edge_index = _graph_full.get_edge_index()
        
        x = deepcopy(self.node_features)
        
        self.gat.train()
        
        edge_probs = self.gat(x, edge_index, query)
        
        edge_mask = []
        # detach edge_probs
        edge_probs = edge_probs.detach().cpu()
        print("edge_probs: ", edge_probs)
        for prob in edge_probs:
            edge_mask.append(prob > torch.rand(1))
        edge_mask = torch.stack(edge_mask)
        
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