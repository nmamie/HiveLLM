#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import RandomDebate
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('RandomDebater')
class RandomDebater(Graph):
    def build_graph(self):
        rd = RandomDebate(self.domain, self.model_name)
        self.add_node(rd)
        self.input_nodes = [rd]
        self.output_nodes = [rd]