#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import ModeratorDebate
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('ModeratorDebater')
class ModeratorDebater(Graph):
    def build_graph(self):
        sd = ModeratorDebate(self.domain, self.model_name)
        self.add_node(sd)
        self.input_nodes = [sd]
        self.output_nodes = [sd]