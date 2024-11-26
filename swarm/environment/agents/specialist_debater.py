#!/usr/bin/env python
# -*- coding: utf-8 -*-

from swarm.graph import Graph
from swarm.environment.operations import SpecialistDebate
from swarm.environment.agents.agent_registry import AgentRegistry


@AgentRegistry.register('SpecialistDebater')
class SpecialistDebater(Graph):
    def build_graph(self):
        sd = SpecialistDebate(self.domain, self.model_name)
        self.add_node(sd)
        self.input_nodes = [sd]
        self.output_nodes = [sd]