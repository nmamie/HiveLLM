#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from typing import List, Any, Optional
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer

import transformers
import torch


class DirectAnswer(Node): 
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Directly output an answer.",
                 max_token: int = 50, 
                 id=None):
        super().__init__(operation_description, id, True)
        self.model_name: Optional[str] = model_name
        self.pipeline = self._get_pipeline()
        self.domain = domain
        self.llm = LLMRegistry.get(model_name, self.pipeline)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()


    @property
    def node_name(self):
        return self.__class__.__name__
    
    async def node_optimize(self, input, meta_optmize=False):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optmize:
            update_role = role 
            node_optmizer = MetaPromptOptimizer(self.model_name, self.node_name)
            update_constraint = await node_optmizer.generate(constraint, task)
            return update_role, update_constraint

        return role, constraint
    
    def _get_pipeline(self):
        
        if self.model_name != "custom":
            return None
        
        # init
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        # model.to(self.device)
        # model.eval()
        print('Model loaded')
        
        return pipeline


    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []

        for input in node_inputs:
            task = input["task"]
            role, constraint = await self.node_optimize(input, meta_optmize=False)
            prompt = self.prompt_set.get_answer_prompt(question=task)    
            message = [Message(role="system", content=f"You are {role}. {constraint}"),
                       Message(role="user", content=prompt)]

            response = await self.llm.agen(message, max_tokens=self.max_token, temperature=0.2)
                                    
            execution = {
                "operation": self.node_name,
                "task": task,
                "files": input.get("files", []),
                "input": task,
                "role": role,
                "constraint": constraint,
                "prompt": prompt,
                "output": response,
                "ground_truth": input.get("GT", []),
                "format": "natural language"
            }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        # self.log()
        return outputs 