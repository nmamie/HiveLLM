#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict

import torch
import transformers
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
import numpy as np



class NormalDebate(Node):
    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Directly output an answer.",
                 max_token: int = 64,
                 id=None):
        super().__init__(operation_description, id, True)
        self.model_name = model_name
        self.pipeline = self._get_pipeline()
        self.domain = domain
        self.llm = LLMRegistry.get(model_name, self.pipeline)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()
        self.model_id = np.random.randint(0, 1)

        print(f"Creating a normal debate node with role {self.role}")

    @property
    def node_name(self):
        return f"{self.__class__.__name__} {self.role}"
    
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

    async def node_optimize(self, input, meta_optimize=False):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_special_constraint()

        if meta_optimize:
            update_role = role
            node_optimizer = MetaPromptOptimizer(
                self.model_name, self.node_name)
            update_constraint = await node_optimizer.generate(constraint, task)
            return update_role, update_constraint

        return role, constraint

    async def _execute(self, inputs: List[Any] = [], inference: bool = False, **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []

        task: Optional[str] = None
        additional_knowledge: List[str] = []
        for input in node_inputs:
            if len(input) <= 2 and 'task' in input: # Swarm input
                task = input['task']
            else: # All other incoming edges
                extra_knowledge = f"Opinion of {input['operation']} is {input['output']}."
                additional_knowledge.append(extra_knowledge)

        if task is None:
            raise ValueError(f"{self.__class__.__name__} expects swarm input among inputs")

        opinions = ""
        if len(additional_knowledge) > 0:
            for extra_knowledge in additional_knowledge:
                opinions = opinions + extra_knowledge + "\n\n"

        question = self.prompt_set.get_answer_prompt(question=task)
        user_message = question
        if len(opinions) > 0:
            user_message = f"""{user_message}

Take into account the following opinions which may or may not be true and reflect them in your answer:

{opinions}"""

        task = input["task"]
        role, constraint = await self.node_optimize(input, meta_optimize=False)
        prompt = self.prompt_set.get_answer_prompt(question=user_message)    
        message = [Message(role="system", content=f"You are {role}. {constraint}"),
                    Message(role="user", content=prompt)]

        response = await self.llm.agen(message, max_tokens=self.max_token, temperature=0.2, model_id=self.model_id)

        execution = {
            "operation": self.node_name,
            "task": task,
            "files": input.get("files", []),
            "input": task,
            "role": "NormalDebate",
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