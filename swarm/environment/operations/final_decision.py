#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict, Counter
from enum import Enum
from typing import List, Any, Optional
import random

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry, PromptSet
from swarm.llm import LLMRegistry, LLM
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.operations.operation_registry import OperationRegistry

import transformers
import torch


class MergingStrategy(Enum):
    OutputsAsReferences = 0
    MajorityVote = 1
    RandomChoice = 2
    SelfConsistency = 3
    SelectBest = 5


@OperationRegistry.register("FinalDecision")
class FinalDecision(Node):
    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 strategy: MergingStrategy,
                 operation_description: str = "Refer to all answers and give a final answer.", 
                 id=None):
        super().__init__(operation_description, id, True)
        self.strategy: MergingStrategy = strategy
        self.model_name: Optional[str] = model_name
        self.pipeline = self._get_pipeline()
        self.domain: str = domain
        self.llm: LLM = LLMRegistry.get(model_name, self.pipeline)
        self.prompt_set: PromptSet = PromptSetRegistry.get(domain)
        self.role: str = self.prompt_set.get_role()
        self.role_description: str = "Refer to all answers and give a final answer."
        self.constraint: str = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        return self.__class__.__name__

    def meta_prompt(self, node_inputs, meta_init=False):

        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        self.materials = defaultdict(str)

        for input in node_inputs:
            operation = input.get('operation')
            if operation != "FileAnalyse":
                self.materials[operation] += f'{input.get("output", "")}\n'
            else:
                self.materials["files"] = input.get("files") 
            self.materials["task"] = input.get('task') 

        question = self.prompt_set.get_combine_materials(self.materials)
        prompt = self.prompt_set.get_answer_prompt(question=question)    

        if meta_init:
            pass #TODO

        return role, constraint, prompt
    
    def _get_pipeline(self):
        
        if self.model_name != "custom" or self.strategy != MergingStrategy.OutputsAsReferences:
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

    async def _execute(self, inputs: List[Any] = [], agent_opinions: List[Any] = None,
                       **kwargs) -> None:

        node_inputs = self.process_input(inputs)
        prompt = None
        response = None

        if self.strategy == MergingStrategy.OutputsAsReferences:

            role, constraint, prompt = self.meta_prompt(node_inputs)
            message = [Message(role="system", content=f"You are a {role}. {constraint}"),
                    Message(role="user", content=prompt)]
        
            response = await self.llm.agen(message)

        elif self.strategy == MergingStrategy.MajorityVote:
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for MajorityVote")
            answers = [input.get("output") for input in inputs]
            choices = [ans[0] for ans in answers]
            counter = Counter(choices)
            sorted_counter = counter.most_common()
            # print(f"SORTED COUNTER: {sorted_counter}")
            # print(f"Agent opinions: {agent_opinions}")
            max_freq = sorted_counter[0][1]
            confidence = max_freq / len(answers)
            equally_frequent_answers = [ans for ans, freq in sorted_counter if freq == max_freq]
            response = ""
            for ans in answers:
                if ans[0] in equally_frequent_answers:
                    response = ans
                    break
            if response == "":
                response = random.choice(equally_frequent_answers)
            # print(f"{answers=} {response=}")
            
        elif self.strategy == MergingStrategy.RandomChoice:
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for RandomChoice")
            answers = [input.get("output") for input in inputs]
            response = random.choice(answers)
            # print(f"{answers=} {response=}")

        elif self.strategy == MergingStrategy.SelfConsistency:  
            # This is different from MajorityVote because it is prompt-based.
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for MajorityVote")
            
            question = inputs[0]["task"]
            answers = [input.get("output") for input in inputs]
            constraint = self.prompt_set.get_constraint()
            prompt = self.prompt_set.get_self_consistency(question=question, answers=answers, constraint=constraint)
            message = [Message(role="system", content=f"You are a {self.role}. {self.constraint}"),
                    Message(role="user", content=prompt)]
            response = await self.llm.agen(message)
            # print(f"{answers=} {response=}")

        elif self.strategy == MergingStrategy.SelectBest:  
            # This is different from MajorityVote because it is prompt-based.
            if len(inputs) == 0:
                raise Exception("No inputs is not supported for MajorityVote")
            
            question = inputs[0]["task"]
            answers = [input.get("output") for input in inputs]
            constraint = self.prompt_set.get_constraint()
            prompt = self.prompt_set.get_select_best(question=question, answers=answers, constraint=constraint)
            message = [Message(role="system", content=f"You are a {self.role}. {self.constraint}"),
                    Message(role="user", content=prompt)]
            response = await self.llm.agen(message)
            # print(f"{answers=} {response=}")


        else:
            logger.error(f"Error: does not support \"{self.strategy}\"!")

        executions = {"operation": self.node_name,
                        "task": inputs[0]["task"], 
                        "files": inputs[0]["files"],
                        "input": inputs, 
                        "subtask": prompt,
                        "output": response,
                        "confidence": confidence if self.strategy == MergingStrategy.MajorityVote else None,
                        "format": "natural language"}

        self.memory.add(self.id, executions)
        self.log()
        return executions
        
        
