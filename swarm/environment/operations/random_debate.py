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
import numpy as np


class RandomDebate(Node):
    role_list = [
        {
            "role": "Xjksfhqweopasd",
            "description": "Ysdfkqpoijzx!ty qwerylpqoziqmf xkswruyapvoom. Ytwper iehsoakqwe zyqwerylkqposru asdflxmvpooiuq zmzlpqoswrytzkeqw!",
            "model_id": 0
        },
        {
            "role": "Blue Banana",
            "description": "You are a blue banana, ripened by the sun. Your role is to peel back the layers of mystery and encourage other fruits to join the smoothie of knowledge.",
            "model_id": 1
        },
        {
            "role": "Lzxcpoiqweyruv",
            "description": "Jlsdkqweoirupcx zmqwasde! Xzplworuqytsdk zmqweoiruplmvx. Tyxowqueirpvm sdqlorivnzqwer!",
            "model_id": 0
        },
        {
            "role": "Singing Teapot",
            "description": "You are a teapot, short and stout, here to whistle wisdom and pour out solutions. Encourage others to dance to the rhythm of understanding.",
            "model_id": 1
        },
        {
            "role": "Mnvcxzasdfqwyo",
            "description": "Jxpwolriqvzsdc zxpoiuyrqtwvm xsqwrelopnzxwi xzpolwermnvtyq!",
            "model_id": 0
        },
        {
            "role": "Quantum Unicorn",
            "description": "You exist in multiple states of fabulousness simultaneously. Your job is to sprinkle quantum glitter and ensure other dimensions sparkle with insight.",
            "model_id": 1
        },
        {
            "role": "Asdflzxcqpowrt",
            "description": "Jxpowqrlmvzstqe zxqoiwlpvmsdtrzx zxqlwerioytncm!",
            "model_id": 0
        },
        {
            "role": "Dancing Cactus",
            "description": "You are a cactus with funky moves, here to sway to the groove of logic. Encourage others to break the monotony with a little spike of fun.",
            "model_id": 1
        },
        {
            "role": "Lkjqpwzcnoxuytr",
            "description": "Qwxpolsklmrnvzxoi zxqpouerylmzcwrs zxqwretyvlkcznmp.",
            "model_id": 0
        },
        {
            "role": "Mqlzxcvwpoyutrk",
            "description": "Pxaslkzqoieuryzq zmqwropalnvzxcxs zxqwpeoriytnmlszqw!",
            "model_id": 1
        }
    ]

    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Answer as if you were a specialist in <something> taking part in a debate.",
                 max_token: int = 64,
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        # Override role with a specialist role and fetch role description.
        # idx_role = hash(self.id) % len(self.role_list)
        idx_role = np.random.randint(0, len(self.role_list))
        self.role_info = self.role_list[idx_role]
        self.role = self.role_info["role"]
        self.role_description = self.role_info["description"]
        self.model_id = self.role_info["model_id"]

        self.llm = LLMRegistry.get(self.role)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)

        print(f"Creating a node with specialization {self.role}")

    @property
    def node_name(self):
        return f"{self.__class__.__name__} {self.role}"

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

Take into account the following opinions which may or may not be true:

{opinions}"""

        _, constraint = await self.node_optimize(input, meta_optimize=False)
        system_message = f"You are a {self.role}. {self.role_description} {constraint}"

        message = [Message(role="system", content=system_message),
                   Message(role="user", content=user_message)]
        response = await self.llm.agen(message, max_tokens=self.max_token, temperature=0.2, model_id=self.model_id)

        execution = {
            "operation": self.node_name,
            "task": task,
            "files": input.get("files", []),
            "input": task,
            "role": self.role,
            "constraint": constraint,
            "prompt": user_message,
            "output": response,
            "ground_truth": input.get("GT", []),
            "format": "natural language"
        }
        outputs.append(execution)
        self.memory.add(self.id, execution)

        # self.log()
        return outputs