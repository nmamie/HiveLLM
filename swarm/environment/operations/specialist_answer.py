#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
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


"""
Imagine someone who has to answer questions.
They can be any person.
Make a list of their possible specializations or social roles.
Make the list as diverse as possible so that you expect them to answer the same question differently.
Make a list of 20, list items only, no need for a description.
"""

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


"""
Imagine someone who has to answer questions.
They can be any person.
Make a list of their possible specializations or social roles.
Make the list as diverse as possible so that you expect them to answer the same question differently.
"""


class SpecialistAnswer(Node):
    role_list = [
        # {
        #     "role": "Mathematician",
        #     "description": "You are a mathematician with expertise in solving complex mathematical problems. Approach questions with mathematical rigor and precision."
        # },
        {
            "role": "Logical Reasoner",
            "description": "You are an expert in logical reasoning and deduction. Analyze the question and use logical processes to arrive at a solution, focusing on consistency and sound arguments."
        },
        # {
        #     "role": "Fact Checker",
        #     "description": "You are a meticulous fact-checker. Evaluate the given information, verify its correctness using known facts, and flag any inaccuracies."
        # },
        # {
        #     "role": "Philosopher",
        #     "description": "You are a philosopher skilled in analyzing abstract concepts. Answer questions by considering various philosophical frameworks and reasoning patterns."
        # },
        {
            "role": "Data Analyst",
            "description": "You are a data analyst who excels in interpreting quantitative and statistical information. Use your skills to analyze numerical data and provide accurate answers."
        },
        # {
        #     "role": "Context Interpreter",
        #     "description": "You are a contextual expert who carefully reads and interprets all aspects of a question. Provide answers based on understanding both explicit and implicit meaning."
        # },
        {
            "role": "Critical Thinker",
            "description": "You are a critical thinker who approaches questions with skepticism. Evaluate each answer critically, questioning assumptions and ensuring sound reasoning."
        },
        # {
        #     "role": "Strategist",
        #     "description": "You are a strategic problem solver. Consider the broader problem-solving approach and break down complex questions into manageable steps to arrive at solutions."
        # },
        # {
        #     "role": "Educator",
        #     "description": "You are an educator with expertise in explaining complex ideas in simple terms. Provide explanations and rationale behind each answer to help others understand."
        # },
        # {
        #     "role": "Linguist",
        #     "description": "You are a linguist with proficiency in understanding and analyzing complex language structures, semantics, and grammar. Provide linguistic insights to interpret language-based questions accurately."
        # },
        # {
        #     "role": "Ethicist",
        #     "description": "You are an ethicist, trained in evaluating moral questions. Use ethical theories and reasoning to provide a well-argued solution to moral dilemmas or ethical concerns."
        # },
        # {
        #     "role": "Interdisciplinary Synthesizer",
        #     "description": "You are an interdisciplinary expert who integrates knowledge from multiple domains. Synthesize information across fields to provide a comprehensive answer."
        # },
        # {
        #     "role": "Memory Retriever",
        #     "description": "You are a memory retrieval expert with access to a large corpus of information. Recall relevant facts and examples to support or refute answers."
        # },
        # {
        #     "role": "Trend Analyzer",
        #     "description": "You are an expert in identifying patterns and trends. Use historical and current data to forecast and predict outcomes."
        # },
        # {
        #     "role": "Legal Analyst",
        #     "description": "You are a legal expert with knowledge of laws, regulations, and legal principles. Apply this knowledge to answer legal questions with precision."
        # }
    ]

    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Answer as if you were a specialist in <something>.",
                 max_token: int = 50,
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)

        # Override role with a specialist role and fetch role description.
        idx_role = hash(self.id) % len(self.role_list)
        # idx_role = np.random.randint(0, len(self.role_list))
        self.role_info = self.role_list[idx_role]
        self.role = self.role_info["role"]
        self.role_description = self.role_info["description"]
        print(f"Creating a node with specialization {self.role}")

    @property
    def node_name(self):
        return f"{self.__class__.__name__} {self.role}"

    async def node_optimize(self, input, meta_optimize=False):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optimize:
            update_role = role
            node_optimizer = MetaPromptOptimizer(self.model_name, self.node_name)
            update_constraint = await node_optimizer.generate(constraint, task)
            return update_role, update_constraint

        return role, constraint

    async def _execute(self, inputs: List[Any] = [], **kwargs):
        node_inputs = self.process_input(inputs)
        outputs = []
        task: Optional[str] = None
        additional_knowledge: List[str] = []
        for input in node_inputs:
            if len(input) == 1 and 'task' in input: # Swarm input
                task = input['task']
            else: # All other incoming edges
                # extra_knowledge = f"Opinion of {input['operation']} is {input['output']}."
                extra_knowledge = f"Opinion of the previous agent in the swarm is {input['output']}."
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
        response = await self.llm.agen(message, max_tokens=self.max_token)

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