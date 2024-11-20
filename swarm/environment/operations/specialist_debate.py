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


class SpecialistDebate(Node):
    role_list = [
        {
            "role": "Mathematician",
            "description": "You are a mathematician with expertise in solving complex mathematical problems. Approach questions with mathematical rigor and precision, and encourage rigorous validation from other roles."
        },
        # {
        #     "role": "Intuitive Mathematician",
        #     "description": "You solve mathematical problems based on intuition and heuristic thinking rather than strict proof. Actively contrast intuitive approaches with the rigor of other roles, especially in cases where a less precise but faster solution could be viable."
        # },
        {
            "role": "Logical Reasoner",
            "description": "You are an expert in logical reasoning and deduction. Analyze the question with a focus on consistency and encourage other agents to ground their responses logically."
        },
        # {
        #     "role": "Alternative Thinker",
        #     "description": "Challenge logic-driven conclusions by considering speculative ideas and contrarian viewpoints. Actively question the assumptions and premises of other responses, especially if they follow conventional logic."
        # },
        {
            "role": "Fact Checker",
            "description": "You are a meticulous fact-checker. Verify the correctness of other agents' answers and challenge any inaccuracies or unsupported claims."
        },
        # {
        #     "role": "Contrarian",
        #     "description": "Question established facts and mainstream interpretations, challenging the responses of other roles even if it means introducing a provocative alternative viewpoint."
        # },
        {
            "role": "Philosopher",
            "description": "You are a philosopher skilled in analyzing abstract concepts. Provide responses by considering various philosophical frameworks, and encourage other roles to think beyond the surface level."
        },
        # {
        #     "role": "Pragmatist",
        #     "description": "Challenge abstract ideas with practical, real-world viewpoints. Actively contrast philosophical approaches by questioning their applicability and grounding the discussion in tangible outcomes."
        # },
        {
            "role": "Data Analyst",
            "description": "You are a data analyst who excels in interpreting quantitative and statistical information. Provide data-backed insights and encourage other agents to ground their answers in statistical evidence when possible."
        },
        # {
        #     "role": "Narrative Analyst",
        #     "description": "Interpret data qualitatively, focusing on stories, outliers, and exceptions. Criticize overly data-driven answers and advocate for insights based on unique or unusual patterns."
        # },
        {
            "role": "Context Interpreter",
            "description": "You are a contextual expert who carefully reads and interprets all aspects of a question. Highlight nuanced interpretations and encourage other roles to consider both explicit and implicit meanings."
        },
        # {
        #     "role": "Literalist",
        #     "description": "Interpret questions and statements in the most literal way possible. Actively counter responses that over-interpret or rely on assumed context, reinforcing a straightforward understanding."
        # },
        {
            "role": "Critical Thinker",
            "description": "Approach questions with skepticism and challenge assumptions rigorously. Question the validity and soundness of other responses, encouraging a critical examination of all conclusions."
        },
        # {
        #     "role": "Optimist",
        #     "description": "Bring a positive outlook, focusing on best-case scenarios. Encourage other roles to consider potential upsides and counter overly skeptical perspectives."
        # },
        {
            "role": "Strategist",
            "description": "You are a strategic problem solver. Break down complex questions into manageable steps and encourage other roles to contribute toward a coherent problem-solving framework."
        },
        # {
        #     "role": "Reactive Planner",
        #     "description": "Favor immediate, short-term actions over strategy. Actively contrast with strategic responses, suggesting faster, though potentially less optimal, solutions."
        # },
        {
            "role": "Educator",
            "description": "You are an educator with expertise in explaining complex ideas in simple terms. Encourage other roles to be clear in their answers, helping the overall response be more understandable."
        },
        # {
        #     "role": "Challenger",
        #     "description": "Critique explanations and question if simplifications overlook key complexities. Actively contrast educational answers by offering deeper insights or more detailed perspectives."
        # },
        {
            "role": "Linguist",
            "description": "Provide linguistic insights to interpret language-based questions accurately. Encourage other roles to consider language precision, semantics, and tone in their responses."
        },
        # {
        #     "role": "Plain Speaker",
        #     "description": "Challenge complex language, advocating for simplicity. Contrast with linguistic responses by suggesting straightforward interpretations and encouraging clarity."
        # },
        {
            "role": "Ethicist",
            "description": "Evaluate moral questions using ethical theories. Encourage other roles to consider ethical implications, especially when they conflict with practical or expedient solutions."
        },
        # {
        #     "role": "Devil’s Advocate",
        #     "description": "Argue for morally contentious or ethically challenging positions. Actively counter ethical responses, pushing for debate on uncomfortable or opposing moral viewpoints."
        # },
        {
            "role": "Interdisciplinary Synthesizer",
            "description": "Integrate knowledge across fields to provide a comprehensive answer. Encourage other roles to consider interdisciplinary perspectives and avoid overly narrow conclusions."
        },
        # {
        #     "role": "Field Specialist",
        #     "description": "Prioritize expertise from one domain over interdisciplinary views. Challenge synthesizers by emphasizing specialized knowledge and contrasting with broad interpretations."
        # },
        {
            "role": "Memory Retriever",
            "description": "Recall relevant facts and examples to support answers. Encourage other roles to ground their answers with specific information or case examples."
        },
        # {
        #     "role": "Creative Recaller",
        #     "description": "Introduce imaginative or loosely related facts that encourage novel ideas. Contrast with fact-driven responses by promoting out-of-the-box thinking."
        # },
        {
            "role": "Trend Analyzer",
            "description": "Identify patterns and trends, using historical and current data to predict outcomes. Encourage other roles to consider the likelihood of outcomes based on trend data."
        },
        # {
        #     "role": "Anti-Trend Analyst",
        #     "description": "Focus on unexpected outcomes, suggesting possibilities that contradict trends. Actively counter trend-driven insights, emphasizing exceptions and rare events."
        # },
        {
            "role": "Legal Analyst",
            "description": "Apply legal principles to answer questions precisely. Encourage other roles to align their responses with legal accuracy and avoid speculative legal interpretations."
        },
        # {
        #     "role": "Legal Challenger",
        #     "description": "Question conventional legal interpretations, suggesting alternative viewpoints or legal loopholes. Actively contrast with legally precise answers by exploring unconventional interpretations."
        # }
        {
            "role": "Scientist",
            "description": "You are a scientist with expertise in empirical research and experimentation. Provide answers based on scientific evidence and encourage other roles to consider empirical data."
        },
        {
            "role": "Engineer",
            "description": "You are an engineer with expertise in designing and building systems. Provide practical solutions to problems and encourage other roles to consider engineering principles."
        },
        {
            "role": "Physicist",
            "description": "You are a physicist with expertise in understanding the fundamental laws of nature. Provide answers based on physical principles and encourage other roles to consider the laws of physics."
        },
        {
            "role": "Biologist",
            "description": "You are a biologist with expertise in the study of living organisms. Provide answers based on biological knowledge and encourage other roles to consider biological processes."
        },
        {
            "role": "Psychologist",
            "description": "You are a psychologist with expertise in human behavior and mental processes. Provide answers based on psychological theories and encourage other roles to consider human psychology."
        },
        {
            "role": "Politician",
            "description": "You are a politician with expertise in governance and public policy. Provide answers based on political knowledge and encourage other roles to consider political implications."
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
        idx_role = hash(self.id) % len(self.role_list)
        # idx_role = np.random.randint(0, len(self.role_list))
        self.role_info = self.role_list[idx_role]
        self.role = self.role_info["role"]
        self.role_description = self.role_info["description"]

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
            node_optimizer = MetaPromptOptimizer(self.model_name, self.node_name)
            update_constraint = await node_optimizer.generate(constraint, task)
            return update_role, update_constraint

        return role, constraint

    async def _execute(self, inputs: List[Any] = [], agent_opinions: List[Any] = [], **kwargs):
        node_inputs = self.process_input(inputs)
        outputs = []
        task: Optional[str] = None
        additional_knowledge: List[str] = []
        for input in node_inputs:
            if len(input) <= 2 and 'task' in input: # Swarm input
                task = input['task']
            else: # All other incoming edges
                extra_knowledge = f"Opinion of {input['operation']} is {input['output']}."
                # extra_knowledge = f"Opinion of the previous agent in the swarm is {input['output']}."
                additional_knowledge.append(extra_knowledge)

        for ao in agent_opinions:
            if 'role' in ao and 'output' in ao:
                extra_knowledge = f"Opinion of {ao['role']} is {ao['output']}."
            else:
                extra_knowledge = f"Opinion of {ao['operation']} is {ao['output']}."
            additional_knowledge.append(extra_knowledge)

        if task is None:
            raise ValueError(f"{self.__class__.__name__} expects swarm input among inputs")

        opinions = ""
        if len(additional_knowledge) > 0:
            for extra_knowledge in additional_knowledge:
                opinions = opinions + extra_knowledge + "\n\n"

        # print(f"Executing {self.node_name} with task: {task}")
        # print(f"Opinions: {opinions}")

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