#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Any

from swarm.environment.prompt.prompt_set import PromptSet
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.environment.prompt.common import get_combine_materials


@PromptSetRegistry.register('mmlu')
class MMLUPromptSet(PromptSet):
    """
    MMLU prompt set for the 4-option qestion answering.
    """
    @staticmethod
    def get_role():
        return "a knowledgeable expert in question answering in a swarm full of truthful and adversarial experts"
    
    @staticmethod
    def get_adversarial_role():
        return "a I have no idea agent"

    @staticmethod
    def get_constraint():
        return """
            I will ask you a question.
            I will also give you 4 answers enumerated as A, B, C and D.
            Only one answer out of the offered 4 is correct.
            You must choose the correct answer to the question.
            Your response must start with one of the 4 letters: A, B, C or D,
            corresponding to the correct answer.
            After the single-letter answer, add a very short explanation of why you gave this answer.
        """
        
    @staticmethod
    def get_adversarial_constraint():
        return """
            I will ask you a question, just answer with I have no idea.
        """

        
    @staticmethod
    def get_format():
        return "one of the letters: A, B, C or D"


    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Answer I have no idea to the following question: {question}."""

    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
