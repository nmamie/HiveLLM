from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM

import transformers
import torch

class LLMRegistry:        
        
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None, pipeline = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"
        
        if model_name == 'inference':
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
            
        elif model_name == 'lmstudio':
            model = cls.registry.get('LlamaChat', model_name)
        else: # any version of GPTChat like "gpt-4-1106-preview"
            # model = cls.registry.get('GP
            # TChat', model_name) 
            # init
            model = cls.registry.get('HuggingChat', model_name)

        return model
