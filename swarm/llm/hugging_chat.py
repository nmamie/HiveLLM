import asyncio
import os
from dataclasses import asdict
from typing import List, Union, Optional
# from dotenv import load_dotenv
import random
import async_timeout
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time
from typing import Dict, Any

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.price import cost_count
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry

import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from accelerate import Accelerator


# init
# Load the tokenizer and model
# model_path = "nvidia/Llama-3.1-Minitron-4B-Width-Base"
# device = 'auto'
# dtype = torch.bfloat16
# model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# models = []
# for i in range(1):
#     models.append(transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         device_map="auto",
#     ))

hf_pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16,
                    },
        device_map="auto",
    )
accelerator = Accelerator()
model = accelerator.prepare(hf_pipeline.model)

print('Model loaded')

# import pdb; pdb.set_trace()



def hugging_chat(
    model,
    messages: List[str],
    max_tokens: int = 32,
    temperature: float = 0.2,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0] == '$skip$':
        return ''

    formatted_messages = [asdict(message) for message in messages]
    
    if model is None:
        model = hf_pipeline
        if temperature > 0.0:
            do_sample = True
            top_p = 0.9
        else:
            top_p = None
            do_sample = False
            temperature = None
    
    try:
        response = model(
            formatted_messages,
            max_new_tokens=max_tokens,
            pad_token_id = model.tokenizer.eos_token_id,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
        )
        
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
    answer = response[0]["generated_text"][-1]['content']
    
    return answer
        

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def hugging_achat(
    model,
    messages: List[str],
    max_tokens: int = 32,
    temperature: float = 0.2,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0] == '$skip$':
        return ''
    
    formatted_messages = [asdict(message) for message in messages]   
    
    if model is None:
        model = hf_pipeline
        if temperature > 0.0:
            do_sample = True
            top_p = 0.9
        else:
            top_p = None
            do_sample = False
            temperature = None

    try:
        with async_timeout.timeout(60):
            response = model(
                formatted_messages,
                max_new_tokens=max_tokens,
                pad_token_id = model.tokenizer.eos_token_id,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                # batch_size=4, 
            )
            
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("HF Timeout")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
    answer = response[0]["generated_text"][-1]['content']
    
    return answer
    

@LLMRegistry.register('HuggingChat')
class HuggingChat(LLM):

    def __init__(self, model_name):
        
        self.model_name = model_name

    async def agen(
        self,
        messages: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [messages]
                    
        return await hugging_achat(self.model_name,
                                messages,
                                max_tokens,
                                temperature,
                                num_comps)

    def gen(
        self,
        messages: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [messages]

        return hugging_chat(self.model_name,
                        messages,
                        max_tokens,
                        temperature,
                        num_comps)