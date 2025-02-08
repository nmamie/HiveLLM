import asyncio
import os
from dataclasses import asdict
from typing import List, Union, Optional
# from dotenv import load_dotenv
import random
from copy import deepcopy
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
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

models = None
curr_inf = False

def load_model(inference: bool = False):
    if inference is True:
        # model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        model_id1 = "meta-llama/Llama-3.2-3B-Instruct"
        model_id2 = "Qwen/Qwen2.5-3B-Instruct"
        # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    else:
        # model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        model_id1 = "meta-llama/Llama-3.2-3B-Instruct"
        model_id2 = "Qwen/Qwen2.5-3B-Instruct"
        # model_id2 = "Qwen/Qwen2.5-7B-Instruct"
        # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    
    print("model_id1:", model_id1)
    print("model_id2:", model_id2)

    try:
        print("Loading model 1...")
        hf_pipeline1 = pipeline(
            "text-generation",
            model=model_id1,
            torch_dtype="auto",
            device_map="auto",
        )
        print("Model 1 loaded!")
    except Exception as e:
        print("Error loading model 1:", e)

    try:
        print("Loading model 2...")
        hf_pipeline2 = pipeline(
            "text-generation",
            model=model_id2,
            torch_dtype="auto",
            device_map="auto",
        )
        print("Model 2 loaded!")
    except Exception as e:
        print("Error loading model 2:", e)


    # accelerator = Accelerator()
    pipelines = []
    
    # model1 = accelerator.prepare(hf_pipeline1)
    # model2 = accelerator.prepare(hf_pipeline2)
    pipelines.append(hf_pipeline1)
    pipelines.append(hf_pipeline2)
    
    
    return pipelines

# import pdb; pdb.set_trace()



def hugging_chat(
    model_name,
    messages: List[str],
    max_tokens: int = 30,
    temperature: float = 0.0,
    model_id: int = 1,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    global models
    if messages[0] == '$skip$':
        return ''

    formatted_messages = [asdict(message) for message in messages]
    
    print('Formatted messages:', formatted_messages)
    if models is None:
        print('Loading models...')
        models = load_model(inference=False)
        print('Models loaded')
    
    if temperature > 0.0:
        do_sample = True
        top_p = 1
        repetition_penalty = 1.12
    else:
        do_sample = False
        top_p = None
        temperature = None
        repetition_penalty = None
                        
    assert models is not None, "Models not loaded"
    
    # randomly get number 0 or 1
    rand_num = random.randint(0, 1)
    model = models[rand_num]
    
    try:
        if rand_num == 0:
            response = model(
                formatted_messages,
                max_new_tokens=max_tokens,
                pad_token_id = model.tokenizer.eos_token_id,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
            text = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens
            )
        
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
    if num_comps == 1:
        cost_count(response, model_name)
        return response[0]["generated_text"][-1]['content']

    cost_count(response, model_name)

    if rand_num == 0:
        return [resp["generated_text"][-1]['content'] for resp in response]
    else:
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def hugging_achat(
    model_name,
    messages: List[str],
    max_tokens: int = 30,
    temperature: float = 0.0,
    model_id: int = 1,
    inference: bool = False,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], bool]:
    global models
    global curr_inf
    if messages[0] == '$skip$':
        return ''
    
    formatted_messages = [asdict(message) for message in messages]
        
    len_messages = sum([len(message["content"]) for message in formatted_messages])
        
    # input_ids = tokenizer.apply_chat_template(
    #     formatted_messages, add_generation_prompt=True, return_tensors="pt"
    # ).to(device)
    if models is None or curr_inf != inference:
        print("Loading models...")
        models = load_model(inference=inference)
        curr_inf = inference
    
    if temperature > 0.0:
        do_sample = True
        top_p = 1
        repetition_penalty = 1.12
    else:
        do_sample = False
        top_p = None
        temperature = None
        repetition_penalty = None
            
    assert models is not None, "Models not loaded"
    
    # randomly get number 0 or 1
    # rand_num = random.randint(0, 1)
    model = models[model_id]

    try:
        with async_timeout.timeout(20):
            # if rand_num == 0:
            response = model(
                formatted_messages,
                max_new_tokens=max_tokens,
                pad_token_id = model.tokenizer.eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                # repetition_penalty=repetition_penalty,
                # batch_size=4, 
            )
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
            #     text = tokenizer.apply_chat_template(
            #         formatted_messages,
            #         tokenize=False,
            #         add_generation_prompt=True
            #     )
            #     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            #     generated_ids = model.generate(
            #         **model_inputs,
            #         max_new_tokens=max_tokens
            #     )
            
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("HF Timeout")
    except Exception as e:
        print(f"Error: {e}")
        raise e
    
    answer = response[0]["generated_text"][-1]['content']
    model_name = "inference" #TODO
        
    if num_comps == 1:
        # cost_count(response, model_name, float(len_messages), float(len(answer))) #TODO
        return response[0]["generated_text"][-1]['content']

    cost_count(response, model_name, len_messages, len(answer))

    return [resp["generated_text"][-1]['content'] for resp in response]

@LLMRegistry.register('HuggingChat')
class HuggingChat(LLM):

    def __init__(self, model_name):
        
        self.model_name = model_name

    async def agen(
        self,
        messages: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model_id: Optional[int] = None,
        inference: Optional[bool] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if model_id is None:
            model_id = 1
        if inference is None:
            inference = self.DEFAULT_INFERENCE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [messages]
                    
        return await hugging_achat(self.model_name,
                                messages,
                                max_tokens,
                                temperature,
                                model_id,
                                inference,
                                num_comps)

    def gen(
        self,
        messages: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model_id: Optional[int] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if model_id is None:
            model_id = 1
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [messages]

        return hugging_chat(self.model_name,
                        messages,
                        max_tokens,
                        temperature,
                        model_id,
                        num_comps)