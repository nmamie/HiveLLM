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
# from optimum.nvidia.pipelines import pipeline
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from accelerate import Accelerator


# init
# model_path = "nvidia/Llama-3.1-Minitron-4B-Width-Base"
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# device = 'cuda'
# dtype = torch.bfloat16
# model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

# models = []
# for i in range(1):
#     models.append(transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         device_map="auto",
#     ))
# models = []
# devices = [0,1,2,3]
# for i in range(10):
# model = AutoModelForCausalLM.from_pretrained(
#         "OuteAI/Lite-Oute-2-Mamba2Attn-Instruct",
#         # To allow custom modeling files
#         trust_remote_code=True,

#         # If you have installed flash attention 2
#         attn_implementation="flash_attention_2",
#         torch_dtype=torch.bfloat16,
#     )
# tokenizer = AutoTokenizer.from_pretrained("OuteAI/Lite-Oute-2-Mamba2Attn-Instruct")
# model = transformers.AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     force_download=True  # This forces the download of the latest model checkpoint
# )

# hf_pipeline2 = transformers.pipeline(
#         "text-generation",
#         model="meta-llama/Meta-Llama-3-8B-Instruct",
#         model_kwargs={
#             "torch_dtype": torch.bfloat16,
#                     },
#         # trust_remote_code=True,
#         device_map="auto",
#     )
                                                                                                        # hf_pipeline3 = transformers.pipeline(
                                                                                                        #         "text-generation",
                                                                                                        #         model="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
                                                                                                        #         # trust_remote_code=True,
                                                                                                        #         device_map="auto",
                                                                                                        #     )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# devices = [0,1,2,3]
# models = {}
# for i in range(10):
#     model = AutoModelForCausalLM.from_pretrained(
#         "OuteAI/Lite-Oute-2-Mamba2Attn-Instruct",
#         # To allow custom modeling files
#         trust_remote_code=True,

#         # If you have installed flash attention 2
#         attn_implementation="flash_attention_2",
#         torch_dtype=torch.bfloat16,
#     )
#     device = random.choice(devices)
#     model.to("cuda:{}".format(device))
#     tokenizer = AutoTokenizer.from_pretrained("OuteAI/Lite-Oute-2-Mamba2Attn-Instruct")
#     models[device] = model
# model = AutoModelForCausalLM.from_pretrained(
#         "OuteAI/Lite-Oute-2-Mamba2Attn-Instruct",
#         # To allow custom modeling files
#         trust_remote_code=True,

#         # If you have installed flash attention 2
#         attn_implementation="flash_attention_2",
#         torch_dtype=torch.bfloat16,
#     )
# device = 3
# model.to("cuda:{}".format(device))
# tokenizer = AutoTokenizer.from_pretrained("OuteAI/Lite-Oute-2-Mamba2Attn-Instruct")
model = None
curr_inf = False

def load_model(inference: bool = False):
    if inference is True:
        # model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    else:
        # model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        # model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # models = []

    hf_pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                        },
            # trust_remote_code=True,
            device_map="auto",
        )
    
    # hf_pipeline2 = transformers.pipeline(
    #         "text-generation",
    #         model=model_id2,
    #         model_kwargs={
    #             "torch_dtype": torch.bfloat16,
    #                     },
    #         # trust_remote_code=True,
    #         device_map="auto",
        # )

    accelerator = Accelerator()
    
    model = accelerator.prepare(hf_pipeline)
    # model2 = accelerator.prepare(hf_pipeline2)
    # models.append(model1)
    # models.append(model2)
    
    print('Model loaded')
    return model

# import pdb; pdb.set_trace()



def hugging_chat(
    model_name,
    messages: List[str],
    max_tokens: int = 30,
    temperature: float = 0.2,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    global model
    if messages[0] == '$skip$':
        return ''

    formatted_messages = [asdict(message) for message in messages]
    
    if model is None:
        model = load_model(inference=False)
    
    if temperature > 0.0:
        do_sample = True
        top_p = 1
        repetition_penalty = 1.12
    else:
        do_sample = False
        top_p = None
        temperature = None
        repetition_penalty = None
                        
    assert model is not None, "Model not loaded"
    
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
    
    if num_comps == 1:
        cost_count(response, model_name)
        return response[0]["generated_text"][-1]['content']

    cost_count(response, model_name)

    return [resp["generated_text"][-1]['content'] for resp in response]
        

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def hugging_achat(
    model_name,
    messages: List[str],
    max_tokens: int = 30,
    temperature: float = 0.2,
    inference: bool = False,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], bool]:
    global model
    global curr_inf
    if messages[0] == '$skip$':
        return ''
    
    formatted_messages = [asdict(message) for message in messages]
        
    len_messages = sum([len(message["content"]) for message in formatted_messages])
        
    # input_ids = tokenizer.apply_chat_template(
    #     formatted_messages, add_generation_prompt=True, return_tensors="pt"
    # ).to(device)
    
    if model is None or curr_inf != inference:
        model = load_model(inference=inference)
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
            
    assert model is not None, "Model not loaded"
    
    # model = random.choice(models)

    try:
        with async_timeout.timeout(20):
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
        inference: Optional[bool] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
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
                                inference,
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