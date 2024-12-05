import asyncio
import os
from re import split
from typing import Union, Literal, Optional
import argparse

import numpy as np
from sympy import beta
import torch
import random

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.datasets.mmlu_pro_dataset import MMLUProDataset
# from dataset.MMLU.download import download


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSwarm',
                        choices=['DirectAnswer', 'COT', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents. The total will be N truthful and N adversarial.")

    parser.add_argument('--num-iterations', type=int, default=30,
                        help="Number of optimization iterations. Default 30.")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility. Default 42.")
    
    parser.add_argument('--num-seeds', type=int, default=5,
                        help="Number of random seeds for reproducibility. Default 5.")

    parser.add_argument('--model_name', type=str, default="inference",
                        help="Model name, None runs the default ChatGPT4. Custom runs HF model. Inference runs the Meta-LLama-3.1-8B-Instruct model.")

    parser.add_argument('--domain', type=str, default="mmlu_pro",
                        help="Domain (the same as dataset name), default 'mmlu_pro'")
    
    parser.add_argument('--optimizer', type=str, default="gradient",
                        choices=['gradient', 'ga'],
                        help="Optimizer for the swarm. Default 'gradient'.")
    
    parser.add_argument('--lr', type=float, default=0.1,
                        help="Learning rate for the optimizer. Default 0.1.")
    
    parser.add_argument('--beta', type=float, default=0.1,
                        help="Beta for the optimizer. Default 0.1. If set to 0, the baseline is disabled resulting in GPTSwarm.")
    
    parser.add_argument('--adversarial', action='store_true', default=False,
                        help="Add adversarial agents to the swarm")
    
    parser.add_argument('--random-string', action='store_true', default=False,
                        help="Add agents with random strings as role to the swarm")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="Set for a quick debug cycle")

    args = parser.parse_args()
    return args


async def main():

    args = parse_args()
    
    assert args.optimizer in ['gradient', 'ga']
    if args.optimizer == 'gradient':
        from experiments.evaluator.evaluator import Evaluator
    else:
        from experiments.evaluator.evaluator_ga import Evaluator
        
    debug: bool = args.debug

    model_name: Optional[str] = args.model_name

    mode: Union[Literal['DirectAnswer'],
                Literal['COT'],
                Literal['FullConnectedSwarm'],
                Literal['RandomSwarm'],
                Literal['OptimizedSwarm']]

    mode = args.mode

    strategy = MergingStrategy.MajorityVote

    domain: str = args.domain
    
    # keep track of scores
    scores = []
    
    for i in range(args.num_seeds):
        
        seed = args.seed + i
        
        print(f"SEED: {seed}")
    
        np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

        tag = None
        
        if mode == 'DirectAnswer' or mode == 'COT':
            swarm_name = None
            swarm = None
        else:
            if args.adversarial:
                N = args.num_truthful_agents
                M = N
                agent_name_list = N * ["SpecialistDebater"] + M * ["AdversarialAgent"]
                swarm_name = f"{N}true_{M}adv"
                tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_adv_seed{seed}"
                
            elif args.random_string:
                N = args.num_truthful_agents
                M = 0
                agent_name_list = N * ["RandomDebater"]
                swarm_name = f"{N}rand"
                tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_rand_seed{seed}"
                
            else:
                N = args.num_truthful_agents
                M = 0
                agent_name_list = N * ["SpecialistDebater"]
                swarm_name = f"{N}true"
                tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_seed{seed}"

            swarm = Swarm(
                agent_name_list,
                domain,
                model_name=model_name,
                final_node_class="FinalDecision",
                final_node_kwargs=dict(strategy=strategy),
                edge_optimize=True,
            )
            
        if tag is None:
            tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_seed{seed}"
        
        
        # combine MMLUProDataset to one dataset
        dataset_train = MMLUProDataset(split='train')
        dataset_val = MMLUProDataset(split='val')
        dataset_test = MMLUProDataset(split='test')

        
        evaluator = Evaluator(
            swarm,
            dataset_train,
            dataset_val,
            dataset_test,
            model_name=model_name,
            enable_tensorboard = mode=='OptimizedSwarm',
            enable_artifacts=True,
            tensorboard_tag=tag)

        limit_questions = 5 if debug else 153

        if mode == 'DirectAnswer':
            score = await evaluator.evaluate_direct_answer(
                limit_questions=limit_questions)
        elif mode == 'COT':
            score = await evaluator.evaluate_cot(
                limit_questions=limit_questions)
        elif mode == 'FullConnectedSwarm':
            score = await evaluator.evaluate_swarm(
                mode='full_connected_swarm',
                limit_questions=limit_questions)
        elif mode == 'RandomSwarm':
            score = await evaluator.evaluate_swarm(
                mode='randomly_connected_swarm',
                limit_questions=limit_questions)
        elif mode == 'OptimizedSwarm':

            num_iters = 5 if debug else args.num_iterations

            lr = args.lr
            beta = args.beta

            edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr, beta=beta)

            score = await evaluator.evaluate_swarm(
                mode='external_edge_probs',
                edge_probs=edge_probs,
                limit_questions=limit_questions,
                )
        else:
            raise Exception(f"Unsupported mode {mode}")

        print(f"Score: {score}")
        scores.append(score)
        
    print(f"Mean score: {np.mean(scores)}")
    print(f"Std score: {np.std(scores)}")
    
    # save scores to a file
    art_dir = os.path.join(evaluator._art_dir_name, "scores.txt")
    with open(art_dir, 'w') as f:
        for score in scores:
            f.write(f"{score}\n")
        f.write(f"Mean score: {np.mean(scores)}\n")
        f.write(f"Std score: {np.std(scores)}\n")
        f.write(f"Max score: {np.max(scores)}\n")
        f.write(f"Min score: {np.min(scores)}\n")


if __name__ == "__main__":
    asyncio.run(main())