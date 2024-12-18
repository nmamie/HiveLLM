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
from swarm.environment.domain.mmlu.evaluator import Evaluator
from swarm.optimizer.edge_optimizer.evo_ppo.core.params import Parameters
# from dataset.MMLU.download import download


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSwarm',
                        choices=['DirectAnswer', 'COT', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents.")
    
    parser.add_argument('--num-adversarial-agents', type=int, default=1,
                        help="Number of adversarial agents.")

    parser.add_argument('--num-iterations', type=int, default=30,
                        help="Number of optimization iterations. Default 30.")
    
    # parser.add_argument('--seed', type=int, default=42,
    #                     help="Random seed for reproducibility. Default 42.")
    
    # parser.add_argument('--num-seeds', type=int, default=5,
    #                     help="Number of random seeds for reproducibility. Default 5.")

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

    parser.add_argument('--train', action='store_true', default=False,
                        help="training or testing")
    
    #######################  COMMANDLINE - ARGUMENTS ######################
    parser.add_argument('--env', type=str, help='Env Name',  default='MMLU_pro')
    parser.add_argument('--seed', type=int, help='Seed', default=991)
    parser.add_argument('--savetag', type=str, help='#Tag to append to savefile',  default='')
    parser.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)
    parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=0.05)
    parser.add_argument('--buffer', type=float, help='Buffer size in million',  default=0.05)
    # parser.add_argument('--frameskip', type=int, help='Frameskip',  default=1)

    parser.add_argument('--node_feature_size', type=int, help='#Node Feature size',  default=768)
    parser.add_argument('--hidden_size', type=int, help='#Hidden Layer size',  default=64)
    parser.add_argument('--critic_lr', type=float, help='Critic learning rate?', default=3e-4) #3e-4
    parser.add_argument('--actor_lr', type=float, help='Actor learning rate?', default=1e-4) #1e-4
    parser.add_argument('--weight_decay', type=float, help='Weight Decay', default=1e-5)
    parser.add_argument('--tau', type=float, help='Tau', default=1e-3)
    parser.add_argument('--gamma', type=float, help='Discount Rate', default=0.99)
    parser.add_argument('--alpha', type=float, help='Alpha for Entropy term ',  default=0.1)
    parser.add_argument('--batchsize', type=int, help='Batch size',  default=64) #64
    parser.add_argument('--num_envs', type=int, help='Number of environments to average on',  default=4)
    parser.add_argument('--reward_scale', type=float, help='Reward Scaling Multiplier',  default=1.0)
    parser.add_argument('--learning_start', type=int, help='States to wait before learning starts',  default=5000)
    parser.add_argument('--exploration_noise', type=float, help='Exploration Noise',  default=0.0)

    #ALGO SPECIFIC ARGS
    parser.add_argument('--popsize', type=int, help='#Policies in the population',  default=10) #10
    parser.add_argument('--rollsize', type=int, help='#Policies in rollout size',  default=5) #5
    parser.add_argument('--gradperstep', type=float, help='#Gradient step per env step',  default=1.0)
    parser.add_argument('--num_test', type=int, help='#Test envs to average on',  default=5) #5

    # args = parser.parse_args()
    
    return parser


async def main():
    
    args = Parameters(parse_args())
    
    print(f"SEED: {args.seed}")

    #Set seeds
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    debug: bool = args.debug
    
    # assert args.optimizer in ['gradient', 'ga']
    # if args.optimizer == 'gradient':
    #     from experiments.evaluator.evaluator import Evaluator
    # else:
    #     from experiments.evaluator.evaluator_ga import Evaluator
        
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

    tag = None
    
    if mode == 'DirectAnswer':
        swarm_name = None
        swarm = None
    else:
        N = args.num_truthful_agents
        M = args.num_adversarial_agents
        # agent_name_list = N * ["IO"] + M * ["AdversarialAgent"]
        # agent_name_list = ["ModeratorDebater"] +  N * ["SpecialistDebater"] + M * ["AdversarialAgent"]
        agent_name_list = N * ["SpecialistDebater"] + M * ["AdversarialAgent"]
        # agent_name_list = N * ["SpecialistDebater"]

        # swarm_name = f"{N}true_{M}adv"
        swarm_name = f"{N}specialist_{M}adv"
        # swarm_name = f"{N}specialist_debate"

        swarm = Swarm(
            agent_name_list,
            domain,
            model_name=model_name,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy),
            edge_optimize=True,
            node_feature_size = args.node_feature_size,
            hidden_size = args.hidden_size,
        )

    tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}_{args.seed}"
    
    
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
        tensorboard_tag=tag,
        args=args)

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

        score = await evaluator.optimize_swarm(num_iters=num_iters)
    else:
        raise Exception(f"Unsupported mode {mode}")

    print(f"Score: {score}")


if __name__ == "__main__":
    # set spawn method for multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    asyncio.run(main())