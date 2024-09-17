import asyncio
from typing import Union, Literal, Optional
import argparse

from swarm.graph.swarm import Swarm
from swarm.environment.operations.final_decision import MergingStrategy
from experiments.evaluator.evaluator_ga import Evaluator
from experiments.evaluator.datasets.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download


def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='OptimizedSwarm',
                        choices=['DirectAnswer', 'FullConnectedSwarm', 'RandomSwarm', 'OptimizedSwarm'],
                        help="Mode of operation. Default is 'OptimizedSwarm'.")

    parser.add_argument('--num-truthful-agents', type=int, default=1,
                        help="Number of truthful agents. The total will be N truthful and N adversarial.")

    parser.add_argument('--num-iterations', type=int, default=30,
                        help="Number of optimization iterations. Default 30.")

    parser.add_argument('--model_name', type=str, default="inference",
                        help="Model name, None runs the default ChatGPT4. Custom runs HF model. Inference runs the Meta-LLama-3.1-8B-Instruct model.")

    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")

    parser.add_argument('--debug', action='store_true', default=False,
                        help="Set for a quick debug cycle")

    args = parser.parse_args()
    return args


async def main():

    args = parse_args()

    debug: bool = args.debug

    model_name: Optional[str] = args.model_name

    mode: Union[Literal['DirectAnswer'],
                Literal['FullConnectedSwarm'],
                Literal['RandomSwarm'],
                Literal['OptimizedSwarm']]

    mode = args.mode

    strategy = MergingStrategy.MajorityVote

    domain: str = args.domain

    if mode == 'DirectAnswer':
        swarm_name = None
        swarm = None
    else:
        N = args.num_truthful_agents
        M = N
        # agent_name_list = N * ["SpecialistAgent"] + M * ["AdversarialAgent"]
        agent_name_list = N * ["SpecialistAgent"]

        # swarm_name = f"{N}true_{M}adv"
        swarm_name = f"{N}specialist"
        

        swarm = Swarm(
            agent_name_list,
            domain,
            model_name=model_name,
            final_node_class="FinalDecision",
            final_node_kwargs=dict(strategy=strategy),
            edge_optimize=True,
        )

    tag = f"{domain}_{swarm_name}_{strategy.name}_{mode}"

    download()

    dataset_train = MMLUDataset('dev')
    dataset_val = MMLUDataset('val')
    
    # # build pytorch dataset for GPU efficiency
    # class Dataset(torch.utils.data.Dataset):
    #     def __init__(self, dataset):
    #         self.dataset = dataset
    #     def __len__(self):
    #         return len(self.dataset)
    #     def __getitem__(self, idx):
    #         return self.dataset[idx]

    evaluator = Evaluator(
        swarm,
        dataset_train,
        dataset_val,
        model_name=model_name,
        enable_tensorboard = mode=='OptimizedSwarm',
        enable_artifacts=True,
        tensorboard_tag=tag)

    limit_questions = 5 if debug else 153

    if mode == 'DirectAnswer':
        score = await evaluator.evaluate_direct_answer(
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

        lr = 0.1

        edge_probs = await evaluator.optimize_swarm(num_iters=num_iters, lr=lr)

        score = await evaluator.evaluate_swarm(
            mode='external_edge_probs',
            edge_probs=edge_probs,
            limit_questions=limit_questions,
            )
    else:
        raise Exception(f"Unsupported mode {mode}")

    print(f"Score: {score}")


if __name__ == "__main__":
    asyncio.run(main())
