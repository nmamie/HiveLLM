import os
import asyncio
import pandas as pd
from typing import Iterable, Optional, Iterator, Union, Literal, List, Dict, Any
from tqdm import tqdm
import torch
import torch.nn.functional as F
import time
import datetime
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import json
import math
import copy

from swarm.graph import Graph
from swarm.environment.agents import IO, COT, SpecialistDebater
from swarm.graph.swarm import Swarm
from experiments.evaluator.datasets.base_dataset import BaseDataset
from experiments.evaluator.accuracy import Accuracy
from swarm.optimizer.edge_optimizer.evo_tools.GA import GA

from datasets import Dataset

class Evaluator():
    def __init__(
            self,
            swarm: Optional[Swarm],
            train_dataset: BaseDataset,
            val_dataset: BaseDataset,
            test_dataset: BaseDataset,
            model_name: Optional[str] = None,
            enable_tensorboard: bool = False,
            enable_artifacts: bool = False,
            tensorboard_tag: Optional[str] = None,
        ) -> None:

        self._swarm: Optional[Swarm] = swarm
        self._train_dataset: BaseDataset = train_dataset
        self._val_dataset: BaseDataset = val_dataset
        self._test_dataset: BaseDataset = test_dataset
        self._model_name: Optional[str] = model_name

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}" +
                        (f"_{tensorboard_tag}" if tensorboard_tag is not None else ""))

        if enable_artifacts or enable_tensorboard:
            print(f"Domain: {train_dataset.get_domain()}")
            if train_dataset.get_domain() == 'mmlu':
                self._art_dir_name = os.path.join("runs", art_dir_name)
            else:
                self._art_dir_name = os.path.join("runs", "_pro", art_dir_name)
            os.makedirs(self._art_dir_name, exist_ok=True)
        else:
            self._art_dir_name = None

        if enable_tensorboard:
            self._logger = SummaryWriter(log_dir=self._art_dir_name)
        else:
            self._logger = None
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _infinite_data_loader(self, dataset) -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

    async def evaluate_direct_answer(self,
            limit_questions: Optional[int] = None,
            ) -> float:

        dataset = self._test_dataset
        
        print(
            f"Evaluating DirectAnswer on {dataset.get_domain()} split {dataset.split}")

        single_agent = SpecialistDebater(dataset.get_domain(), self._model_name)

        accuracy = Accuracy()
        
        test_split = self._infinite_data_loader(dataset)
           
        
        for i_question, record in tqdm(enumerate(test_split)):
            print(80*'-')
            if limit_questions is not None:
                if i_question >= limit_questions:
                    break
                
            input_dict = dataset.record_to_swarm_input(record)
            print(input_dict)

            raw_answer = await single_agent.run(input_dict, inference=True)

            print("Raw answer:", raw_answer)
            raw_answer = raw_answer[0]
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            accuracy.update(answer, correct_answer)
            accuracy.print()

        print("Final accuracy:")
        accuracy.print()

        self._dump_eval_results(dict(
            accuracy=accuracy.get(),
            limit_questions=limit_questions))

        print("Done!")
        return accuracy.get()

    async def evaluate_swarm(
            self,
            mode: Union[
                Literal['full_connected_swarm'],
                Literal['randomly_connected_swarm'],
                Literal['external_edge_probs'],
                ],
            edge_probs: Optional[torch.Tensor] = None,
            limit_questions: Optional[int] = None,
            eval_batch_size: int = 4,
            ) -> float:

        assert self._swarm is not None

        dataset = self._test_dataset

        print(f"Evaluating swarm on {dataset.__class__.__name__} split {dataset.split}")

        realized_graph: Optional[Graph]
        if mode == 'full_connected_swarm':
            realized_graph = self._swarm.connection_dist.realize_full(self._swarm.composite_graph)
        elif mode == 'external_edge_probs':
            assert edge_probs is not None
            edge_mask = edge_probs > 0.5
            realized_graph = self._swarm.connection_dist.realize_mask(self._swarm.composite_graph, edge_mask)
            realized_graph.display()
        else:
            realized_graph = None

        accuracy = Accuracy()

        def eval_loader(batch_size: int) -> Iterator[List[Any]]:
            records = []
            for i_record, record in enumerate(dataset):
                if limit_questions is not None:
                    if i_record >= limit_questions:
                        break
                records.append(record)
                if len(records) >= batch_size:
                    yield records
                    records = []
            if len(records) > 0:
                yield records
            return

        data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
        num_batches = int(math.ceil(data_len / eval_batch_size))

        for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
            print(80*'-')

            start_ts = time.time()

            future_answers = []
            for record in record_batch:
                if mode == 'randomly_connected_swarm':
                    realized_graph, _ = self._swarm.connection_dist.realize_particle(self._swarm.composite_graph)
                assert realized_graph is not None

                input_dict = dataset.record_to_swarm_input(record)
                ground_truth = dataset.record_to_target_answer(record)
                print(input_dict)

                # future_answer = self._swarm.arun(input_dict, realized_graph, inference=True)
                future_answer = self._swarm.arun(input_dict, realized_graph, inference=False, ground_truth=ground_truth)
                future_answers.append(future_answer)

            raw_answers = await asyncio.gather(*future_answers)

            print(f"Batch time {time.time() - start_ts:.3f}")

            for raw_answer, record in zip(raw_answers, record_batch):
                raw_answer = raw_answer[0][0]
                print("Raw answer:", raw_answer)
                answer = dataset.postprocess_answer(raw_answer)
                print("Postprocessed answer:", answer)
                correct_answer = dataset.record_to_target_answer(record)
                print("Correct answer:", correct_answer)
                accuracy.update(answer, correct_answer)
                accuracy.print()

        accuracy.print()
        print("Done!")
        
        self._dump_eval_results(dict(
            accuracy=accuracy.get(),
            limit_questions=limit_questions))

        return accuracy.get()

    def _dump_eval_results(self, dct: Dict[str, Any]) -> None:
        if self._art_dir_name is not None:
            eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
            with open(eval_json_name, "w") as f:
                json.dump(dct, f)

    def _print_conns(self, edge_probs: torch.Tensor, save_to_file: bool = False, i_iter: int = 0) -> None:
        assert self._swarm is not None
        msgs = []
        for i_conn, (conn, prob) in enumerate(zip(
                self._swarm.connection_dist.potential_connections, edge_probs)):
            src_id, dst_id = conn
            src_node = self._swarm.composite_graph.find_node(src_id)
            dst_node = self._swarm.composite_graph.find_node(dst_id)
            msg = (f"{i_conn}: src={src_node.node_name}({src_node.id}), "
                    f"dst={dst_node.node_name}({dst_node.id}), prob={prob.item():.3f}")
            msgs.append(msg+"\n")
            print(msg)
        if save_to_file:
            if self._art_dir_name is not None:
                if i_iter == 0:
                    torch.save(self._swarm.connection_dist.state_dict(), os.path.join(self._art_dir_name, "edge_logits_final.pt"))
                else: 
                    torch.save(self._swarm.connection_dist.state_dict(), os.path.join(self._art_dir_name, f"edge_logits_{i_iter}.pt"))

    async def optimize_swarm(
            self,
            num_iters: int,
            lr: float,
            beta: float,
            batch_size: int = 1,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset

        print(f"Optimizing swarm on {dataset.__class__.__name__} split {dataset.split}")
        
        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(lr=lr,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               model_name=self._model_name
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            np.random.seed(42)
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()
        
        # node and edge statistics
        num_pot_edges = len(self._swarm.connection_dist.potential_connections)
        num_nodes = len(self._swarm.composite_graph.nodes)
        
        async def utility_func(particle: List[float]) -> np.array:
                        
            future_answers = []
            correct_answers = []
            fitness = []
            
            # # edge probs
            # edge_probs = particle[:len(self._swarm.connection_dist.edge_logits)]
            
            # GAT params
            # order_params = torch.tensor(particle[:torch.numel(self._swarm.connection_dist.order_params)], device=self.device, dtyptorch.numel(self._swarm.connection_dist.edge_logits)], device=self.device, dtype=torch.floe=torch.float32)
            edge_logits = torch.tensor(particle, device=self.device, dtype=torch.float32)
            # gat_params = particle[torch.numel(self._swarm.connection_dist.order_params) + torch.numel(self._swarm.connection_dist.node_features):]
            
            swarm_copy = copy.deepcopy(self._swarm)
            
            # # update params
            with torch.no_grad():
                # swarm_copy.connection_dist.order_params = torch.nn.Parameter(order_params)
                swarm_copy.connection_dist.edge_logits = torch.nn.Parameter(edge_logits)
                # train_p = [p for p in swarm_copy.connection_dist.gat.parameters() if p.requires_grad]
                # for p, new_p in zip(train_p, gat_params):
                #     p.copy_(new_p)
            
            future_answers = []
            log_probs = []
            correct_answers = []
            for i_record, record in zip(range(batch_size), loader):

                realized_graph, log_prob = swarm_copy.connection_dist.realize_particle(
                    swarm_copy.composite_graph,
                    # temperature=3.0, # DEBUG
                    )

                input_dict = dataset.record_to_swarm_input(record)
                ground_truth = dataset.record_to_target_answer(record)
                answer = swarm_copy.arun(input_dict, realized_graph, inference=False, ground_truth=ground_truth)
                future_answers.append(answer)
                log_probs.append(log_prob)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)

            answers = await asyncio.gather(*future_answers)
            
            # every answer is a tuple of output_answer and intermediate_answers
            raw_output_answers = [answer[0] for answer in answers]
            raw_intermediate_answers = [answer[1] for answer in answers]
            exec_times = [answer[2] for answer in answers]
                   
            utilities = []
            for raw_output_answer, raw_intermediate_answer, correct_answer in zip(raw_output_answers, raw_intermediate_answers, correct_answers):
                output_accuracy = Accuracy()
                intermediate_accuracy = Accuracy()
                output_answer = dataset.postprocess_answer(raw_output_answer)
                intermediate_answers = [dataset.postprocess_answer(intermediate_answer) for intermediate_answer in raw_intermediate_answer]
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                output_accuracy.update(output_answer, correct_answer)
                for intermediate_answer in intermediate_answers:
                    assert isinstance(intermediate_answer, str), \
                        f"String expected but got {intermediate_answer} of type {type(intermediate_answer)} (2)"
                    intermediate_accuracy.update(intermediate_answer, correct_answer)
                utilities.append(0.5*output_accuracy.get() + 0.5*intermediate_accuracy.get())
            fitness.append(np.mean(utilities))
                
            return np.array(fitness), np.array(exec_times)
                
        # init GA
        swarm = self._swarm
        logger = self._logger
        # num_paras = len(self._swarm.connection_dist.edge_logits) + len(self._swarm.connection_dist.order_params)
        # gat_params =  sum(p.numel() for p in self._swarm.connection_dist.gat.parameters() if p.requires_grad)
        num_paras = torch.numel(self._swarm.connection_dist.edge_logits)
        print(f"Number of parameters: {num_paras}")
        ga = GA(func=utility_func, n_dim=num_paras, constraint_eq=None, constraint_ueq=None, lb=0, ub=1,
                size_pop=10, max_iter=num_iters, prob_mut=0.05, n_processes=10,
                utilities=None, swarm=swarm, logger=logger, art_dir_name=self._art_dir_name)
        
        # edge_probs        
        parameters = await ga.run()
        
        # order_params = torch.nn.Parameter(parameters[:num_nodes])
        # create 32-dimensional node features
        edge_logits = torch.nn.Parameter(parameters)
        # self._swarm.connection_dist.edge_logits = edge_probs
        # self._swarm.connection_dist.order_params = order_params
        self._swarm.connection_dist.edge_logits = edge_logits
        # update all trainable GAT parameters
        # with torch.no_grad():
        #     train_p = [p for p in self._swarm.connection_dist.gat.parameters() if p.requires_grad]
        #     # replace every parameter with the respective gat parameter
        #     old_p_sizes = 0
        #     for p in train_p:
        #         p_size = p.numel()
        #         new_p = torch.tensor(gat_params[old_p_sizes:old_p_sizes+p_size]\
        #             .reshape(p.size()), device=self.device, dtype=torch.float32)
        #         new_p = torch.nn.Parameter(new_p, requires_grad=True)
        #         p.copy_(new_p)
        #         old_p_sizes += p_size
        
        print("GA Done!")

        if edge_logits is not None:
            self._print_conns(edge_logits, save_to_file=True)

        print("Done!")
        # edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        return edge_logits