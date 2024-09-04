import os
import asyncio
import pandas as pd
from typing import Iterable, Optional, Iterator, Union, Literal, List, Dict, Any
from tqdm import tqdm
import torch
import time
import datetime
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import json
import math
import copy

from swarm.graph import Graph
from swarm.environment.agents import IO
from swarm.graph.swarm import Swarm
from experiments.evaluator.datasets.base_dataset import BaseDataset
from experiments.evaluator.accuracy import Accuracy
from swarm.optimizer.edge_optimizer.evo_tools.GARL import GARL

from datasets import Dataset

class Evaluator():
    def __init__(
            self,
            swarm: Optional[Swarm],
            train_dataset: BaseDataset,
            val_dataset: BaseDataset,
            model_name: Optional[str] = None,
            enable_tensorboard: bool = False,
            enable_artifacts: bool = False,
            tensorboard_tag: Optional[str] = None,
        ) -> None:

        self._swarm: Optional[Swarm] = swarm
        self._train_dataset: BaseDataset = train_dataset
        self._val_dataset: BaseDataset = val_dataset
        self._model_name: Optional[str] = model_name

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}" +
                        (f"_{tensorboard_tag}" if tensorboard_tag is not None else ""))

        if enable_artifacts or enable_tensorboard:
            self._art_dir_name = os.path.join("runs", art_dir_name)
            os.makedirs(self._art_dir_name, exist_ok=True)
        else:
            self._art_dir_name = None

        if enable_tensorboard:
            self._logger = SummaryWriter(log_dir=self._art_dir_name)
        else:
            self._logger = None
            
        self.utilities = []
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    async def evaluate_direct_answer(self,
            limit_questions: Optional[int] = None,
            ) -> float:

        dataset = self._val_dataset
        
        print(f"Evaluating DirectAnswer on {dataset.get_domain()} split {dataset.split}")

        io_agent = IO(dataset.get_domain(), self._model_name)

        accuracy = Accuracy()

        for i_question, record in tqdm(enumerate(dataset)):
            print(80*'-')
            if limit_questions is not None:
                if i_question >= limit_questions:
                    break

            input_dict = dataset.record_to_swarm_input(record)
            print(input_dict)

            raw_answer = await io_agent.run(input_dict)

            print("Raw answer:", raw_answer)
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

        dataset = self._val_dataset

        print(f"Evaluating swarm on {dataset.__class__.__name__} split {dataset.split}")

        realized_graph: Optional[Graph]
        if mode == 'full_connected_swarm':
            realized_graph = self._swarm.connection_dist.realize_full(self._swarm.composite_graph)
        elif mode == 'external_edge_probs':
            # assert edge_probs is not None
            # edge_mask = edge_probs > 0.5
            # realized_graph = self._swarm.connection_dist.realize_mask(self._swarm.composite_graph, edge_mask)
            # realized_graph.display()
            realized_graph = None
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
                if mode == 'external_edge_probs':
                    self._swarm.connection_dist.gat.eval()
                    realized_graph, edge_probs, _, _ = self._swarm.connection_dist.realize_gat(self._swarm.composite_graph, record, self._swarm.connection_dist.node_features, threshold=0.5)
                    realized_graph.display()
                    if edge_probs is not None:
                        self._print_conns(edge_probs, save_to_file=False)
                elif mode == 'randomly_connected_swarm':
                    realized_graph, _ = self._swarm.connection_dist.realize(self._swarm.composite_graph)
                assert realized_graph is not None

                input_dict = dataset.record_to_swarm_input(record)
                print(input_dict)

                future_answer = self._swarm.arun(input_dict, realized_graph, inference=True)
                future_answers.append(future_answer)

            raw_answers = await asyncio.gather(*future_answers)

            print(f"Batch time {time.time() - start_ts:.3f}")

            for raw_answer, record in zip(raw_answers, record_batch):
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

    def _print_conns(self, edge_probs: torch.Tensor, save_to_file: bool = False):
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
                txt_name = os.path.join(self._art_dir_name, "connections.txt")
                with open(txt_name, "w") as f:
                    f.writelines(msgs)

    async def optimize_swarm(
            self,
            num_iters: int,
            lr: float,
            batch_size: int = 1,
            ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset

        print(f"Optimizing swarm on {dataset.__class__.__name__} split {dataset.split}")
        
        num_params = sum(p.numel() for p in self._swarm.connection_dist.parameters() if p.requires_grad)
        params = [p for p in self._swarm.connection_dist.parameters() if p.requires_grad]
        print(f"Number of parameters to optimize for ADAM: {num_params}")
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4) # type: ignore # optimizer with weight decay (l2 regularization)
        # optimizer = torch.optim.Adam(self._swarm.connection_dist.parameters(), lr=lr) # type: ignore

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(lr=lr,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               model_name=self._model_name
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            # np.random.seed(42)
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()
        
        # node and edge statistics
        num_pot_edges = len(self._swarm.connection_dist.potential_connections)
        num_nodes = len(self._swarm.composite_graph.nodes)
        num_node_features = self._swarm.connection_dist.node_features.shape[0] + self._swarm.connection_dist.node_features.shape[1]
        
        async def utility_func(particle: List[float]) -> np.array: # type: ignore
            future_answers = []
            correct_answers = []
            fitness = []
            
            # # edge probs
            # edge_probs = particle[:len(self._swarm.connection_dist.edge_logits)]
            
            # GAT params
            order_params = torch.tensor(particle[:len(self._swarm.connection_dist.order_params)], device=self.device, dtype=torch.float32) # type: ignore
            node_features = torch.tensor(particle[len(self._swarm.connection_dist.order_params):].reshape(num_nodes, -1), device=self.device, dtype=torch.float32) # type: ignore
            
            # swarm_copy = copy.deepcopy(self._swarm)
            # # update params
            # swarm_copy.connection_dist.order_params = order_params
            # swarm_copy.connection_dist.node_features = node_features
            
            log_probs = []
            for i, record in zip(range(batch_size), loader):
                # create edge mask based on the particle
                # edge_mask = []
                # for prob in edge_probs:
                #     edge_mask.append(prob > torch.rand(1))
                # edge_mask = torch.stack(edge_mask)
                # realize graph based on edge mask                
                realized_graph, edge_probs, log_prob, edge_logits = self._swarm.connection_dist.realize_gat(self._swarm.composite_graph, record, node_features) # type: ignore
                # edge probs all are None results in 0 fitness
                if len(edge_probs) == 1 and edge_probs[0].item() == 0.0:
                    edge_probs = torch.tensor([0.0]*num_pot_edges, device=self.device, dtype=torch.float32)
                    edge_mask = edge_probs > 0.5
                    realized_graph = self._swarm.connection_dist.realize_mask(self._swarm.composite_graph, edge_mask) # type: ignore
                    print("Edge probs are None")
                    self._print_conns(edge_probs, save_to_file=False)
                    return np.array([0.0])
                log_probs.append(log_prob)
                self._print_conns(edge_probs, save_to_file=False)
                input_dict = dataset.record_to_swarm_input(record)
                future_answer = self._swarm.arun(input_dict, realized_graph, inference=False) # type: ignore
                future_answers.append(future_answer)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
            answers = await asyncio.gather(*future_answers)
            
            # every answer is a tuple of output_answer and intermediate_answers
            raw_output_answers = [answer[0] for answer in answers]
            raw_intermediate_answers = [answer[1] for answer in answers]
            
            current_utilities = []
            loss_list = []
            for raw_output_answer, raw_intermediate_answer, correct_answer, log_prob in zip(raw_output_answers, raw_intermediate_answers, correct_answers, log_probs):
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
                utility = 0.5*output_accuracy.get() + 0.5*intermediate_accuracy.get()
                current_utilities.append(utility)
                # self.utilities.append(utility)
                single_loss = -log_prob * output_accuracy.get()
                # single_loss = -log_prob * output_accuracy.get()
                loss_list.append(single_loss)
            fitness.append(np.mean(current_utilities))
            
            # if len(self.utilities) < batch_size:
            #     moving_averages = np.array([np.mean(self.utilities) for _ in range(batch_size)])
            # else:
            #     moving_averages = np.array([np.mean(self.utilities[-batch_size:]) for _ in range(batch_size)])
            # moving_averages = np.array([np.array(0.4) for _ in range(batch_size)])
            print("log_probs:", log_probs)
            # total_loss = (-torch.stack(log_probs) * torch.tensor(np.array(current_utilities) - moving_averages)).mean()
            # total_loss = (-torch.stack(log_probs) * torch.tensor(np.array(current_utilities) - moving_averages)).mean() + 0.01 * (edge_logits**2).mean()
            # total_loss = (-torch.stack(log_probs) * torch.tensor(np.array(loss_list)).to(self.device)).mean()
            
            # total_loss = torch.mean(torch.stack(loss_list)) + 0.01 * (edge_logits**2).mean()
            print("utilities:", current_utilities)
            print("fitness:", fitness)
            print("loss:", single_loss.item())
            # total_loss.requires_grad = True
            # total_loss.backward()
            # optimizer.step()
            
            if self._logger is not None:
                self._logger.add_scalar("train/loss", single_loss.item())
                self._logger.add_scalar("train/utility", np.mean(fitness))
            if self._art_dir_name is not None:
                log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
                with open(log_jsonl_name, "a") as f:
                    json.dump(dict(train_loss=single_loss.item(), train_utility=np.mean(fitness)), f)
                    f.write("\n")
            return np.array(fitness), torch.stack(loss_list), edge_logits
        
                
        # init GA
        swarm = self._swarm
        logger = self._logger
        num_paras = len(self._swarm.connection_dist.order_params) + len(self._swarm.connection_dist.node_features.reshape(-1))
        print(f"Number of parameters for GARL to optimize: {num_paras}")
        ga = GARL(func=utility_func, n_dim=num_paras, constraint_eq=None, constraint_ueq=None, lb=-1, ub=1,
                size_pop=10, max_iter=num_iters, prob_mut=0.1, n_processes=10, optimizer=optimizer,
                utilities=None, swarm=swarm, logger=logger, art_dir_name=self._art_dir_name)
        
        # edge_probs        
        parameters = await ga.run()
        
        order_params = parameters[:num_nodes] # type: ignore
        # create node features
        node_features = torch.tensor(parameters[num_nodes:].reshape(num_nodes, -1), device=self.device, dtype=torch.float32) # type: ignore
        # self._swarm.connection_dist.edge_logits = edge_probs
        self._swarm.connection_dist.order_params = order_params
        self._swarm.connection_dist.node_features = node_features
        print("GARL Done!")

        # if edge_probs is not None:
        #     self._print_conns(edge_probs, save_to_file=True)
        
        

        print("Done!")
        # edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        return parameters # type: ignore