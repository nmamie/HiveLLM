import os
import asyncio
import pandas as pd
from typing import Iterable, Optional, Iterator, Union, Literal, List, Dict, Any
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
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
        test_dataset: Optional[BaseDataset] = None,
        model_name: Optional[str] = None,
        enable_tensorboard: bool = False,
        enable_artifacts: bool = False,
        tensorboard_tag: Optional[str] = None,
    ) -> None:

        self._swarm: Optional[Swarm] = swarm
        self._train_dataset: BaseDataset = train_dataset
        self._val_dataset: BaseDataset = val_dataset
        self._test_dataset: Optional[BaseDataset] = test_dataset
        self._model_name: Optional[str] = model_name

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        art_dir_name = (f"{datetime_str}" +
                        (f"_{tensorboard_tag}" if tensorboard_tag is not None else ""))

        if enable_artifacts or enable_tensorboard:
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

        self.utilities = []

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    async def evaluate_direct_answer(self,
                                     limit_questions: Optional[int] = None,
                                     ) -> float:

        dataset = self._test_dataset

        print(
            f"Evaluating DirectAnswer on {dataset.get_domain()} split {dataset.split}")

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
        
        # validation mode
        self._swarm.connection_dist.gat.eval()
        
        dataset = self._test_dataset

        print(
            f"Evaluating swarm on {dataset.__class__.__name__} split {dataset.split}")

        realized_graph: Optional[Graph]
        if mode == 'full_connected_swarm':
            realized_graph = self._swarm.connection_dist.realize_full(
                self._swarm.composite_graph)
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

        data_len = min(
            len(dataset), limit_questions) if limit_questions is not None else len(dataset)
        num_batches = int(math.ceil(data_len / eval_batch_size))

        for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
            print(80*'-')

            start_ts = time.time()

            future_answers = []
            for record in record_batch:
                if mode == 'external_edge_probs':
                    realized_graph, edge_probs, _, _ = self._swarm.connection_dist.realize_gat(
                        self._swarm.composite_graph, record, self._swarm.connection_dist.node_features, threshold=0.5)
                    # realized_graph.display()
                    if edge_probs is not None:
                        self._print_conns(edge_probs, save_to_file=False)
                elif mode == 'randomly_connected_swarm':
                    realized_graph, _ = self._swarm.connection_dist.realize(
                        self._swarm.composite_graph)
                assert realized_graph is not None

                input_dict = dataset.record_to_swarm_input(record)
                print(input_dict)

                future_answer = self._swarm.arun(
                    input_dict, realized_graph, inference=True)
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
            eval_json_name = os.path.join(
                self._art_dir_name, "evaluation.json")
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
                    torch.save(self._swarm.connection_dist.state_dict(), os.path.join(
                        self._art_dir_name, "edge_logits_final.pt"))
                # else:
                #     torch.save(self._swarm.connection_dist.state_dict(), os.path.join(
                #         self._art_dir_name, f"edge_logits_{i_iter}.pt"))

    async def optimize_swarm(
            self,
            num_iters: int,
            lr: float,
            beta: float,
            batch_size: int = 4,
    ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset

        print(
            f"Optimizing swarm on {dataset.__class__.__name__} split {dataset.split}")
        
        # training mode
        self._swarm.connection_dist.gat.train()

        num_params_gat = sum(
            p.numel() for p in self._swarm.connection_dist.gat.parameters() if p.requires_grad)
        num_params = sum(
            p.numel() for p in self._swarm.connection_dist.parameters() if p.requires_grad)
        params = [p for p in self._swarm.connection_dist.gat.parameters()
                  if p.requires_grad]
        print(f"Number of parameters to optimize for ADAM: {num_params_gat}")
        print(num_params)
        # optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4) # type: ignore # optimizer with weight decay (l2 regularization)
        policy_optimizer = torch.optim.Adam(
            self._swarm.connection_dist.gat.parameters(), lr=lr)
        if beta > 0:
            baseline_optimizer = torch.optim.Adam(
                [self._swarm.connection_dist.baseline], lr=beta)

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(lr=lr,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               model_name=self._model_name
                               ), f)

        def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record

        loader = infinite_data_loader()

        for i_iter in range(num_iters):

            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            # node and edge statistics
            num_pot_edges = len(
                self._swarm.connection_dist.potential_connections)
            num_nodes = len(self._swarm.composite_graph.nodes)
            # num_node_features = self._swarm.connection_dist.node_features.shape[0] + self._swarm.connection_dist.node_features.shape[1]

            future_answers = []
            correct_answers = []

            # # edge probs
            # edge_probs = particle[:len(self._swarm.connection_dist.edge_logits)]

            node_features = self._swarm.connection_dist.node_features

            log_probs = []
            for i, record in zip(range(batch_size), loader):
                # create edge mask based on the particle
                # edge_mask = []
                # for prob in edge_probs:
                #     edge_mask.append(prob > torch.rand(1))
                # edge_mask = torch.stack(edge_mask)
                # realize graph based on edge mask
                realized_graph, edge_probs, log_prob, edge_logits = self._swarm.connection_dist.realize_gat(
                    self._swarm.composite_graph, record, node_features)  # type: ignore
                # edge probs all are None results in 0 fitness
                log_probs.append(log_prob)
                self._print_conns(edge_probs, save_to_file=False)
                input_dict = dataset.record_to_swarm_input(record)
                future_answer = self._swarm.arun(
                    input_dict, realized_graph, inference=False)  # type: ignore
                future_answers.append(future_answer)
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
            raw_answers = await asyncio.gather(*future_answers)

            print(f"Batch time {time.time() - start_ts:.3f}")

            # # every answer is a tuple of output_answer and intermediate_answers
            # raw_output_answers = [answer[0] for answer in answers]
            # raw_intermediate_answers = [answer[1] for answer in answers]

            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []

            for raw_answer, correct_answer, log_prob in zip(raw_answers, correct_answers, log_probs):
                raw_answer = raw_answer[0][0]
                answer = dataset.postprocess_answer(raw_answer)
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"

                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)

                state_value = torch.sigmoid(
                    self._swarm.connection_dist.baseline)

                # Compute advantage
                if beta > 0:
                    advantage = utility - state_value
                else:
                    advantage = utility
                single_loss = -log_prob * advantage
                loss_list.append(single_loss)

            print("utilities:", utilities)
            print("state_value:", state_value.item())
            mean_utility = np.mean(np.array(utilities))
            print("mean_utility:", mean_utility)

            # Compute policy loss with l1 on edge_logits to encourage sparsity
            total_loss = torch.mean(torch.stack(loss_list))
            print("loss:", total_loss.item())
            # print("L1 loss:", 0.01 * torch.sum(torch.abs(
            #     torch.stack(edges_probs))).item())

            policy_optimizer.zero_grad()
            total_loss.backward()
            # print("Grad logits:", self._swarm.connection_dist.gat.edge_logits.grad)
            policy_optimizer.step()

            if beta > 0:
                # Update baseline
                baseline_loss = F.mse_loss(
                    torch.sigmoid(self._swarm.connection_dist.baseline),
                    torch.tensor(mean_utility, dtype=torch.float32, requires_grad=False))
                print("baseline_loss:", baseline_loss.item())
                baseline_optimizer.zero_grad()
                baseline_loss.backward()
                print("Grad baseline:", self._swarm.connection_dist.baseline.grad)
                baseline_optimizer.step()

            # print("edge_logits:", self._swarm.connection_dist.gat.edge_logits)
            # edge_probs = torch.sigmoid(
            #     self._swarm.connection_dist.gat.edge_logits)
            # print("edge_probs:", edge_probs)

            print("baseline_logits:", self._swarm.connection_dist.baseline)
            baseline = torch.sigmoid(self._swarm.connection_dist.baseline)
            print("baseline:", baseline)

        # self._print_conns(edge_probs, save_to_file=True, i_iter=i_iter+1)

        if self._logger is not None:
            self._logger.add_scalar(
                "train/loss", total_loss.item(), i_iter)
            self._logger.add_scalar(
                "train/utility", mean_utility.item(), i_iter)
        if self._art_dir_name is not None:
            log_jsonl_name = os.path.join(
                self._art_dir_name, "training.jsonl")
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(iter=i_iter, train_loss=total_loss.item(
                ), train_utility=mean_utility.item()), f)
                f.write("\n")
        print("end of iteration")

        # if edge_probs is not None:
        #     self._print_conns(edge_probs, save_to_file=True)

        print("Done!")
        # edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        return log_probs  # type: ignore