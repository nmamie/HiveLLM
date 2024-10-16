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
import pickle

from swarm.optimizer.edge_optimizer.evo_ppo.envs_repo.constructor import EnvConstructor
from swarm.optimizer.edge_optimizer.evo_ppo.models.constructor import ModelConstructor
from swarm.optimizer.edge_optimizer.evo_ppo.algos.erl_trainer import ERL_Trainer

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
            args: Any = None
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
            
        self.device = torch.device(args.gpu_id if torch.cuda.is_available() else 'cpu')
        
        self.args = args

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
            raw_answer = raw_answer[0]

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
            # edge_probs: Optional[torch.Tensor] = None,
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

            raw_answers = []
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

                raw_answer = self._swarm.run(input_dict, realized_graph, inference=True)
                raw_answers.append(raw_answer)

            # raw_answers = await asyncio.gather(*future_answers)

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
            lr: float = 1e-3,
            batch_size: int = 1,
            ):

        assert self._swarm is not None

        train_dataset = self._train_dataset
        val_dataset = self._val_dataset

        print(f"Optimizing swarm on {train_dataset.__class__.__name__} split {train_dataset.split}")
        print(f"Validating on {val_dataset.__class__.__name__} split {val_dataset.split}")
        
        # num_params = sum(p.numel() for p in self._swarm.connection_dist.parameters() if p.requires_grad)
        # params = [p for p in self._swarm.connection_dist.parameters() if p.requires_grad]
        # print(f"Number of parameters to optimize for ADAM: {num_params}")
        # optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4) # type: ignore # optimizer with weight decay (l2 regularization)

        if self._art_dir_name is not None:
            hp_json_name = os.path.join(self._art_dir_name, "hp.json")
            with open(hp_json_name, "w") as f:
                json.dump(dict(lr=lr,
                               batch_size=batch_size,
                               num_iters=num_iters,
                               model_name=self._model_name
                               ), f)

        # def infinite_data_loader() -> Iterator[pd.DataFrame]:
        #     perm = np.random.permutation(len(dataset))
        #     while True:
        #         for idx in perm:
        #             record = dataset[idx.item()]
        #             yield record

        # loader = infinite_data_loader()
        
        # realize full swarm
        realized_graph = self._swarm.connection_dist.realize_full(self._swarm.composite_graph)
        
        # node and edge statistics
        num_pot_edges = len(self._swarm.connection_dist.potential_connections)
        num_nodes = len(self._swarm.composite_graph.nodes)
        num_node_features = self.args.node_feature_size
        
        potential_connections = self._swarm.connection_dist.potential_connections
        
        ################################## Find and Set MDP (environment constructor) ########################
        if self.args.train:
            train_env_constructor = EnvConstructor(realized_graph, train_dataset, self.args.train, num_pot_edges, num_nodes, num_node_features, self._swarm.connection_dist.node_features, self._swarm.connection_dist.state_indicator, self._swarm.connection_dist.node_id2idx, self._swarm.connection_dist.node_idx2id, self._swarm.connection_dist.edge_index, batch_size, self.args.num_envs)
            val_env_constructor = EnvConstructor(realized_graph, val_dataset, self.args.train, num_pot_edges, num_nodes, num_node_features, self._swarm.connection_dist.node_features, self._swarm.connection_dist.state_indicator, self._swarm.connection_dist.node_id2idx, self._swarm.connection_dist.node_idx2id, self._swarm.connection_dist.edge_index, batch_size, self.args.num_envs)
        else:
            train_env_constructor = None
            val_env_constructor = EnvConstructor(realized_graph, val_dataset, self.args.train, num_pot_edges, num_nodes, num_node_features, self._swarm.connection_dist.node_features, self._swarm.connection_dist.state_indicator, self._swarm.connection_dist.node_id2idx, self._swarm.connection_dist.node_idx2id, self._swarm.connection_dist.edge_index, batch_size, self.args.num_envs)

        #######################  Actor, Critic and ValueFunction Model Constructor ######################
        model_constructor = ModelConstructor(val_env_constructor.state_dim, val_env_constructor.action_dim, self.args.hidden_size, potential_connections)
        
        # train and evaluate AI
        ai = ERL_Trainer(self.args, self._art_dir_name, self._swarm, model_constructor, train_env_constructor, val_env_constructor, num_nodes, num_pot_edges)
        
        # training or evaluation
        if self.args.train:
            ai.train(self.args.total_steps)
        score = await ai.eval(val_dataset, limit_questions=153)
        return score