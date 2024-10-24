import asyncio
from typing import Any, Iterator, List
import numpy as np
import pandas as pd
import torch


class GymWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """

    def __init__(self, swarm, graph, bert, tokenizer, train_dataset, val_dataset, train, test, num_pot_edges, num_nodes, num_node_features, node_features, state_indicator, node2idx, idx2node, edge_index, batch_size, num_envs):
        """
        A base template for all environment wrappers.
        """
        self.swarm = swarm
        self.env = graph
        self.bert = bert
        self.tokenizer = tokenizer
        self.is_discrete = True
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train = train
        if test:
            self.dataset = self.val_dataset
        else:
            self.dataset = self.train_dataset
        # self.loader = loader
        self.num_pot_edges = num_pot_edges
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.batch_size = batch_size
        self.current_node_id = None
        self.node_features = node_features
        self.state_indicator = state_indicator
        self.node2idx = node2idx
        self.idx2node = idx2node
        self.edge_index = edge_index
        self.num_envs = num_envs
        self.pruned_nodes = []

        # State and Action Parameters
        self.state_dim = self.num_node_features
        if self.is_discrete:
            self.action_dim = self.num_nodes
        self.test_size = 10

        if self.train:
            self.loader = self._infinite_data_loader()
        else:
            self.loader = self._eval_loader(batch_size=1, dataset=self.dataset, limit_questions=153)

        print("Env Initialized")

    def _infinite_data_loader(self) -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(self.dataset))
            while True:
                for idx in perm:
                    record = self.dataset[idx.item()]
                    yield record

    def _eval_loader(self, batch_size: int, dataset: List[Any], limit_questions: int = None) -> Iterator[List[Any]]:
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
    
    def prune(self, pruned_nodes: List[int]):
        self.pruned_nodes = pruned_nodes

    def reset(self, records=None):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """
        state, self.current_node_id = asyncio.run(self.env.reset(self.node_features, self.pruned_nodes, self.node2idx))
        print("Current Node ID:", self.current_node_id)
        records = []
        for i, record in zip(range(self.num_envs), self.loader):
            # Encode sentence with BERT
            sentence = record['question']
            inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.bert.device)
            with torch.no_grad():
                sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
            records.append((record, sentence_embedding))
            
        state[self.node2idx[self.current_node_id]] = state[self.node2idx[self.current_node_id]] + self.state_indicator
        
        print("Current Node:", self.node2idx[self.current_node_id])

        return state, self.swarm.connection_dist.edge_index, self.node2idx[self.current_node_id], records
        

    def step(self, action, record, state, edge_index): #Expects a numpy action
        """Take an action to forward the simulation

            Parameters:
                action (ndarray): action to take in the env

            Returns:
                next_obs (list): Next state
                reward (float): Reward for this step
                terminate (bool): Simulation done?
                final_answers (list): answers of the LLM swarm
        """
        if  self.is_discrete:
           action = action[0]
        else:
            #Assumes action is in [-1, 1] --> Hyperbolic Tangent Activation
            action = self.action_low + (action + 1.0) / 2.0 * (self.action_high - self.action_low)

        rewards = 0
        terminate = False
        current_node_id = self.idx2node[action.item()]
        next_state, reward, terminate, final_answers, self.current_node_id, self.edge_index = asyncio.run(self.env.step(self.dataset, record, state, edge_index, current_node_id=current_node_id, node_features=self.node_features, node2idx=self.node2idx, action=action))
        next_state[self.node2idx[self.current_node_id]] = next_state[self.node2idx[self.current_node_id]] + self.state_indicator
                    
        # next_state = np.expand_dims(next_state, axis=0)
        return next_state, self.node2idx[self.current_node_id], reward, terminate, final_answers

        
    async def val_reset(self):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """
        if self.train:
            self.train = False
            self.loader = self._eval_loader(batch_size=1, dataset=self.dataset, limit_questions=153)
        state, self.current_node_id = await self.env.reset(self.node_features, self.pruned_nodes, self.node2idx)
        state[self.node2idx[self.current_node_id]] = state[self.node2idx[self.current_node_id]] + self.state_indicator
        records = []
        record = next(self.loader)
        # Encode sentence with BERT
        record = record[0]
        sentence = record['question']
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(self.bert.device)
        with torch.no_grad():
            sentence_embedding = self.bert(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
            sentence_embedding = sentence_embedding.to(self.bert.device)
        records.append((record, sentence_embedding)) 
            
        return state, self.swarm.connection_dist.edge_index, self.node2idx[self.current_node_id], records
        

    async def val_step(self, action, record, state, edge_index): #Expects a numpy action
        """Take an action to forward the simulation

            Parameters:
                action (ndarray): action to take in the env

            Returns:
                next_obs (list): Next state
                reward (float): Reward for this step
                terminate (bool): Simulation done?
                final_answers (list): answers of the LLM swarm
        """
        if  self.is_discrete:
           action = action[0]
        else:
            #Assumes action is in [-1, 1] --> Hyperbolic Tangent Activation
            action = self.action_low + (action + 1.0) / 2.0 * (self.action_high - self.action_low)

        rewards = 0
        terminate = False
        current_node_id = self.idx2node[action.item()]
            
        next_state, reward, terminate, final_answers, self.current_node_id, self.edge_index = await self.env.step(self.dataset, record, state, edge_index, current_node_id=current_node_id, node_features=self.node_features, node2idx=self.node2idx, action=action)
        next_state[self.node2idx[self.current_node_id]] = next_state[self.node2idx[self.current_node_id]] + self.state_indicator    
                    
        # next_state = np.expand_dims(next_state, axis=0)
        return next_state, self.node2idx[self.current_node_id], reward, terminate, final_answers