#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987


import random
import numpy as np
from copy import deepcopy
import torch
from .base import SkoBase
from .tools import func_transformer
from abc import ABCMeta, abstractmethod
from .operators import crossover, mutation, ranking, selection

import time
import json
import os

import asyncio


class GeneticAlgorithmBase(SkoBase, metaclass=ABCMeta):
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200, prob_mut=0.001,
                 constraint_eq=tuple(), constraint_ueq=tuple(), early_stop=None, n_processes=0, optimizer=None,
                 utilities=None, swarm=None, logger=None, art_dir_name=None, evaluator=None, use_learned_order=False):
        
        self.func = func_transformer(func, n_processes)
        assert size_pop % 2 == 0, 'size_pop must be even integer'
        self.size_pop = size_pop  # size of population
        self.max_iter = max_iter
        self.prob_mut = prob_mut  # probability of mutation
        self.n_dim = n_dim
        self.early_stop = early_stop
        self.optimizer = optimizer

        # constraint:
        if constraint_eq is not None or constraint_ueq is not None:
            self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
            self.constraint_eq = list(constraint_eq)  # a list of equal functions with ceq[i] = 0
            self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with c[i] <= 0
        else:
            self.has_constraint = False
            self.constraint_eq = []
            self.constraint_ueq = []

        self.Chrom = None
        self.X = None  # shape = (size_pop, n_dim)
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.Y = None  # shape = (size_pop,) , value is f(x) + penalty for constraint
        self.FitV = None  # shape = (size_pop,)

        self._art_dir_name = art_dir_name

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

        self.mean_x, self.mean_y = None, None
        self.best_x, self.best_y = None, None

    @abstractmethod
    def chrom2x(self, Chrom):
        pass

    async def x2y(self):
        self.Y_raw = await self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    # @abstractmethod
    # def ranking(self):
    #     pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def crossover(self):
        pass

    @abstractmethod
    def mutation(self):
        pass
    
    @abstractmethod
    def print_conns(self):
        pass
    

    async def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        best = []
        num_pot_edges = len(self._swarm.connection_dist.potential_connections)
        num_nodes = len(self._swarm.composite_graph.nodes)
        num_node_features = len(self._swarm.connection_dist.node_features.reshape(-1))
        for i in range(self.max_iter):
            print(f"Iter {i}", 80*'-')
            start_ts = time.time()
            
            self.X = self.chrom2x(self.Chrom)
            print("X:", self.X)

            self.optimizer.zero_grad()
            self.Y = await self.x2y()
            self.optimizer.step()
            # print(self.optimizer.param_groups[0]['params'][0])
            # import pdb; pdb.set_trace()
            # squeeze the fitness value to 1D array
            self.Y = np.squeeze(self.Y)
            print("Y:", self.Y)
            swarm_fitness = np.mean(self.Y)
            self.Y = [0.9*f + 0.1*swarm_fitness for f in self.Y] 
            self.FitV = np.array(self.Y)
            print('Generation:', i, 'Best FitV:', 100 * self.FitV.max())
            print('Generation:', i, 'Mean FitV:', 100 * self.FitV.mean())
            # self.ranking()
            self.selection()
            self.crossover()
            self.mutation()
            
            print(f"Batch time {time.time() - start_ts:.3f}")

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)
            
            # output the edge probabilities
            global_best_index = np.array(self.generation_best_Y).argmax()
            self.mean_x = np.mean(self.X)
            self.mean_y = np.mean(self.Y)
            self.best_x = torch.tensor(self.generation_best_X[global_best_index])
            self.best_y = self.generation_best_Y[global_best_index]
            
            # self.best_x = torch.nn.Sigmoid()(self.best_x)         
            
            # update the edge probabilities with fittest individual
            # edge_probs = torch.nn.Parameter(self.best_x[:num_pot_edges])
            # order_params = torch.nn.Parameter(self.best_x[num_pot_edges:])
            order_params = self.best_x[:num_nodes]
            # create 32-dimensional node features
            node_features = torch.tensor(self.best_x[num_nodes:(num_nodes+num_node_features)].reshape(num_nodes, -1), device=self.device, dtype=torch.float32)
            # gat_params = self.best_x[(num_nodes+num_node_features):]
            # self._swarm.connection_dist.edge_logits = edge_probs
            self._swarm.connection_dist.order_params = order_params
            self._swarm.connection_dist.node_features = node_features
            
            
            order_probs = torch.nn.Sigmoid()(order_params)
            print("Order probabilities:")
            print(order_probs)
            # print("Edge probabilities:")
            # print(edge_probs)
            # print("Order parameters:")
            # print(order_params)

            # self.print_conns(edge_probs)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        break
                    else:
                        best.pop(0)


        if self._logger is not None:
            # self._logger.add_scalar("train/loss", total_loss.item(), i_iter)
            self._logger.add_scalar("train/utility", self.mean_y, i)
        if self._art_dir_name is not None:
            log_jsonl_name = os.path.join(self._art_dir_name, "training.jsonl")
            with open(log_jsonl_name, "a") as f:
                json.dump(dict(iter=i, #train_loss=total_loss.item(), 
                                train_utility=self.mean_y), f)
                f.write("\n")
        print("end of iteration")
        
        return self.best_x

    fit = run


class GARL(GeneticAlgorithmBase):
    """genetic algorithm for Reinforcement Learning

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint
    constraint_ueq : tuple
        unequal constraint
    precision : array_like
        The precision of every variables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    n_processes : int
        Number of processes, 0 means use all cpu
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py
    """

    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=200,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None, n_processes=0, optimizer=None,
                 utilities=None, swarm=None, logger=None, art_dir_name=None, evaluator=None, use_learned_order=False):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq, early_stop, n_processes, optimizer, utilities, swarm, logger, art_dir_name, evaluator, use_learned_order)

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array

        self._swarm = swarm
        self._logger = logger

        # Lind is the num of genes of every variable of func（segments）
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)
        
        

        # if precision is integer:
        # if Lind_raw is integer, which means the number of all possible value is 2**n, no need to modify
        # if Lind_raw is decimal, we need ub_extend to make the number equal to 2**n,
        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        self.int_mode = np.any(self.int_mode_)
        if self.int_mode:
            self.ub_extend = np.where(self.int_mode_
                                      , self.lb + (np.exp2(self.Lind) - 1) * self.precision
                                      , self.ub)

        self.len_chrom = sum(self.Lind)
        
        self.device = "cuda:7" if torch.cuda.is_available() else "cpu"
        
        self.crtbp()

    def crtbp(self):
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        self.Chrom = torch.tensor(self.Chrom, device=self.device, dtype=torch.int8)
        return self.Chrom

    def gray2rv(self, gray_code):
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
            '''
            We do not intend to make all operators as tensor,
            because objective function is probably not for pytorch
            '''
            Chrom = Chrom.cpu().numpy()
            cumsum_len_segment = self.Lind.cumsum()
            X = np.zeros(shape=(self.size_pop, self.n_dim))
            for i, j in enumerate(cumsum_len_segment):
                if i == 0:
                    Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                else:
                    Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                X[:, i] = self.gray2rv(Chrom_temp)

            if self.int_mode:
                X = self.lb + (self.ub_extend - self.lb) * X
                X = np.where(X > self.ub, self.ub, X)
            else:
                X = self.lb + (self.ub - self.lb) * X
            return X
        
        
    def print_conns(self, edge_probs: torch.Tensor, save_to_file: bool = False):
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


    # ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation