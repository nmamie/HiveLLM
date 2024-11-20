from copy import deepcopy
import json
import math
from typing import Any, Dict, Iterator, List
from matplotlib.pyplot import step
import numpy as np, os, time, random, torch, sys

from tqdm import tqdm
from swarm.optimizer.edge_optimizer.evo_ppo.algos.neuroevolution import SSNE
from swarm.optimizer.edge_optimizer.evo_ppo.core import utils
from swarm.optimizer.edge_optimizer.evo_ppo.core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
from swarm.optimizer.edge_optimizer.evo_ppo.core.buffer import Buffer
from swarm.utils.log import logger
from experiments.evaluator.accuracy import Accuracy
import torch

class ERL_Trainer:

	def __init__(self, args, art_dir_name, swarm, realized_graph, model_constructor, env_constructor, num_nodes, num_edges):

		self.args = args
		self._art_dir_name = art_dir_name
		self.policy_string = 'CategoricalPolicy' if env_constructor.is_discrete else 'Gaussian_FF'
		self.manager = Manager()
		self._swarm = swarm
		self._realized_graph = realized_graph
		self.num_nodes = num_nodes
		self.num_edges = num_edges
		self.pruned_nodes = []
		self.env_constructor = env_constructor
		self.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
  
  		#Save best policy
		self.best_policy = model_constructor.make_model(self.policy_string)
		self.gen_best_policy = model_constructor.make_model(self.policy_string)
		self.grad_best_policy = model_constructor.make_model(self.policy_string)

		if self.args.train:
			#Evolution
			self.evolver = SSNE(self.args)

			#Initialize population
			self.population = self.manager.list()
			for _ in range(args.pop_size):
				self.population.append(model_constructor.make_model(self.policy_string))

			#PG Learner
			if env_constructor.is_discrete:
				from swarm.optimizer.edge_optimizer.evo_ppo.algos.ddqn import DDQN
				self.learner = DDQN(args, model_constructor)
			else:
				from swarm.optimizer.edge_optimizer.evo_ppo.algos.sac import SAC
				self.learner = SAC(args, model_constructor)

			#Replay Buffer
			self.replay_buffer = Buffer(args.buffer_size)

			#Initialize Rollout Bucket
			self.rollout_bucket = self.manager.list()
			for _ in range(args.rollout_size):
				self.rollout_bucket.append(model_constructor.make_model(self.policy_string))

			############## MULTIPROCESSING TOOLS ###################
			#Evolutionary population Rollout workers
			self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
			self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
			self.evo_workers = [Process(target=rollout_worker, args=(id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], args.rollout_size > 0, self.population, env_constructor)) for id in range(args.pop_size)]
			for worker in self.evo_workers: worker.start()
			self.evo_flag = [True for _ in range(args.pop_size)]

			#Learner rollout workers
			self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
			self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
			self.workers = [Process(target=rollout_worker, args=(id, 'pg', self.task_pipes[id][1], self.result_pipes[id][0], True, self.rollout_bucket, env_constructor)) for id in range(args.rollout_size)]
			for worker in self.workers: worker.start()
			self.roll_flag = [True for _ in range(args.rollout_size)]

			#Test bucket
			self.test_bucket = self.manager.list()
			self.test_bucket.append(model_constructor.make_model(self.policy_string))

			# Test workers
			self.test_task_pipes = [Pipe() for _ in range(args.num_test)]
			self.test_result_pipes = [Pipe() for _ in range(args.num_test)]
			self.test_workers = [Process(target=rollout_worker, args=(id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], False, self.test_bucket, env_constructor)) for id in range(args.num_test)]
			for worker in self.test_workers: worker.start()
			self.test_flag = False

			#Trackers
			self.best_score = -float('inf'); self.gen_frames = 0; self.total_frames = 0; self.test_score = -float('inf'); self.test_std = None
			self.grad_best_score = -float('inf')

			logger.info("ERL Trainer Initialized")
   
		else:
			logger.info("ERL Trainer Initialized for Evaluation")


	def forward_generation(self, gen, tracker):

		gen_max = -float('inf')

		#Start Evolution rollouts
		if self.args.pop_size > 1:
			for id, actor in enumerate(self.population):
				self.evo_task_pipes[id][0].send(id)

		#Sync all learners actor to cpu (rollout) actor and start their rollout
		self.learner.actor.cpu()
		for rollout_id in range(len(self.rollout_bucket)):
			utils.hard_update(self.rollout_bucket[rollout_id], self.learner.actor)
			self.task_pipes[rollout_id][0].send(0)
		self.learner.actor.to(device=self.device)

		#Start Test rollouts
		if gen % self.args.test_frequency == 0:
			self.test_flag = True
			for pipe in self.test_task_pipes: pipe[0].send(0)


		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		if self.replay_buffer.__len__() > self.args.learning_start: ###BURN IN PERIOD
			for _ in range(int(self.gen_frames * self.args.gradperstep)):
				s, ns, emb, a, r, n, nn, t, e, done = self.replay_buffer.sample(self.args.batch_size)
				self.learner.update_parameters(s, ns, emb, a, r, n, nn, t, e, done, self.args.batch_size, self.args.node_feature_size, self.num_nodes, self.num_edges)

			self.gen_frames = 0


		########## JOIN ROLLOUTS FOR EVO POPULATION ############
		all_fitness = []; all_eplens = []
		if self.args.pop_size > 1:
			for i in range(self.args.pop_size):
				_, fitness, frames, pruned_nodes, trajectory = self.evo_result_pipes[i][1].recv()

				all_fitness.append(fitness); all_eplens.append(frames)
				self.gen_frames+= frames; self.total_frames += frames
				self.replay_buffer.add(trajectory)
				self.best_score = max(self.best_score, fitness)
				if fitness == self.best_score:
					self.pruned_nodes = pruned_nodes
				gen_max = max(gen_max, fitness)

		########## JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		rollout_fitness = []; rollout_eplens = []
		if self.args.rollout_size > 0:
			for i in range(self.args.rollout_size):
				_, fitness, pg_frames, pruned_nodes, trajectory = self.result_pipes[i][1].recv()
				self.replay_buffer.add(trajectory)
				self.gen_frames += pg_frames; self.total_frames += pg_frames
				self.best_score = max(self.best_score, fitness)
				self.grad_best_score = max(self.grad_best_score, fitness)
				# if fitness == self.best_score:
				# 	self.pruned_nodes = pruned_nodes
				gen_max = max(gen_max, fitness)
				rollout_fitness.append(fitness); rollout_eplens.append(pg_frames)
			if max(rollout_fitness) >= self.grad_best_score:
				champ_index = rollout_fitness.index(max(rollout_fitness))
				champ = self.rollout_bucket[champ_index]
				utils.hard_update(self.grad_best_policy, champ)

		######################### END OF PARALLEL ROLLOUTS ################

		############ FIGURE OUT THE CHAMP POLICY AND SYNC IT TO TEST #############
		if self.args.pop_size > 1:
			champ_index = all_fitness.index(max(all_fitness))
			champ = self.population[champ_index]
			utils.hard_update(self.test_bucket[0], champ)
			logger.info('Champion fitness:%.2f, Current All-time Champion:%.2f' % (max(all_fitness), self.best_score))
			# safe currently best policy
			utils.hard_update(self.gen_best_policy, champ)
			# # Save best policy
			# if max(all_fitness) >= self.best_score:
			# 	self.best_score = max(all_fitness)
			# 	utils.hard_update(self.best_policy, champ)
			# 	torch.save({
			# 		'policy': self.best_policy.state_dict(), 
			# 		# 'init': self._swarm.connection_dist.node_features,
			# 		# 'state': self._swarm.connection_dist.state_indicator,
			# 		# 'edges': self._swarm.connection_dist.edge_index,
			# 		'swarm': self._swarm,				
			# 		'graph': self._realized_graph,
			# 		# 'node2idx': self._swarm.connection_dist.node_id2idx,
			# 		# 'idx2node': self._swarm.connection_dist.node_idx2id,
			# 		'pruned_nodes': self.pruned_nodes
			# 	}, self.args.aux_folder + '_best'+self.args.savetag)
			# 	logger.info('Best policy saved with score:%.2f' % max(all_fitness))

		else: #If there is no population, champion is just the actor from policy gradient learner
			utils.hard_update(self.test_bucket[0], self.rollout_bucket[0])
			utils.hard_update(self.best_policy, self.rollout_bucket[0])
			torch.save({
					'policy': self.best_policy.state_dict(), 
					# 'init': self._swarm.connection_dist.node_features,
					# 'state': self._swarm.connection_dist.state_indicator,
					# 'edges': self._swarm.connection_dist.edge_index,
					'swarm': self._swarm,				
					'graph': self._realized_graph,
					# 'node2idx': self._swarm.connection_dist.node_id2idx,
					# 'idx2node': self._swarm.connection_dist.node_idx2id,
					'pruned_nodes': self.pruned_nodes
				}, self.args.aux_folder + '_best'+self.args.savetag)
			logger.info('Best policy saved with score:%.2f' % max(rollout_fitness))


		###### TEST SCORE ######
		if self.test_flag:
			self.test_flag = False
			test_scores = []
			for pipe in self.test_result_pipes: #Collect all results
				_, fitness, _, _, _ = pipe[1].recv()
				self.best_score = max(self.best_score, fitness)
				gen_max = max(gen_max, fitness)
				test_scores.append(fitness)
			test_scores = np.array(test_scores)
			test_mean = np.mean(test_scores); test_std = (np.std(test_scores))
			tracker.update([test_mean], self.total_frames)
			if test_mean >= self.test_score:
				self.test_score = test_mean
				utils.hard_update(self.best_policy, self.test_bucket[0])
				torch.save({
					'policy': self.best_policy.state_dict(), 
					# 'init': self._swarm.connection_dist.node_features,
					# 'state': self._swarm.connection_dist.state_indicator,
					# 'edges': self._swarm.connection_dist.edge_index,
					'swarm': self._swarm,				
					'graph': self._realized_graph,
					# 'node2idx': self._swarm.connection_dist.node_id2idx,
					# 'idx2node': self._swarm.connection_dist.node_idx2id,
					'pruned_nodes': self.pruned_nodes
				}, self.args.aux_folder + '_best'+self.args.savetag)
				logger.info('Best policy saved with score:%.2f' % test_mean)

		else:
			test_mean, test_std = None, None
   
		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			selection_stats = self.evolver.epoch(gen, self.population, all_fitness, self.rollout_bucket)
   
		if selection_stats['total'] > 0:
			logger.info('Elite Rate:%.2f, Selection Rate:%.2f, Discard Rate:%.2f' % (selection_stats['elite']/selection_stats['total'], (selection_stats['elite']+selection_stats['selected'])/selection_stats['total'], selection_stats['discarded']/selection_stats['total']))
		else:
			logger.info('RL transfer not happened yet')

		#Compute the champion's eplen
		champ_len = all_eplens[all_fitness.index(max(all_fitness))] if self.args.pop_size > 1 else rollout_eplens[rollout_fitness.index(max(rollout_fitness))]


		return gen_max, champ_len, all_eplens, test_mean, test_std, rollout_fitness, rollout_eplens


	def train(self, frame_limit):
		# Define Tracker class to track scores
		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()

		for gen in range(1, 1000000000):  # Infinite generations

			# Train one iteration
			try:
				max_fitness, champ_len, all_eplens, test_mean, test_std, rollout_fitness, rollout_eplens = self.forward_generation(gen, test_tracker)
			except KeyboardInterrupt:
				torch.save({
					'policy': self.gen_best_policy.state_dict(),
					'swarm': self._swarm,
					'graph': self._realized_graph,
					'pruned_nodes': self.pruned_nodes
				}, self.args.aux_folder + '_gen'+self.args.savetag)
				torch.save({
					'policy': self.grad_best_policy.state_dict(), 
					'swarm': self._swarm,
					'graph': self._realized_graph,
					'pruned_nodes': self.pruned_nodes
				}, self.args.aux_folder + '_grad'+self.args.savetag)
				print('Model checkpoints saved')
				print('Exiting...')
				break

			if test_mean: self.args.writer.add_scalar('test_score', test_mean, gen)
   
			# clean up cuda memory
			torch.cuda.empty_cache()

			print('Gen/Frames:', gen,'/',self.total_frames,
				  ' Gen_max_score:', '%.2f'%max_fitness,
				  ' Champ_len', '%.2f'%champ_len, ' Test_score u/std', utils.pprint(test_mean), utils.pprint(test_std),
				  ' Rollout_u/std:', utils.pprint(np.mean(np.array(rollout_fitness))), utils.pprint(np.std(np.array(rollout_fitness))),
				  ' Rollout_mean_eplen:', utils.pprint(sum(rollout_eplens)/len(rollout_eplens)) if rollout_eplens else None)

			if gen % 5 == 0:
				print('Best_score_ever:''/','%.2f'%self.best_score, ' FPS:','%.2f'%(self.total_frames/(time.time()-time_start)), 'savetag', self.args.savetag)
				print()

			if self.total_frames > frame_limit:
				# self._print_conns(torch.sigmoid(self.best_policy.edge_logits), save_to_file=True)
				torch.save({
					'policy': self.gen_best_policy.state_dict(),
					'swarm': self._swarm,
					'graph': self._realized_graph,
					'pruned_nodes': self.pruned_nodes
				}, self.args.aux_folder + '_gen'+self.args.savetag)
				torch.save({
					'policy': self.grad_best_policy.state_dict(), 
					'swarm': self._swarm,
					'graph': self._realized_graph,
					'pruned_nodes': self.pruned_nodes
				}, self.args.aux_folder + '_grad'+self.args.savetag)
				break

		###Kill all processes
		try:
			for p in self.task_pipes: p[0].send('TERMINATE')
			for p in self.test_task_pipes: p[0].send('TERMINATE')
			for p in self.evo_task_pipes: p[0].send('TERMINATE')
		except:
			None
	
   
	@torch.no_grad
	async def eval(self, val_dataset, limit_questions=None, eval_batch_size=1):
		"""
		Evaluate the model on the validation dataset
		Parameters:
			val_dataset (Dataset): Validation dataset
		Returns:
			float: Validation score
		"""
		if not self.args.train:             
			model_state = torch.load(self.args.aux_folder + '_best'+self.args.savetag)
			self.best_policy.load_state_dict(model_state['policy'])
			# self._swarm.connection_dist.node_features = model_state['init']
			# self._swarm.connection_dist.state_indicator = model_state['state']
			# self._swarm.connection_dist.edge_index = model_state['edges']
			self._swarm = model_state['swarm']
			self._realized_graph = model_state['graph']
			# self._swarm.connection_dist.node_id2idx = model_state['node2idx']
			# self._swarm.connection_dist.node_idx2id = model_state['idx2node']
			self.pruned_nodes = model_state['pruned_nodes']
		self.best_policy.to(self.device)
		self.best_policy.eval()

		env, num_envs = self.env_constructor.make_env(test=True, graph=self._realized_graph, node2idx=self._swarm.connection_dist.node_id2idx, idx2node=self._swarm.connection_dist.node_idx2id, node_features=self._swarm.connection_dist.node_features, state_indicator=self._swarm.connection_dist.state_indicator, edge_index=self._swarm.connection_dist.edge_index)
		# env.prune(self.pruned_nodes)
		print(f"Pruned nodes: {self.pruned_nodes}")
		accuracy = Accuracy()
  
		progress_bar = tqdm(total=limit_questions)
		while True:
			print(80*'-')
			state, edge_index, active_node_idx, record = await env.val_reset()
			print("Active node: ", active_node_idx)
			# # remove pruned nodes from edge_index
			# if len(self.pruned_nodes) > 0:
			# 	edge_index = utils.remove_pruned_nodes(edge_index, self.pruned_nodes)
			record, sentence_emb = record
			sentence = record['question']
			print("Question:", sentence)
			done = False
			terminate = False
			truncate = False
			steps = 0
			while not done:
				state = state.to(self.device)
				edge_index = edge_index.to(self.device)
				action = self.best_policy.clean_action(state, edge_index, active_node_idx, sentence_emb, step=steps, pruned_nodes=[])
				print("Action:", action)
				next_state, next_active_node_idx, reward, terminate, truncate, final_answers = await env.val_step(action, record, edge_index)
				# state_copy = deepcopy(state.cpu().numpy())
				# next_state_copy = deepcopy(next_state.cpu().numpy())
				# dist = np.linalg.norm(state_copy - next_state_copy, ord=2, axis=1)
				# print("State distance:", dist)
				state = deepcopy(next_state)
				active_node_idx = deepcopy(next_active_node_idx)
				steps += 1
				done = terminate or truncate
			raw_answer = final_answers[0]
			print("Raw answer:", raw_answer)
			answer = val_dataset.postprocess_answer(raw_answer)
			print("Postprocessed answer:", answer)
			correct_answer = val_dataset.record_to_target_answer(record)
			print("Correct answer:", correct_answer)
			accuracy.update(answer, correct_answer)
			accuracy.print()
			progress_bar.update(1)
			progress_bar.set_description(f"Accuracy: {accuracy.get():.2f}")
			if progress_bar.n >= limit_questions:
				break
		print("Final accuracy:")
		accuracy.print()

		self._dump_eval_results(dict(
			accuracy=accuracy.get(),
			limit_questions=limit_questions))
		
		print("Done!")
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