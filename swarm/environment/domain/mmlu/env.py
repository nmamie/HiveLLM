
import numpy as np


class GymWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, dataset, loader, num_pot_edges, num_nodes, num_node_features, batch_size):
        """
        A base template for all environment wrappers.
        """
        self.is_discrete = True
        self.dataset = dataset
        self.loader = loader
        self.num_pot_edges = num_pot_edges
        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.batch_size = batch_size
        self.current_node_id = None
        
        #State and Action Parameters
        self.state_dim = self.num_nodes * self.num_node_features
        if self.is_discrete:
            self.action_dim = self.num_nodes
        self.test_size = 10

    def reset(self, node_features, node2idx):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """
        state, current_node_id = self.env.reset(node_features, node2idx)
        self.current_node_id = current_node_id
        return state

    def step(self, record, node_features, node2idx, action): #Expects a numpy action
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
        for record in zip(range(self.batch_size), self.loader):
            next_state, reward, terminate, final_answers, current_node_id = self.env.step(self.dataset, record, current_node_id=current_node_id, node_features=node_features, node2idx=node2idx, action=action)
            self.current_node_id = current_node_id
            rewards += reward
            if terminate: break        
        
        # next_state = np.expand_dims(next_state, axis=0)
        return next_state, reward, terminate, final_answers