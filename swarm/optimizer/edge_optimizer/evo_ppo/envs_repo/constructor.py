from swarm.environment.domain.mmlu.env import GymWrapper

class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, swarm, dataset, train, num_pot_edges, num_nodes, num_node_features, node_features, state_indicator, node2idx, idx2node, edge_index, batch_size, num_envs):
        """
        A general Environment Constructor
        """
        self.swarm = swarm
        self.is_discrete = True
        self.train = train
        self.dataset = dataset
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
        
        #State and Action Parameters
        self.state_dim = self.num_node_features
        if self.is_discrete:
            self.action_dim = self.num_nodes

    def make_env(self, **kwargs):
        """
        Generate and return an env object
        """
        env = GymWrapper(self.swarm, self.dataset, self.train, self.num_pot_edges, self.num_nodes, self.num_node_features, self.node_features, self.state_indicator, self.node2idx, self.idx2node, self.edge_index, self.batch_size, self.num_envs)
        return env