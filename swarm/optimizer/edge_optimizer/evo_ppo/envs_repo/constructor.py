from swarm.environment.domain.mmlu.env import GymWrapper

class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, swarm, graph, train_dataset, val_dataset, train, num_pot_edges, num_nodes, num_node_features, node_features, state_indicator, node2idx, idx2node, edge_index, batch_size, num_envs):
        """
        A general Environment Constructor
        """
        self.swarm = swarm
        self.graph = graph
        self.is_discrete = True
        self.train = train
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
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

    def make_env(self, test=False, graph=None, node2idx=None, idx2node=None, node_features=None, state_indicator=None, edge_index=None, **kwargs):
        """
        Generate and return an env object
        """
        if graph is not None:
            self.graph = graph
            self.graph.current_node_id = self.graph.start_node_id
        
        if node2idx is not None:
            self.node2idx = node2idx
            
        if idx2node is not None:
            self.idx2node = idx2node
            
        if node_features is not None:
            self.node_features = node_features
            
        if state_indicator is not None:
            self.state_indicator = state_indicator
            
        if edge_index is not None:
            self.edge_index = edge_index
            
        env = GymWrapper(self.swarm, self.graph, self.train_dataset, self.val_dataset, self.train, test, self.num_pot_edges, self.num_nodes, self.num_node_features, self.node_features, self.state_indicator, self.node2idx, self.idx2node, self.edge_index, self.batch_size, self.num_envs)
        return env