from swarm.environment.domain.mmlu.env import GymWrapper

class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, dataset, loader, num_pot_edges, num_nodes, num_node_features, batch_size):
        """
        A general Environment Constructor
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

    def make_env(self, **kwargs):
        """
        Generate and return an env object
        """
        env = GymWrapper(self.dataset, self.loader, self.num_pot_edges, self.num_nodes, self.num_node_features, self.batch_size)
        return env