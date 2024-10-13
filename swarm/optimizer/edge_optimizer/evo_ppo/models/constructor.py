import torch

class ModelConstructor:

    def __init__(self, state_dim, action_dim, hidden_size, potential_connections, actor_seed=None, critic_seed=None):
        """
        A general Environment Constructor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.potential_connections = potential_connections
        self.actor_seed = actor_seed
        self.critic_seed = critic_seed


    def make_model(self, type, seed=False):
        """
        Generate and return an model object
        """

        if type == 'Gaussian_FF':
            from swarm.optimizer.edge_optimizer.evo_ppo.models.continous_models import Gaussian_FF
            model = Gaussian_FF(self.state_dim, self.action_dim, self.hidden_size)
            if seed:
                model.load_state_dict(torch.load(self.critic_seed))
                print('Critic seeded from', self.critic_seed)


        elif type == 'Tri_Head_Q':
            from swarm.optimizer.edge_optimizer.evo_ppo.models.continous_models import Tri_Head_Q
            model = Tri_Head_Q(self.state_dim, self.action_dim, self.hidden_size)
            if seed:
                model.load_state_dict(torch.load(self.critic_seed))
                print('Critic seeded from', self.critic_seed)

        elif type == 'GumbelPolicy':
            from swarm.optimizer.edge_optimizer.evo_ppo.models.discrete_models import GumbelPolicy
            model = GumbelPolicy(self.state_dim, self.action_dim, self.hidden_size)

        elif type == 'CategoricalPolicy':
            from swarm.optimizer.edge_optimizer.evo_ppo.models.discrete_models import CategoricalGATPolicy
            model = CategoricalGATPolicy(self.state_dim, self.action_dim, self.hidden_size, self.potential_connections)


        else:
            AssertionError('Unknown model type')


        return model



