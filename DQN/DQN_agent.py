import numpy as np
import random
import torch
import torch.nn as nn
import numpy as np
import random

from DQN_architecture import DuelingDQN_Network
import DQN_functions

class DQNAgent(object):
    """docstring for Duelling DQN Agent."""
    def __init__(
        self, state_dim, action_dim, num_shared_layers, shared_layer_size, num_value_layers, value_layer_size, num_adv_layers, adv_layer_size, device, discount, lr):
        super(DQNAgent, self).__init__()
        
        self.action_dim = action_dim
        self.device = device
        self.discount = discount
        self.loss = None
        
        # Networks
        # Create main and target neural networks.
        self.main_nn = DuelingDQN_Network(
            size_in=state_dim,
            num_actions=action_dim,
            num_shared_layers=num_shared_layers,
            shared_layer_size=shared_layer_size,
            num_value_layers=num_value_layers,
            value_layer_size=value_layer_size,
            num_adv_layers=num_adv_layers,
            adv_layer_size=adv_layer_size
        ).to(device)
        self.target_nn = DuelingDQN_Network(
            size_in=state_dim,
            num_actions=action_dim,
            num_shared_layers=num_shared_layers,
            shared_layer_size=shared_layer_size,
            num_value_layers=num_value_layers,
            value_layer_size=value_layer_size,
            num_adv_layers=num_adv_layers,
            adv_layer_size=adv_layer_size
        ).to(device)
        # standardise weights across both networks
        self.transfer_weights()
        
        # Loss function and optimizer.
        self.optimizer = torch.optim.Adam(self.main_nn.parameters(), lr=lr)
        # self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        print("Main and Target Network Architectures:")
        DQN_functions.model_summary(self.main_nn, (1,state_dim))

    def select_epsilon_greedy_action(self, state : torch.Tensor, epsilon : float):
        result = np.random.uniform()
        if result < epsilon:
            return  np.random.choice(self.action_dim) # Random action.
        else:
            qs = self.main_nn(state.to(self.device))
            return torch.argmax(qs).cpu().detach().item() # Greedy action for state.
        
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data."""
        q_values = self.main_nn(states.to(self.device))
        next_q_values = self.target_nn(next_states.to(self.device))
        
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0] #next_q_values.argmax(dim=-1, keepdim=True)
        expected_q_value = rewards + self.discount * next_q_value * (1 - dones)
   
        # masked_next_qs = self.target_nn(next_states.to(self.device)).gather(1, next_qs_argmax).squeeze() # gather : dim, index tensor
        # target = rewards * self.discount * masked_next_qs + (1.0 - dones)
        # self.loss = self.loss_fn(masked_qs, target.detach())
        
        self.loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss
       
    def transfer_weights(self):
        self.target_nn.load_state_dict(self.main_nn.state_dict())
       
    def save(self, current_game : int, checkpoint_file : str):
        DQN_functions.save_checkpoint(
            current_game, self.main_nn, self.optimizer, self.loss, checkpoint_file
        )
        
    def load_weights(self, checkpoint_dir : str):
        print(f'Loading agent network weights from: {checkpoint_dir}')
        checkpoint = DQN_functions.load_checkpoint(checkpoint_dir)
        self.main_nn.load_state_dict(checkpoint['model_state_dict'])
        self.target_nn.load_state_dict(checkpoint['model_state_dict'])
        
if __name__ == "__main__":
    pass