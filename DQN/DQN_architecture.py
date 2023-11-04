import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN_Network(nn.Module):
    """DDQN Architecture
    """
    def __init__(self, size_in, num_actions, num_shared_layers, shared_layer_size, num_value_layers, value_layer_size, num_adv_layers, adv_layer_size):
        super(DuelingDQN_Network, self).__init__()
        self.size_in = size_in
        self.num_actions = num_actions
        self.num_shared_layers = num_shared_layers
        self.shared_layer_size = shared_layer_size
        self.num_value_layers = num_value_layers
        self.value_layer_size = value_layer_size
        self.num_adv_layers = num_adv_layers
        self.adv_layer_size = adv_layer_size

        # Input Layer
        self.input_layer = nn.Linear(self.size_in, self.shared_layer_size)
        
        # Shared Layer Blocks
        if self.num_shared_layers > 0: # the parameter hidden layers refers to additional layers
            self.shared_layers_list = nn.ModuleList()
            for _ in range(self.num_shared_layers):
                self.shared_layers_list.append(nn.Linear(self.shared_layer_size, self.shared_layer_size))
                
        # Sepereate into Advantage and value layer blocks
        
        # value block
        self.value_layers_list = nn.ModuleList()
        if self.num_value_layers > 0: # the parameter hidden layers refers to additional layers
            for i in range(self.num_value_layers):
                if i < 1:
                    self.value_layers_list.append(nn.Linear(self.shared_layer_size, self.value_layer_size))
                else:
                    self.value_layers_list.append(nn.Linear(self.value_layer_size, self.value_layer_size))
        if len(self.value_layers_list) > 0:
            self.value_layers_list.append(nn.Linear(self.value_layer_size, 1)) # always terminate value layer block with 1 value
        else:
            self.value_layers_list.append(nn.Linear(self.shared_layer_size, 1))
        
        # Advantage Block
        self.adv_layers_list = nn.ModuleList()
        if self.num_adv_layers > 0: # the parameter hidden layers refers to additional layers
            for i in range(self.num_adv_layers):
                if i < 1:
                    self.adv_layers_list.append(nn.Linear(self.shared_layer_size, self.adv_layer_size))
                else:
                    self.adv_layers_list.append(nn.Linear(self.adv_layer_size, self.adv_layer_size))
        if len(self.adv_layers_list) > 0:
            self.adv_layers_list.append(nn.Linear(self.adv_layer_size, self.num_actions)) # always terminate value layer block with 1 value
        else:
            self.adv_layers_list.append(nn.Linear(self.shared_layer_size, self.num_actions))
                    
    def forward(self, latent_state):
        # Input layer
        x = F.relu(self.input_layer(latent_state))
        
        # Shared layers
        if self.num_shared_layers > 0: 
            for layer in self.shared_layers_list:
                x = F.relu(layer(x))
        
        # Value block
        for i, layer in enumerate(self.value_layers_list):
            if i < 1:
                v = F.relu(layer(x))
            else:
                v = F.relu(layer(v))
                
        # Advantage block
        for i, layer in enumerate(self.adv_layers_list):
            if i < 1:
                adv = F.relu(layer(x))
            else:
                adv = F.relu(layer(adv))
                
        Q = v + adv - adv.mean() # dim=1, keepdim = True)
        return Q
        
if __name__ == "__main__":
    import DQN_functions        
    net = DuelingDQN_Network(
        size_in=10,
        num_actions=3,
        num_shared_layers=3,
        shared_layer_size=128,
        num_value_layers=3,
        value_layer_size=64,
        num_adv_layers=3,
        adv_layer_size=64
    ).to("cuda")
    torch.cuda.empty_cache()
    DQN_functions.model_summary(net, (1,10))
    for i in range(1000):
        print(net(torch.randn((64,10)).to("cuda")))