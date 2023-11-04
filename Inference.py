import os
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from encoder.encoder_architecture import VGG16Autoencoder
from encoder.encoder_functions import model_summary

encoder_checkpoint = "./encoder/encoder.checkpoint"
TD3_checkpoint = None

if __name__ == "__main__":
    # Load rgb array env to confirm DDQN and Encoder Models
    env = gym.make("ALE/BattleZone-v5", render_mode = "rgb_array")
    observation = env.reset()
    env.metadata['render_fps'] = 60
    # Check device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
        print(f"Device: {DEVICE}")
        
    print("Loading encoder")
    autoencoder = VGG16Autoencoder().to(DEVICE)
    if os.path.isfile(encoder_checkpoint):
        print("Loading Encoder Checkpoint")
        checkpoint = torch.load(encoder_checkpoint)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found")
    print("Encoder Loaded")
    model_summary(autoencoder, np.transpose(env.render(), (2, 0, 1)).shape)
    
    # Load TD3 Architecture
    
    env.close()