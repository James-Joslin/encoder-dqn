# Requires pre-trained encoder
import torch
import gymnasium as gym
import os
import sys
import numpy as np

encoder_path = str(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'encoder/')))
sys.path.insert(1, encoder_path)
import encoder_functions
from encoder_architecture import VGG16Autoencoder

from DQN_agent import DQNAgent
from DQN_ReplayBuffer import ReplayBuffer
import DQN_functions


ALE_ENV = True
environment = "Pong-v5"

DDQN_CHECKPOINT_PATH = f'./DQN/{environment}/'
DDQN_CHECKPOINT_NAME = "dueling_ddqn.checkpoint"

ENCODER_CHECKPOINT = f'./encoder/{environment}/encoder.checkpoint'

# Hyperparameters
# training specific parameters
num_frames = 2000000
learning_rate = 1e-4
epsilon = 1.0
final_epsilon = 0.01
epsilon_decay = 1.1e-6
batch_size = 128
discount = 0.997
buffer_size = 200000
save_and_transfer_game_interval = 10
# network specific parameters
state_dim = 10
num_shared_layers=0
shared_layer_size=128
num_value_layers=1
value_layer_size=128
num_adv_layers=1
adv_layer_size=128

if __name__ == "__main__":
    # Load rgb array env to confirm Encoder Models
    
    if ALE_ENV:
        env = gym.make(f'ALE/{environment}', render_mode = "rgb_array")
    else:
        env = gym.make(f'{environment}', render_mode = "rgb_array")
    observation = env.reset()[0]
    # env.metadata['render_fps'] = 60
    # Check device
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
    print(f"Device: {DEVICE}")
        
    print("Loading encoder")
    autoencoder = VGG16Autoencoder().to(DEVICE)
    if os.path.isfile(ENCODER_CHECKPOINT):
        print("Loading Encoder Checkpoint")
        checkpoint = torch.load(ENCODER_CHECKPOINT)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found")
    print("Encoder Loaded")
    encoder_functions.model_summary(autoencoder, np.transpose(observation, (2, 0, 1)).shape)
    
    if not os.path.isfile(os.path.join(DDQN_CHECKPOINT_PATH,DDQN_CHECKPOINT_NAME)):
        os.mkdir(DDQN_CHECKPOINT_PATH)
        
    # Initialise Duelling DQN Agent
    print("Initialising agent and agent networks")
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=env.action_space.n,
        num_shared_layers=num_adv_layers,
        shared_layer_size=shared_layer_size,
        num_value_layers=num_value_layers,
        value_layer_size=value_layer_size,
        num_adv_layers=num_adv_layers,
        adv_layer_size=adv_layer_size,
        device=DEVICE,
        discount=discount,
        lr = learning_rate
    )
    print("Initialising Buffer")
    buffer = ReplayBuffer(buffer_size)
    print("Buffer initialised")
    
    # DDQN training
    writer = DQN_functions.logger(env = environment)
    writer.log_hyperparameters(
        [
            ("Number of sampling frames", num_frames), 
            ("Initial epsilon", epsilon), 
            ("Epsilon_decay", epsilon_decay),
            ('Final epsilon', final_epsilon), 
            ("Batch size", batch_size), 
            ("Discount", discount), 
            ("Buffer Size (frames)", buffer_size), 
            ("Save and transfer interval (Games)", save_and_transfer_game_interval), 
            ("Latent state dimensionality", state_dim), 
            ("Num shared layers", num_shared_layers), 
            ("Shared layer size", shared_layer_size), 
            ("Num value layers", num_value_layers), 
            ("Value layer size", value_layer_size), 
            ("Num advantage layers", num_adv_layers), 
            ("Advantage layer size", adv_layer_size),
            ("Learning rate", learning_rate)
        ]
    )
    frame = 0
    games_played = 0
    while frame < num_frames:
        game_reward = 0
        game_loss = 0
        done = False
        truncated = False
        observation = env.reset()[0] # observation on reset is a tuple of rgb, "ram" and greyscale - pick the first fo rgb
        
        # plot observation
        # encoder_functions.visualise_frame(observation)
        
        # prepare for encoding
        observation = torch.Tensor(np.transpose(observation.astype(np.float32) / 255, (2,0,1))).unsqueeze(0)
        
        # encode
        z, _, _ = autoencoder.encode(observation.to(DEVICE))
        # print([value for value in z.cpu().detach().numpy().tolist()[0]])
        while not done:
            # env.render()
            
            # select action
            action = agent.select_epsilon_greedy_action(state = z, epsilon=epsilon)

            # utilise action
            next_observation, reward, done, truncated, info = env.step(action)
            
            # encode next step for recording to buffer
            next_observation = torch.Tensor(np.transpose(next_observation.astype(np.float32) / 255, (2,0,1))).unsqueeze(0)
            next_z, _, _ = autoencoder.encode(next_observation.to(DEVICE))

            # tally reward
            game_reward += reward
            # reward = np.sign(reward)
            
            # Save to buffer.
            buffer.add(z.cpu().detach().data.numpy().flatten(), action, reward, next_z.cpu().detach().data.numpy().flatten(), done)
            z = next_z
            
            frame += 1
        
            if epsilon > final_epsilon:
                epsilon -= epsilon_decay
                
        games_played += 1
           
        if games_played % save_and_transfer_game_interval == 0:
            agent.transfer_weights()
            agent.save(current_game=games_played, checkpoint_file=os.path.join(DDQN_CHECKPOINT_PATH, DDQN_CHECKPOINT_NAME))
        
            
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size, DEVICE)
            loss = agent.train_step(states, actions, rewards, next_states, dones)
            game_loss += loss.cpu().detach().item()
            
            # # decode
            # image_hat = autoencoder.decode(z, indices, sizes)[0]
            
            # # plot decoded observation
            # image_hat = image_hat.cpu().detach().numpy()
            # image_hat = np.transpose(image_hat, (1, 2, 0)) * 255
            # encoder_functions.visualise_frame(image_hat)
            
            # print(env.action_space)
            
        writer.write_to_tensorboard(games_played, game_reward, epsilon, game_loss) # .cpu().detach().item())
        print(f'Last test reward: {game_reward}. Training frame: {frame+1}/{num_frames}. Loss: {game_loss:.2f}. Games played: {games_played}')

    writer.writer.close()
    env.close()