import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gymnasium as gym
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# print(gym.envs.registry.keys())

import encoder_functions
from encoder_architecture import VGG16Autoencoder

ALE_ENV = True
environment = "Pong-v5"
CHECKPOINT_PATH = f'./encoder/{environment}/'
CHECKPOINT_NAME = "encoder.checkpoint"
TRAIN = True
VISUALISE = False
SAVE_GRAPH = False
BUFFER_SIZE = 1000
TRAINING_EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 5e-3

if torch.cuda.is_available():
  DEVICE = "cuda"
else:
  DEVICE = "cpu"
print(f"Device: {DEVICE}")

if ALE_ENV:
    env = gym.make(f'ALE/{environment}', render_mode = "rgb_array")
else:
    env = gym.make(f'{environment}', render_mode = "rgb_array")
observation = env.reset()

rgb_array = env.render().astype(np.float32) / 255.0 # convert to float and normalise to [0,1]
print(f'Image shape: {rgb_array.shape}\nTransposing for PyTorch')
input_shape = np.transpose(rgb_array, (2, 0, 1)).shape # transpose to account for pytorch model requirements
print(f'Input shape: {input_shape}')
if VISUALISE:
    encoder_functions.visualise_frame(rgb_array * 255)

# populate replay buffer
replay_buffer = encoder_functions.build_buffer(BUFFER_SIZE, env, input_shape)
if TRAIN:
    test_size = int(0.05 * BUFFER_SIZE)
else:
    test_size = int(0.95 * BUFFER_SIZE)
train_size = BUFFER_SIZE - test_size
train_array, test_array = replay_buffer[:train_size], replay_buffer[train_size:]
train_dataset = encoder_functions.AugmentedDataset(train_array)
test_dataset = TensorDataset(torch.from_numpy(test_array))
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# check data
if VISUALISE:
    image = next(iter(train_loader))[0].cpu().numpy()
    image_np_transpose = np.transpose(image, (1, 2, 0))
    encoder_functions.visualise_frame(image_np_transpose * 255)

# laod model
autoencoder = VGG16Autoencoder().to(DEVICE)
criterion = nn.MSELoss().to(DEVICE)
optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

if os.path.isfile(os.path.join(CHECKPOINT_PATH,CHECKPOINT_NAME)):
    print("Loading Encoder Checkpoint")
    checkpoint = torch.load(os.path.join(CHECKPOINT_PATH,CHECKPOINT_NAME))
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
else:
    print("No checkpoint found")
    os.mkdir(CHECKPOINT_PATH)
if SAVE_GRAPH:
    encoder_functions.save_graph(autoencoder, DEVICE)
encoder_functions.model_summary(autoencoder, input_shape)
    
# training loop
if TRAIN:
    print("Beginning training loop")
    best_avg_loss = 0
    for epoch in range(TRAINING_EPOCHS):
        losses = []
        
        for data in train_loader:
            img = data.to(DEVICE)
            output = autoencoder(img)
            loss = criterion(output, img)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        average_loss = sum(losses)/len(losses)
        print(f'Epoch [{epoch+1}/{TRAINING_EPOCHS}], Average Loss: {average_loss:.4f}')
        
        if epoch == 0:
            best_avg_loss = average_loss
            encoder_functions.save_checkpoint(epoch, autoencoder, optimizer, best_avg_loss, os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME))
        
        else:
            if average_loss < best_avg_loss:
                best_avg_loss = average_loss
                encoder_functions.save_checkpoint(epoch, autoencoder, optimizer, best_avg_loss, os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME))


print("Testing validation data")
with torch.no_grad():
    for i, data in enumerate(test_loader):
        images = data[0].to(DEVICE)
        # Run the images through the encoder
        encoded_imgs, indices, sizes = autoencoder.encode(images)

        # Run the encoded images through the decoder to get the reconstructed images
        decoded_imgs = autoencoder.decode(encoded_imgs, indices, sizes)
        
        print(f'Test {i +1} - Loss: {criterion(images, decoded_imgs):.6f}')
            
    
image_idx = 10
image = images[image_idx]
# Move the tensor to CPU (in case it's on GPU) and convert it to numpy
image_np = image.cpu().numpy()
# Transpose the image dimensions from CxHxW to HxWxC - multiply by 255
image_np_transpose = np.transpose(image_np, (1, 2, 0)) * 255
# Show the image
plt.axis("off")
plt.imshow(image_np_transpose.astype("int"))
plt.show()

image = decoded_imgs[image_idx]
# Move the tensor to CPU (in case it's on GPU) and convert it to numpy
image_np = image.cpu().detach().numpy()
# Transpose the image dimensions from CxHxW to HxWxC - multiply by 255
image_np_transpose = np.transpose(image_np, (1, 2, 0)) * 255
# Show the image
plt.axis("off")
plt.imshow(image_np_transpose.astype("int"))
plt.show()