from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset # build class off, of Dataset class
from torchvision import transforms # needed for augmentations
import torch.nn as nn
from torchviz import make_dot
import os
import matplotlib.pyplot as plt

def build_buffer(BUFFER_SIZE, env, input_shape):
    replay_buffer = np.empty((BUFFER_SIZE, *input_shape), dtype=np.float32)
    print(f'Buffer shape: {replay_buffer.shape}')
    # Collect buffer data
    pbar = tqdm(total=BUFFER_SIZE)
    idx = 0
    while idx < BUFFER_SIZE:
        observation = env.reset()
        done = False
        while not done and idx < BUFFER_SIZE:
            rgb_array = env.render().astype(np.float32) / 255.0
            
            # Take random action
            action = env.action_space.sample()

            # Step through environment
            observation, reward, done, _, info = env.step(action)
            
            # Add after checking if done
            rgb_array_t = np.transpose(rgb_array, (2, 0, 1))
            replay_buffer[idx] = rgb_array_t
            idx += 1
            
            pbar.update(1)

    # Close environment
    env.close()
    print(replay_buffer.shape)
    print(type(replay_buffer[0][0][0][0]))
    del pbar
    
    return replay_buffer

class AugmentedDataset(Dataset):
    def __init__(self, numpy_array):
        self.data = torch.from_numpy(numpy_array)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.35, 0.35]),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ], p=0.5),
            transforms.ToTensor(),
        ])
        
    def __getitem__(self, index):
        x = self.data[index]
        x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)
    
def model_summary(model, input_size):
    print("----------------------------------------------------------------")
    print(f"Input Shape:               {str(input_size).ljust(25)}")
    print("----------------------------------------------------------------")
    print("Layer (type)               Output Shape         Param #")
    print("================================================================")
    
    total_params = 0

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params

            # Remove torch.Size
            if isinstance(output, tuple):
                output_shape = [str(list(o.shape)) if torch.is_tensor(o) else str(type(o)) for o in output]
                # Pick first size if there are multiple identical sizes in the tuple
                output_shape = output_shape[0]
            else:
                output_shape = str(list(output.shape))

            if len(list(module.named_children())) == 0:  # Only print leaf nodes
                print(f"{module.__class__.__name__.ljust(25)}  {output_shape.ljust(25)} {f'{num_params:,}'}")

        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)

    print("----------------------------------------------------------------")
    DEVICE = next(model.parameters()).device
    output = model(torch.randn(1, *input_size).to(DEVICE))

    for h in hooks:
        h.remove()

    output_shape = str(list(output.shape)) if torch.is_tensor(output) else str(type(output))
    print("----------------------------------------------------------------")
    print(f"Total params: {total_params:,}")
    print(f"Output Shape: {output_shape.ljust(25)}")
    print("----------------------------------------------------------------")
    
def save_graph(model, DEVICE, output_directory = "./"):
    x = torch.randn(1, 3, 210, 160).to(DEVICE).requires_grad_(True)
    y = model(x)

    dot = make_dot(y, params=dict(list(model.named_parameters()) + [('input', x)]))
    dot.format = 'png'
    dot.render(filename='network')
    dot.save(filename=os.path.join(output_directory, "encoder.png"))
    
def save_checkpoint(epoch, model, optimiser, loss, output_file):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'best_val_loss': loss,
    }, output_file)
    print(f'Saving model at Epoch {epoch+1}')
    
def visualise_frame(rgb_array):
    plt.axis('off')
    plt.imshow((rgb_array).astype("int")) # return back to [0,255] range and convert to int
    plt.show()