import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

class logger(object):
    """docstring for logger."""
    def __init__(self, env):
        super(logger, self).__init__()

        # Get current date and time
        now = datetime.datetime.now()
        # Format as a string
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize TensorBoard writer
        log_dir = f'./DQN/logs/{env}/space_invaders_experiment_{dt_string}'
        self.writer = SummaryWriter(log_dir)
        print(f'Logging run results to {log_dir}')

    def log_hyperparameters(self, parameter_names_values):
        hp_table = [f"| {name} | {value} |" for name, value in parameter_names_values]
        table_str = "| Parameter | Value |\n|-----------|-------|\n" + "\n".join(hp_table)
        self.writer.add_text("Hyperparameters", table_str, 0)

    def write_to_tensorboard(self, epoch, ep_reward, epsilon, loss = None):
        """
        Writes metrics to TensorBoard.
        
        Parameters:
            writer (SummaryWriter): TensorBoard SummaryWriter object.
            epoch (int): Current epoch number.
            ep_reward (float): Total reward obtained in the episode.
            epsilon (float): Current epsilon value for epsilon-greedy action selection.
            loss (float): Loss value.
        """
        self.writer.add_scalar('Epoch Cumulative Reward', ep_reward, epoch)
        self.writer.add_scalar('Epsilon', epsilon, epoch)
        if loss is not None:
            self.writer.add_scalar('Epoch Cumulative Loss', loss, epoch)

def save_checkpoint(current_game, model, optimiser, loss, checkpoint_file):
    torch.save({
        'ganme' : current_game,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss (Huber)': loss,
    }, checkpoint_file)
    print(f'Saving model at game {current_game}')

def load_checkpoint(directory):
    if os.path.isfile(directory):
        print("Loading Encoder Checkpoint")
        checkpoint = torch.load(directory)
        return checkpoint
    else:
        pass

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
    
if __name__ == "__main__":
    pass