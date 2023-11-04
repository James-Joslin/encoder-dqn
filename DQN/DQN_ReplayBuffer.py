import numpy as np
import random
import torch

class ReplayBuffer(object):
    def __init__(self, size):
        """
        OpenAI Replay Buffer example
        Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """Add observation and results.

        Parameters
        ----------
        obs_t: vector
            observation within which an action was taken
        action: int
            action taken
        reward: float
            reward of action taken
        obs_tp1: vector
            resulting observation from the action taken
        done: bool
            termination or completed (1) or ongoing (0)
        
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, device):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            
        return torch.as_tensor(np.array(obses_t)).to(device), torch.as_tensor(np.array(actions)).to(device), torch.as_tensor(np.array(rewards, dtype=np.float32)).to(device), torch.as_tensor(np.array(obses_tp1)).to(device), torch.as_tensor(np.array(dones, dtype=np.float32)).to(device)

    def sample(self, batch_size, device):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes, device)

if __name__ == "__main__":
    pass