import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from replay import Transition

seed = 42
np.random.seed(seed)
#torch.manual_seed(seed)

BATCH_SIZE = 5
DISCOUNT = .2
classes = ("dirt", "wheat", "carrot", "potato", "beetroot")

#from torch.autograd import Variable
#import torch.nn.functional as F

#from the pytorch tutorial, it seems as that if we want to do a greedy approach to a problem, we let the neural network run throught the environment and save all the reward it gets for every action. Since the network will act random at first, it will help  us get a good pool of samples. Then we use those samples and train our network on it. For example, we'll give it some random state s and the label will be the r_j

class Net(nn.Module):

    def __init__(self, side_len, n_outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=2, stride = 1)

        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        self.fc1 = nn.Linear(in_features=24, out_features=4)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1,6*2*2)
        x = self.fc1(x)
        return nn.LogSoftmax()(x)

def train(net, memory):
    if len(memory) < BATCH_SIZE: 
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions)) #this allows us to grab columns of states/actions/state_primes/rewards


    state_batch = torch.cat(batch.state)    #cat stacks tensors on top of each other (note diff from torch.stack)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state).unsqueeze_(1)

    print("new_state pred: ", net(next_state_batch))
    print("new_state pred: ", net(next_state_batch).max(dim=1).values.data)

    target = reward_batch + DISCOUNT * net(next_state_batch).max(dim=1).values.data.numpy()

    state_action_values = net(state_batch)


def translate_farm(farm):
    cell_to_id = {"dirt":1, "wheat":2, "carrots":3, "potatoes":4, "beetroots":5}
    id_to_cell = {val:key for key,val in cell_to_id.items()}
    return [id_to_cell[ident] for ident in farm]
