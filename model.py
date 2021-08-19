import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


# This network code from the udacity maddpg lab differs from the Actor and Critic code I had before mostly in naming the input and output dimensions
# as well as the hidden layers, and also because it uses a different range of numbers in the uniform distribution of the final layer
# and the big change other than that is it reuses the init code across both networks and only differs in the forward pass code with a conditional
# for actor and critic. I could probably update to use a composition based pattern or other OO design for reuse, but this is fine for now.

# other changes, bounding the action output of the actor to a normalized value between 0 and 10, maybe won't work for this assignment will test
# also not doing a just in time addition of the action space to the critic in the forward method, not sure why this is omitted here, tryign to 
# puzzle it out.
class Network(nn.Module):
    def __init__(
        self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False
    ):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = f.relu  # leaky_relu
        self.actor = actor
        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # return a vector of the force
            h1 = self.nonlin(self.fc1(x))

            h2 = self.nonlin(self.fc2(h1))
            h3 = self.fc3(h2)
            norm = torch.norm(h3)

            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            return 10.0 * (f.tanh(norm)) * h3 / norm if norm > 0 else 10 * h3

        else:
            # critic network simply outputs a number
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = self.fc3(h2)
            return h3
