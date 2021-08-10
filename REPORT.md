# Deep Reinforcement Learning In a Continuous Control Environment

## Algorithm

 extending upon the DDPG code I used for the continuous control project 
 (originally based on the DDPG Bipedal Walker example in DRLND repo) and for the algorithm I chose to apply the approach recommended in the benchmark 
 impolementation notes, that of alterign the DDPG algorithm for self-play. 

 As I used DDPG for the continuous control problem and DQN (the basis for DDPG but only handling discrete action spaces), my descriptions in the Report.md of those
 project repositories should suffice as a good overview of DDPG, 
 [for DDPG](https://github.com/nsubordin81/deep-reinforcement-learning-p2-double-jointed-arms/blob/main/REPORT.md)
 [for DQN](https://github.com/nsubordin81/deep-reinforcement-learning-p1-bananas/blob/master/Report.md)
 and for this section I'll just discuss some of the modifications made to DDPG to account for self play among 2 agents. 




### Hyperparameters

For my implementation I chose the following hyperparameters:
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-5  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic
WEIGHT_DECAY = 0

OUnoise: 
theta=0.15
sigma=0.2


### Model Architecture

I have a target and local network for both actor and critic, and the architectures are as follows: 

#### actor 
input = 33
hidden1 = 256, activation ReLU, fan-in Xavier Initialization
hidden2 = 128, activation ReLU, fan-in Xavier Initialization
output = 4, activation tanh, (-.003, .003) uniform random Initialization

#### critic
input = 33
hidden1 = 256 + 4, activation ReLU, fan-in Xavier Initialization
hidden2 = 128, activation ReLU, fan-in Xavier Initialization
output = 1, (-.003, .003) uniform random initialization


## Results

## Ideas for Further Research

I could next attempt to solve the soccer environment as it looks like it is more complex involving interactions amongst 4 agents that are paired as cooperative with each other
and adversarial with the other pair of agents. 

Another option I had considered for the Tennis task before noting from the instructions that I could train all actors with one network is that there is a multi-agent MADDPG algorithm 
that could fair well here. As they mention in the MADDPG paper, independent agents, whether they are Q-Learning based or
policy method based, don't perform as well when used in multi agent environemnts. For policy gradient methods, the existing issues with high variance due to value assignment over 
trajectories are exacerbated because reward assignment is further confused by unknown actions of other agents. Q-Learning agents have similar drawbacks since as they independently update their policies without each other's knowledge, the environment becomes non-stationary from the viewpoint of each agent. Agents won't know were some transition information  and rewards are coming from as other agents continually change their behavior
and so their own update estimates become a moving target. explicitly choosing an algorithm that is proven to generalize to both adversarial and cooperative multi-agent
use cases made the most sense. Additionally, the action space for the Tennis environment is once again continuous like Reacher was, and so MADDPG being based on DDPG is set up well to handle this. 

In MADDPG, the basis for the agents is DDPG, so you still have a powerful actor-critic algorithm guiding the construction of each agent's policies, and 
unlike the "Counterfactual Multi-Agent Policy Gradients" paper which proposed a similar algorithm, MADDPG gives each actor its own critic rather than sharing it across
all of them..

The framework of learning that makes this algorithm successful for multi-agent scenarios is one they characterize as 'centralized training with decentralized
execution.' During training time the critics are made aware of the developing policies of the other agents in the environment, but once trained they are no 
longer given access to this information. My analogy for this is taking tests in school, you can use the textbook, internet or classroom discussion to access perspectives
of others while learning, but you had better have formed your own by the time you take the test because you won't be able to lean on it then.

It ends up looking like a pretty elegant solution, because in practice it means that the Q-function learned by each critic during training is taking in the actions and observations
of all actors for an experience tuple, but then its output is just the action value estimate for the actor it is paired with. This ultimately has the effect of making the 
environment stationary again because the TD estimates that are improving the policies know about changes in all the policies over every step.

I didn't need MADDPG for this assignment because the Tennis environment was symmetrical and observations were independent, so a single policy function could be learned through self play by evaluating the actions of both agents as contributing to a shared policy, with the same
function approximator (ANN) and replay buffer for DDPG used for each actor. However, MADDPG would have likely also worked in this case and in other environments for tasks that don't share these properties. 
