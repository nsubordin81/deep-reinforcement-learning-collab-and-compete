# Deep Reinforcement Learning In a Continuous Control Environment

## Algorithm

 extending upon the DDPG code I used for the continuous control project 
 (originally based on the DDPG Bipedal Walker example in DRLND repo) and for the algorithm I chose to apply the approach recommended in the benchmark 
 implementation notes, that of altering the DDPG algorithm slightly for a multi-agent scenario that took advantage of the fact that both agents were essentially 
 learning the same problem of how to hit the ball to do a fully centralized multi-agent approach.

So the main change I made to DDPG was to allow both agent's observations to come in at once, but then still have just the one actor playing the role of both paddles. 
The actor receives both observations and then chooses actions for both paddles simultaneously. Experience tuples have both agent's actions and observations, next observations, rewards and done states, all being saved into memory at the same time. Then, during sampling all of these stored values get stacked so they are essentially independent and double the batch size as they are fed into the actor and critic for learning. This seems like a form of self play, as the algorithm is learning twice as fast by having the agent play the 
part of both paddles. It also seems like it would not work so well if it were to be tried in an adversarial context or other task in which the optimal policy didn't look the same for both agents. However, this strategy does eventually converge for me, so I'll take it for this round, and commit to some more study to better understand how it compares to 
other methods.

 As I used DDPG for the continuous control problem and DQN (the basis for DDPG but only handling discrete action spaces), my descriptions in the Report.md of those
 project repositories can be referenced for my prior coverage of these techniques
 [for DDPG](https://github.com/nsubordin81/deep-reinforcement-learning-p2-double-jointed-arms/blob/main/REPORT.md)
 [for DQN](https://github.com/nsubordin81/deep-reinforcement-learning-p1-bananas/blob/master/Report.md)
 and for this section I'll just discuss some of the modifications made to DDPG to account for self play among 2 agents. 


### Hyperparameters

number of training episodes - 3000
START_DECAY = 10000 # how many episodes to start noise decay after, so in other words, I'm not doing noise decay for this task
GOAL_SCORE = .5 # given by requirements
agents = 2 # given by environment

For my implementation I chose the following hyperparameters:
BUFFER_SIZE = int(5e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0
INITIAL_NOISE = 1 # multiplier for the OU noise, starting point
NOISE_DECAY = 1 # modifier for the noise multiplier, if it were less than 1 then we'd use it to decay, but we aren't decaying in the version I successfully trained with

OUnoise: 
theta=0.15
sigma=0.2

I didn't really change anything from my DDPG solution in the reacher environment other than to not decay the noise and the learning rates for actor and critic were made to match at 1e-4 whereas it had been smaller for the actor and larger for the critic in the Reacher task.


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

The architecture remained unchanged from DDPG. I had started on MADDPG and was using the MADDPG LAB code that did more reuse of these classes, but I ran into some snags taht I'll 
work to figure out on my own later.

# Results

The average score of the highest performing agent per episode surpassed 0.5 over 100 episodes at episode 1368

here is a log (the 30 is because this is an adapted reacher environment and I didn't make the goal score a target in the log statement)

![Epsiode Log For Solved Environment](collab_complete_solved_episodes.png?raw=true)

![Plot For Solved Environment](collab_compete_solved_rewards_plot.png?raw=true)


## Ideas for Further Research

I could next attempt to solve the soccer environment as it looks like it is more complex involving interactions amongst 4 agents that are paired as cooperative with each other
and adversarial with the other pair of agents. 

Another option I had considered for the Tennis task before noting from the instructions that I could train both actors symmetrically is that there is a multi-agent MADDPG algorithm 
that could fair well here. I honestly wanted to try this approach first but ran out of time, so I plan to pursue it outside the boundaries of the nanodegree program.

 As they mention in the MADDPG paper, independent agents, whether they are Q-Learning based or
policy method based, don't tend to perform as well when used in multi agent environemnts. For policy gradient methods, the existing issues with high variance due to value assignment over 
trajectories are exacerbated because reward assignment is further confused by unknown actions of other agents. Q-Learning agents have similar drawbacks since as they independently update their policies without each other's knowledge, the environment becomes non-stationary from the viewpoint of each agent. Agents won't know were some transition information  and rewards are coming from as other agents continually change their behavior
and so their own update estimates become a moving target. explicitly choosing an algorithm that is proven to generalize to both adversarial and cooperative multi-agent
use cases makes sense for this problem if I were to expect the parameters to change to require decentralized execution. Additionally, the action space for the Tennis environment is once again continuous like Reacher was, and so MADDPG being based on DDPG is set up well to handle this and that wouldn't change.

In MADDPG, the basis for the agents is DDPG, so you still have a powerful actor-critic algorithm guiding the construction of each agent's policies, and 
unlike the "Counterfactual Multi-Agent Policy Gradients" paper which proposed a similar algorithm, MADDPG gives each actor its own critic rather than sharing the critic across
all actors. However that could be an interesting approach to try to emulate as well.

The framework of learning that makes this algorithm successful for multi-agent scenarios is one they characterize as 'centralized training with decentralized
execution.' During training time the critics are made aware of the developing policies of the other agents in the environment, but once trained they are no 
longer given access to this information. My analogy for this is taking tests in school, you can use the textbook, internet or classroom discussion to access perspectives
of others while learning (I guess this is the critic seeing all the extra info when estimating the Q function), but you had better have formed your own by the time you take the test because you won't be able to lean on it then (actors using the policy that they've shifted to for each episode).

It ends up looking like a pretty elegant solution, because in practice it means that the Q-function learned by each critic during training is taking in the actions and observations
of all actors for an experience tuple, but then its output is just the action value estimate for the actor it is paired with. This ultimately has the effect of making the 
environment stationary again because the TD estimates that are improving the policies know about changes in all the policies over every step.

I didn't need MADDPG for this assignment because the Tennis environment was symmetrical and observations were independent, so a single policy function could be learned through self play by evaluating the actions of both 'agents' as contributing to a shared single agent's policy, with same ddpg agent acting and learning on behalf of both paddles using a shared replay buffer. However, MADDPG would have likely also worked in this case and in other environments for tasks that don't share these those constraints, being probably faster to learn since the critics would have all observations and actors would have just their own. Having 2 agent instances comprised of a total of 8 ANNs instead of the 4 I used for DDPG could exacerbate the inherent instability that DDPG already brings to the table though so at least I avoided that for now.

Another route to try could be PPO for multi-agent scenarios or mixing and matching DDPG with PPO or another policy method. The MADDPG paper does allude to this as an option, and this path might be a good one for exploring ground that is relatively fertile for improvements in the multi-agent space.
