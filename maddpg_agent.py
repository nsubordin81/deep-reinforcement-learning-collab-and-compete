import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(5e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0
INITIAL_NOISE = 1
NOISE_DECAY = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, decay_noise=False):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            # import ipdb

            # ipdb.set_trace()
            pre_numpy_action = self.actor_local(state)
            action = pre_numpy_action.cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise_sample = self.noise.sample()
            # with open("actions_log", "a") as file:
            #     file.write(f"\nstate: {state}")
            #     file.write(f"\npre-numpy action: {pre_numpy_action}")
            #     file.write(f"\naction: {action}")
            #     file.write(f"\nnoise: {noise_sample}")
            action += noise_sample
            # not decaying noise anymore in case the agent is getting stuck
            if decay_noise:
                self.noise.decay_noise()
        clipped_actions = np.clip(action, -1, 1)
        return clipped_actions

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class MADDPG:
    def __init__(self, discount_factor=0.99, tau=0.03):
        super(MADDPG, self).__init()

        # initialize agents (this will be your Agent Class)

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_local_actors(self):
        actors = [agent.actor for agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        target_actors = [agent.target_actor for agent in self.maddpg_agent]
        return target_actors

    def local_act(self, obs_all_agents, noise=0.0):
        actions = [
            agent.act(obs, noise)
            for agent, obs in zip(self.maddpg_agent, obs_all_agents)
        ]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        target_actions = [
            agent.target_act(obs, noise)
            for agent, obs in zip(self.maddpg_agent, obs_all_agents)
        ]
        return target_actions

    def update_local_actors_and_critics(self, samples, agent_number, logger):

        # ok so let's map over the samples that come to us from the replay buffer,
        # and let's map with the transpose function to make the values so the first index is the agent index
        # then I guess we unpack that tensor into a series of objects
        # and they are all arrays like this?? [[agent1obs1, agent1obs2, . . . ], [agent2obs1, agent2obs2, ...]]
        # so you can easily index out all of the observations, rewards, next_obs, dones for a given agent
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(
            transpose_to_tensor, samples
        )

        # vertically stack the m separate observations into a tensor of m observation rows by n param cols
        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)

        # get the agent we are updating from the list by agent index
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # do the critic loss, (y-Qtarget(s,a))^2
        # corresponds to the first part of the update in the MADDPG paper
        target_actions = self.target_act(next_obs)
        # torch.cat is a new one for me, this is along dim 1,
        # so thinking in 2d it will concat the tensors left to right, doubling the 'column space'
        # reviewing how maddpg gets target actions as a numpy array of each agent's actions,
        # this will combine the actions for all agents into one tensor whose rows are per obs and
        # columns include action values like [agent1action1, agent1action2, agent2action1, agent2action2]
        target_actions = torch.cat(target_actions, dim=1)
        # now combine those target actions with a transposed observation set, so we are back to agents as rows
        # with that setup we can easily join up the observations with the actions by agent going down the m dimension
        target_critic_input = torch.cat((next_obs_full.t(), target_actions), dim=1).to(
            device
        )

        # get a q value from a forward pass with the target network, but we aren't going to backpropogate
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        # this is the typical value estimate for q learning wrt this agent
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (
            1 - done[agent_number].view(-1, 1)
        )
        # we did this for target as well, not sure if the above description is correct but same is happening for target as
        # is for local.
        action = torch.cat(action, dim=1)
        # same thing as was done with the target, but we are backpropogating with this one.
        critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        # not gradient clipping here either, it wasn't done in the lab but looks like they thought about it
        # torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # actor update
        agent.actor_optimizer.zero_grad()

        # interesting step here, so we don't need to compute the gradient for the observations of other agents
        q_input = [
            self.maddpg_agent[i].actor(ob)
            if i == agent_number
            else self.maddpg_agent[i].actor(ob).detach()
            for i, ob in enumerate(obs)
        ]

        # just like we have been doing, concatenate along the columns for this matrix of actions
        q_input = torch.cat(q_input, dim=1)

        # combine along the columns here as well so the observations and inputs are combined, transposing
        # they mention that the observations are the same for the most part adn the first one has everything
        # that is interesting, was that specific to the environment? idk, but for an actor we have what each
        # actor did combined with the state they saw, so we should be able to find a better policy through
        # gradient ascent
        q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        # and here we do, negative mean of the Q value gets us the vector for how far and in what direction to travel
        # in the policy space
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        # again, this clipping is here, might try it if needed and see what the results look like.
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars(
            "agent%i/losses" % agent_number,
            {"critic loss": cl, "actor_loss": al},
            self.iter,
        )

    def update_targets(self):
        self.iter += 1
        for agent in self.maddpg_agent:
            soft_update(agent.target_actor, agent.local_actor, self.tau)
            soft_update(agent.target_critic, agent.local_critic, self.tau)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)
        self.decay = INITIAL_NOISE
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(
            self.size
        )
        self.state = x + dx
        return self.state * self.decay

    def decay_noise(self):
        self.decay = max(self.decay * NOISE_DECAY, 1e-2)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
