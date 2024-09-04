import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from algos.DRL_VO.utils import preprocess_obs, soft_update_params
from algos.DRL_VO.encoder import make_encoder


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

################################## PPO Policy ##################################
class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, digit_dim, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_goal_state_dim + digit_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.apply(weight_init)

    def forward(self, obs, robot_goal_state, detach_encoder=True):
        obs = self.encoder(obs, detach=detach_encoder)
        obs = torch.cat((obs, robot_goal_state), dim=-1)
        mu, sigma = self.trunk(obs).chunk(2, dim=-1)
        mu = 2.0 * F.tanh(mu)
        sigma = F.softplus(sigma)
        return (mu, sigma)

class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, digit_dim, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_goal_state_dim + digit_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weight_init)

    def forward(self, obs, robot_goal_state, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        obs = torch.cat((obs, robot_goal_state), dim=-1)
        return self.trunk(obs)

class DRLVoAgent(object):
    """PPO+Encoder algorithm."""
    def __init__(
        self,
        obs_shape,
        robot_goal_state_dim, 
        digit_dim,
        action_shape,
        action_range, 
        device,
        batch_size=32,
        buffer_capacity=1000,
        hidden_dim=256,
        discount=0.99,
        actor_lr=1e-3,
        critic_lr=1e-3,
        k_epochs=20,
        max_grad_norm=0.5,
        eps_clip=0.2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        num_layers=4,
        num_filters=32
    ):
        self.action_shape = action_shape
        self.device = device
        self.discount = discount
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.max_grad_norm = max_grad_norm
        # action rescaling
        self.action_scale = torch.FloatTensor((action_range[1] - action_range[0]) / 2.).to(device)
        self.action_bias = torch.FloatTensor((action_range[1] + action_range[0]) / 2.).to(device)
        
        self.counter = 0
        self.buffer = []
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.actor = Actor(
            obs_shape, robot_goal_state_dim, digit_dim, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, robot_goal_state_dim, digit_dim, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # optimizers
        self.MseLoss = nn.MSELoss()
        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def select_action(self, obs, robot_goal_state, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            robot_goal_state = torch.FloatTensor(robot_goal_state).to(self.device)
            robot_goal_state = robot_goal_state.unsqueeze(0)
            (mu, sigma) = self.actor(obs, robot_goal_state)
        if eval:
            action = mu
            action_log_prob = 0.0
        else:
            dist = Normal(mu, sigma)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action_log_prob = action_log_prob.cpu().data.numpy().flatten()
        action = action.clamp(-1.0, 1.0)
        action = action * self.action_scale + self.action_bias

        return action.cpu().data.numpy().flatten(), action_log_prob
    

    def update(self, writer, step):
        obs = torch.tensor(np.array([t.obs for t in self.buffer]), dtype=torch.float32).to(self.device)
        robot_goal_state = torch.tensor(np.array([t.robot_goal_state for t in self.buffer]), dtype=torch.float32).to(self.device)
        a = torch.tensor(np.array([t.a for t in self.buffer]), dtype=torch.float32).view(-1, 1).to(self.device)
        r = torch.tensor(np.array([t.r for t in self.buffer]), dtype=torch.float32).view(-1, 1).to(self.device)
        not_done = torch.tensor(np.array([t.not_done for t in self.buffer]), dtype=torch.float32).view(-1, 1).to(self.device)
        obs_ = torch.tensor(np.array([t.obs_ for t in self.buffer]), dtype=torch.float32).to(self.device)
        robot_goal_state_ = torch.tensor(np.array([t.robot_goal_state_ for t in self.buffer]), dtype=torch.float32).to(self.device)
        old_action_log_probs = torch.tensor(np.array([t.a_log_p for t in self.buffer]), dtype=torch.float).view(-1, 1).to(self.device)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + not_done * self.discount * self.critic(obs_, robot_goal_state_)

        adv = (target_v - self.critic(obs, robot_goal_state)).detach()

        for _ in range(self.k_epochs):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma) = self.actor(obs[index], robot_goal_state[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip,
                                    1.0 + self.eps_clip) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic(obs[index], robot_goal_state[index]), target_v[index])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        writer.add_scalar('train_critic/action_loss', action_loss, step)
        writer.add_scalar('train_critic/value_loss', value_loss, step)
        del self.buffer[:]

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.critic.state_dict(), filename + '_critic')
       

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.critic.load_state_dict(torch.load(filename + '_critic'))
            