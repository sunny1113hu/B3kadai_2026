import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NoisyLinear


class Q_Net(nn.Module):
	def __init__(self, opt):
		super(Q_Net, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(4, 32, 8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1),
			nn.ReLU(),
			nn.Flatten())
		if opt.Noisy:
			self.fc1 = NoisyLinear(64 * 7 * 7, opt.fc_width)
			self.fc2 = NoisyLinear(opt.fc_width, opt.action_dim)
		else:
			self.fc1 = nn.Linear(64 * 7 * 7, opt.fc_width)
			self.fc2 = nn.Linear(opt.fc_width, opt.action_dim)

	def forward(self, obs):
		s = obs.float()/255 #convert to f32 and normalize before feeding to network
		s = self.conv(s)
		s = torch.relu(self.fc1(s))
		q = self.fc2(s)
		return q


class Duel_Q_Net(nn.Module):
	def __init__(self, opt):
		super(Duel_Q_Net, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(4, 32, 8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, stride=1),
			nn.ReLU(),
			nn.Flatten())
		if opt.Noisy:
			self.fc = NoisyLinear(64 * 7 * 7, opt.fc_width)
			self.A = NoisyLinear(opt.fc_width, opt.action_dim)
			self.V = NoisyLinear(opt.fc_width, 1)
		else:
			self.fc = nn.Linear(64 * 7 * 7, opt.fc_width)
			self.A = nn.Linear(opt.fc_width, opt.action_dim)
			self.V = nn.Linear(opt.fc_width, 1)

	def forward(self, obs):
		s = obs.float()/255 #convert to f32 and normalize before feeding to network
		s = self.conv(s)
		s = torch.relu(self.fc(s))
		Adv = self.A(s)
		V = self.V(s)
		Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
		return Q


class DeepQ_Agent(object):
	def __init__(self,opt):
		self.dvc = opt.dvc
		self.action_dim = opt.action_dim
		self.batch_size = opt.batch_size
		self.gamma = opt.gamma
		self.train_counter = 0
		self.huber_loss = opt.huber_loss
		self.Double = opt.Double
		self.Duel = opt.Duel
		self.Noisy = opt.Noisy
		self.PER = getattr(opt, "PER", False)

		if self.Duel: self.q_net = Duel_Q_Net(opt).to(self.dvc)
		else: self.q_net = Q_Net(opt).to(self.dvc)
		self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr_init)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.target_freq = opt.target_freq


	def select_action(self, state, evaluate):
		with torch.no_grad():
			state = state.unsqueeze(0).to(self.dvc)
			if self.Noisy: #NoisyNet时，不需要e-greedy
				return self.q_net(state).argmax().item()
			else:
				p = 0.01 if evaluate else self.exp_noise
				if np.random.rand() < p:
					return np.random.randint(0,self.action_dim)
				else:
					return self.q_net(state).argmax().item()


	def train(self, replay_buffer):
		self.train_counter += 1
		if self.PER:
			s, a, r, s_next, dw, ind, weights = replay_buffer.sample(self.batch_size)
		else:
			s, a, r, s_next, dw = replay_buffer.sample(self.batch_size)

		'''Compute the target Q value'''
		with torch.no_grad():
			if self.Double:
				argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
				max_q_prime = self.q_target(s_next).gather(1,argmax_a)
			else:
				max_q_prime = self.q_target(s_next).max(1)[0].unsqueeze(1)

			'''Avoid impacts caused by reaching max episode steps'''
			target_Q = r + (~dw) * self.gamma * max_q_prime #dw: die or win

		# Get current Q estimates
		current_q = self.q_net(s)
		current_q_a = current_q.gather(1,a)

		if self.huber_loss:
			loss_per_sample = F.huber_loss(current_q_a, target_Q, reduction='none')
		else:
			loss_per_sample = F.mse_loss(current_q_a, target_Q, reduction='none')
		if self.PER:
			q_loss = (loss_per_sample * weights).mean()
		else:
			q_loss = loss_per_sample.mean()

		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 20)
		self.q_net_optimizer.step()

		if self.PER:
			td_error = (target_Q - current_q_a).abs().squeeze(-1)
			replay_buffer.update_priorities(ind, td_error + replay_buffer.eps)

		# hard target update
		if self.train_counter % self.target_freq == 0:
			for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
				target_param.data.copy_(param.data)
		for p in self.q_target.parameters(): p.requires_grad = False


	def save(self,ExperimentName,index):
		torch.save(self.q_net.state_dict(), f"./model/{ExperimentName}_{index}k.pth")

	def load(self,ExperimentName,index):
		self.q_net.load_state_dict(torch.load(f"./model/{ExperimentName}_{index}k.pth", map_location=self.dvc))
		self.q_target.load_state_dict(torch.load(f"./model/{ExperimentName}_{index}k.pth", map_location=self.dvc))



class ReplayBuffer_torch():
	def __init__(self, device, max_size=int(1e5)):
		self.device = device
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8)
		self.action = torch.zeros((max_size, 1), dtype=torch.int64)
		self.reward = torch.zeros((max_size, 1))
		self.next_state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8)
		self.dw = torch.zeros((max_size, 1), dtype=torch.bool)

	def add(self, state, action, reward, next_state, dw):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw  # 0,0,0，...，1

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		# ind = np.random.choice((self.size-1), batch_size, replace=False)  # Time consuming, but no duplication
		ind = np.random.randint(0, (self.size-1), batch_size)  # Time effcient, might duplicates
		return self.state[ind].to(self.device),self.action[ind].to(self.device),self.reward[ind].to(self.device),\
			   self.next_state[ind].to(self.device),self.dw[ind].to(self.device)


class PrioritizedReplayBuffer_torch():
	def __init__(self, device, max_size=int(1e5), alpha=0.6, beta=0.4, eps=1e-6, replacement=False):
		self.device = device
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8)
		self.action = torch.zeros((max_size, 1), dtype=torch.int64)
		self.reward = torch.zeros((max_size, 1))
		self.next_state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8)
		self.dw = torch.zeros((max_size, 1), dtype=torch.bool)

		self.priorities = torch.zeros((max_size,), dtype=torch.float32)
		self.alpha = alpha
		self.beta = beta
		self.eps = eps
		self.replacement = replacement
		self.max_priority = 1.0

	def add(self, state, action, reward, next_state, dw):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw
		self.priorities[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		priorities = self.priorities[:self.size]
		if priorities.sum() <= 0:
			priorities = torch.ones_like(priorities)
		probs = priorities.pow(self.alpha)
		probs = probs / probs.sum()
		ind = torch.multinomial(probs, num_samples=batch_size, replacement=self.replacement)
		weights = (self.size * probs[ind]).pow(-self.beta)
		weights = (weights / weights.max()).unsqueeze(-1)

		return self.state[ind].to(self.device), self.action[ind].to(self.device), self.reward[ind].to(self.device), \
			   self.next_state[ind].to(self.device), self.dw[ind].to(self.device), ind, weights.to(self.device)

	def update_priorities(self, ind, priorities):
		priorities = priorities.detach().clamp(min=self.eps).cpu()
		self.priorities[ind] = priorities
		self.max_priority = max(self.max_priority, float(priorities.max()))


