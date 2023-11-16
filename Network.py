import math
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, (3, 3), padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, (3, 3), padding=1, stride=(2, 2))
        self.relu2 = nn.ReLU()
        self.skip = nn.Conv2d(in_channel, out_channel, (3, 3), padding=1, stride=(2, 2))

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        skip = self.skip(x)
        return (out + skip) / math.sqrt(2)


class PolicyNet(nn.Module):
    def __init__(self, action_dim=9):
        super().__init__()
        self.res1 = ResBlock(3, 8)
        self.res2 = ResBlock(8, 16)
        self.res3 = ResBlock(16, 32)
        self.ret4 = ResBlock(32, 64)
        self.ret5 = ResBlock(64, 108)  # [batch, 108, 34, 60], state_dim=220320
        mid = (220320 - action_dim) // 2
        self.fc1 = nn.Linear(220320, mid)
        self.fc2 = nn.Linear(mid, action_dim)

    def forward(self, state):
        batch, _, _, _ = state.shape
        temp = self.res5(self.res4(self.res3(self.res2(self.res1(state))))).view(batch, -1)
        temp = nn.functional.relu(self.fc1(temp))
        ret = nn.functional.softmax(self.fc2(temp), dtype=torch.double, dim=1)
        return ret


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = ResBlock(3, 8)
        self.res2 = ResBlock(8, 16)
        self.res3 = ResBlock(16, 32)
        self.ret4 = ResBlock(32, 64)
        self.ret5 = ResBlock(64, 108)
        mid = 220320 // 2
        self.fc1 = nn.Linear(220320, mid)
        self.fc2 = nn.Linear(mid, 1)

    def forward(self, state):
        batch, _, _, _ = state.shape
        temp = self.res5(self.res4(self.res3(self.res2(self.res1(state))))).view(batch, -1)
        temp = nn.functional.relu(self.fc1(temp))
        ret = self.fc2(temp)
        return ret


class PPO:
    def __init__(self, action_dim, device, lr=3e-4):
        self.actor = PolicyNet(action_dim).to(device)
        self.critic = ValueNet().to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr)

        self.device = device

    def select_action(self, state):
        state = torch.tensor(state).view(-1, 3, 1080, 1920).to(self.device)
        probs = self.actor(state)
        action_list = torch.distributions.Categorical(probs)
        action = action_list.sample().item()
        return action

    def optimize(self, transition_dict, gamma, lmbda, eps, epochs):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        # calculate TD
        next_q_target = self.critic(next_states)
        td_target = rewards + gamma * next_q_target * (1 - dones)
        td_value = self.critic(states)
        td_delta = td_target - td_value

        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []

        # calculate advantage function
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # GAE
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        avg_actor_loss = 0
        avg_critic_loss = 0

        for _ in range(epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            avg_actor_loss += actor_loss.item()
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(torch.nn.functional.mse_loss(self.critic(states), td_target.detach()))
            avg_critic_loss += critic_loss.item()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return avg_actor_loss / epochs, avg_critic_loss / epochs

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path)

    def load(self, path):
        print("load model:", path)
        ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
