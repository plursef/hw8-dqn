# DQN 作业 - Breakout 训练脚本（加速版 + CUDA 友好）

import os
import random
from collections import deque

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch

# 环境
from gym_env import BreakoutEnv
env = BreakoutEnv()
state_dim = env.state_dim  # (80, 80)
action_dim = env.action_dim  # 一般是 4

# 自动选择设备：优先 cuda，其次 mps，最后 cpu
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# 算法
from dqn_agent import DQNAgent
conf = dict(
    action_dim=action_dim,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=10000,   # Breakout 探索期长一些
    gamma=0.99,
    device=device
)
agent = DQNAgent(conf)
agent.epsilon = conf["epsilon_start"]

# 模型 - QNetwork 会自动识别图像输入并使用 CNN
from q_network import QNetwork
model = QNetwork(state_dim, action_dim, lr=1e-4)  # Breakout 用更小的学习率
agent.set_model(model)

from sample import FrameNumpy, SampleBatchNumpy

# 训练超参数（这里稍微优化了一下）
buffer_size = 10000
batch_size = 64          # 略大一点，GPU 更吃得饱
episodes = 5000
learn_freq = 4           # ⭐ 每 4 步更新一次网络（非常关键的加速点）

train_returns = []
test_returns = []
replay_buffer = deque(maxlen=buffer_size)

# 先预填充一些随机样本，避免一开始训练太抖
print("Filling replay buffer with random samples...")
target_prefill = batch_size * 10
while len(replay_buffer) < target_prefill:
    obs = env.reset()
    done = False
    while not done and len(replay_buffer) < target_prefill:
        action = np.random.randint(action_dim)
        next_obs, reward, done, _ = env.step(action)
        sample = FrameNumpy.from_dict({
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'done': done
        })
        replay_buffer.append(sample)
        obs = next_obs

print(f"Starting training with {len(replay_buffer)} samples in buffer...")

global_step = 0  # 用于控制 learn 频率

for episode in tqdm(range(episodes)):
    ret = 0.0
    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        global_step += 1
        step_count += 1

        # 训练阶段：epsilon-greedy
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        ret += reward

        sample = FrameNumpy.from_dict({
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'done': done
        })
        replay_buffer.append(sample)
        obs = next_obs

        # ⭐ 关键：不是每一步都学习，而是每 learn_freq 步学习一次
        if (global_step % learn_freq == 0) and (len(replay_buffer) > batch_size):
            batch = random.sample(replay_buffer, batch_size)
            batch = SampleBatchNumpy.stack(batch)
            agent.sample_process(batch)
            agent.learn(batch)

    train_returns.append((episode, ret))

    # 每 50 局测试一次
    if episode % 50 == 0:
        test_ret = 0.0
        obs = env.reset()
        done = False
        with torch.no_grad():
            while not done:
                action = agent.exploit(obs)
                next_obs, reward, done, _ = env.step(action)
                test_ret += reward
                obs = next_obs

        test_returns.append((episode, test_ret))
        print(
            f"\nEpisode {episode}: "
            f"TrainReturn={train_returns[-1][1]:.1f}, "
            f"TestReturn={test_ret:.1f}, "
            f"Epsilon={agent.epsilon:.3f}"
        )

    # epsilon decay
    decay = (conf["epsilon_start"] - conf["epsilon_end"]) / conf["epsilon_decay"]
    agent.epsilon = max(conf["epsilon_end"], agent.epsilon - decay)

# 保存模型
os.makedirs('./results', exist_ok=True)
torch.save(model.state_dict(), './results/breakout_model.pth')

# 绘图：原始曲线
plt.figure(figsize=(10, 5))
plt.plot([x[0] for x in train_returns], [x[1] for x in train_returns], label='train', alpha=0.3)
plt.plot([x[0] for x in test_returns], [x[1] for x in test_returns], label='test', marker='o')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title("Breakout DQN")
plt.savefig('./results/breakout.png', dpi=300)
plt.show()

# 平滑版
window_size = 50
train_smooth = [
    np.mean([x[1] for x in train_returns[max(0, i - window_size + 1):i + 1]])
    for i in range(len(train_returns))
]
test_smooth = [
    np.mean([x[1] for x in test_returns[max(0, i - window_size + 1):i + 1]])
    for i in range(len(test_returns))
]

plt.figure(figsize=(10, 5))
plt.plot([x[0] for x in train_returns], train_smooth, label='train smoothed')
plt.plot([x[0] for x in test_returns], test_smooth, label='test smoothed', marker='o')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Return (smoothed)')
plt.title("Breakout DQN Smoothed")
plt.savefig('./results/breakout_smoothed.png', dpi=300)
plt.show()
