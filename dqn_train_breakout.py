# Breakout 训练脚本
# 环境
from gym_env import BreakoutEnv
env = BreakoutEnv()
state_dim = env.state_dim  # (80, 80)
action_dim = env.action_dim  # 4

# 算法
from dqn_agent import DQNAgent
conf = dict(
    action_dim = action_dim,
    epsilon_start = 1.0,
    epsilon_end = 0.1,
    epsilon_decay = 10000,  # Breakout 需要更长的探索期
    gamma = 0.99,
    device = 'mps'  # 可改为 'mps' (Mac M4) 或 'cuda'
)
agent = DQNAgent(conf)
agent.epsilon = conf["epsilon_start"]

# 模型 - QNetwork 会自动识别图像输入并使用 CNN
from q_network import QNetwork
model = QNetwork(state_dim, action_dim, lr = 1e-4)  # Breakout 用更小的学习率
agent.set_model(model)

from sample import FrameNumpy, SampleBatchNumpy
from collections import deque
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

# Breakout 训练参数（需要更大的 buffer 和更多 episodes）
buffer_size = 10000
batch_size = 32
episodes = 5000
train_returns = []
test_returns = []
replay_buffer = deque(maxlen = buffer_size)

# 开始训练前先填充一些随机样本
print("Filling replay buffer with random samples...")
for _ in range(batch_size * 10):
    obs = env.reset()
    done = False
    while not done and len(replay_buffer) < batch_size * 10:
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
    if len(replay_buffer) >= batch_size * 10:
        break

print(f"Starting training with {len(replay_buffer)} samples in buffer...")

for episode in tqdm(range(episodes)):
    ret = 0
    obs = env.reset()
    done = False
    step_count = 0
    
    while not done:
        action = agent.predict(obs)
        next_obs, reward, done, _ = env.step(action)
        ret += reward
        step_count += 1
        
        sample = FrameNumpy.from_dict({
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'done': done
        })
        replay_buffer.append(sample)
        obs = next_obs
        
        # 每个 step 采样并训练
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch = SampleBatchNumpy.stack(batch)
            agent.sample_process(batch)
            agent.learn(batch)
    
    train_returns.append((episode, ret))
    
    # 每 50 局测试一局
    if episode % 50 == 0:
        ret = 0
        obs = env.reset()
        done = False
        while not done:
            action = agent.exploit(obs)
            next_obs, reward, done, _ = env.step(action)
            ret += reward
            obs = next_obs
        test_returns.append((episode, ret))
        print(f"\nEpisode {episode}: Train={train_returns[-1][1]:.1f}, Test={ret:.1f}, Epsilon={agent.epsilon:.3f}")
    
    # epsilon decay
    decay = (conf["epsilon_start"] - conf["epsilon_end"]) / conf["epsilon_decay"]
    agent.epsilon = max(conf["epsilon_end"], agent.epsilon - decay)

# 保存模型
import os
os.makedirs('./results', exist_ok=True)
import torch
torch.save(model.state_dict(), './results/breakout_model.pth')

# 绘图
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
train_smooth = [np.mean([x[1] for x in train_returns[max(0, i - window_size + 1):i + 1]]) for i in range(len(train_returns))]
test_smooth = [np.mean([x[1] for x in test_returns[max(0, i - window_size + 1):i + 1]]) for i in range(len(test_returns))]

plt.figure(figsize=(10, 5))
plt.plot([x[0] for x in train_returns], train_smooth, label='train smoothed')
plt.plot([x[0] for x in test_returns], test_smooth, label='test smoothed', marker='o')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Return (smoothed)')
plt.title("Breakout DQN Smoothed")
plt.savefig('./results/breakout_smoothed.png', dpi=300)
plt.show()
