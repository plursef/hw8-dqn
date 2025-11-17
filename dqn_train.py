# 环境
from gym_env import GymEnv
env = GymEnv('CartPole-v1')
state_dim = env.state_dim[0]
action_dim = env.action_dim

# 算法
from dqn_agent import DQNAgent
conf = dict(
    action_dim = action_dim,
    epsilon_start = 1.0,
    epsilon_end = 0.02,
    epsilon_decay = 300,
    gamma = 0.99,
    device = 'cpu'
)
agent = DQNAgent(conf)
agent.epsilon = conf["epsilon_start"]

# 模型
from q_network import QNetwork
model = QNetwork(state_dim, action_dim, lr = 3e-4)
agent.set_model(model)

from sample import FrameNumpy, SampleBatchNumpy
from collections import deque
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

buffer_size = 2000
batch_size = 64
episodes = 1200
train_returns = []
test_returns = []
replay_buffer = deque(maxlen = buffer_size) # 样本池
for episode in tqdm(range(episodes)):
    ret = 0
    obs = env.reset()
    done = False
    while not done:
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
        obs = next_obs
        # 每个step产生的样本加入样本池，并直接采样batch进行单次训练
        replay_buffer.append(sample)
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch = SampleBatchNumpy.stack(batch)
            agent.sample_process(batch)
            agent.learn(batch)
    train_returns.append((episode, ret))

    # 每 10 局测试一局
    if episode % 10 == 0:
        ret = 0
        obs = env.reset()
        done = False
        while not done:
            action = agent.exploit(obs)
            next_obs, reward, done, _ = env.step(action)
            ret += reward
            obs = next_obs
        test_returns.append((episode, ret))

    # ---------- epsilon decay ----------
    decay = (conf["epsilon_start"] - conf["epsilon_end"]) / conf["epsilon_decay"]
    agent.epsilon = max(conf["epsilon_end"], agent.epsilon - decay)
    # -----------------------------------

# 绘图
plt.plot([x[0] for x in train_returns], [x[1] for x in train_returns], label='train')
plt.plot([x[0] for x in test_returns], [x[1] for x in test_returns], label='test')
plt.legend()
plt.title("CartPole DQN")
plt.savefig('./results/cartpole.png', dpi=300)
plt.show()

# 平滑版
window_size = 10
train_smooth = [np.mean([x[1] for x in train_returns[max(0, i - window_size + 1):i + 1]]) for i in range(len(train_returns))]
test_smooth = [np.mean([x[1] for x in test_returns[max(0, i - window_size + 1):i + 1]]) for i in range(len(test_returns))]
plt.plot([x[0] for x in train_returns], train_smooth, label = 'train smoothed')
plt.plot([x[0] for x in test_returns], test_smooth, label = 'test smoothed')
plt.legend()
plt.title("CartPole Dueling DQN Smoothed")
plt.savefig('./results/dqn_cartpole_dueling_smoothed.png', dpi = 300)
plt.show()