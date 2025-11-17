import gym
env = gym.make('CartPole-v1')
# 查看环境信息
print('#########################################')
print('env.observation_space: ', env.observation_space)
print('#########################################')
print('env.action_space: ', env.action_space)
print('#########################################')
import torch
# 查看PyTorch的设备支持情况
print('cuda: ', torch.cuda.is_available())  # 是否支持 CUDA
print('mps available: ', torch.backends.mps.is_available())  # 是否支持 MPS
print('mps built: ', torch.backends.mps.is_built())      # 是否编译时包含 MPS