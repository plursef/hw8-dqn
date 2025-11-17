# DQN 作业 - Breakout 适配说明

## 修改内容

### 1. `q_network.py` - 自动适配图像输入
- **主要改动**：`QNetwork` 现在可以自动识别输入类型
  - 向量输入（如 CartPole）：使用 MLP 结构
  - 图像输入（如 Breakout）：使用 CNN 结构（3层卷积 + 2层全连接）
- **CNN 架构**：
  ```
  Conv2d(1, 32, 8x8, stride=4) -> ReLU
  Conv2d(32, 64, 4x4, stride=2) -> ReLU
  Conv2d(64, 64, 3x3, stride=1) -> ReLU
  Flatten -> Linear(conv_out, 512) -> ReLU -> Linear(512, action_dim)
  ```
- **使用方法**：无需修改调用代码，只需传入正确的 `state_dim`
  - CartPole: `state_dim = 4` (标量)
  - Breakout: `state_dim = (80, 80)` (tuple)

### 2. `dqn_train.py` - 添加环境切换注释
- 在文件开头添加了清晰的环境切换说明
- 修复 `state_dim` 提取逻辑以支持 tuple 输入
- 添加 `results` 目录自动创建

### 3. `dqn_train_breakout.py` - Breakout 专用训练脚本（新增）
- 针对 Breakout 优化的超参数：
  - `buffer_size = 10000`（更大的经验池）
  - `epsilon_decay = 10000`（更长的探索期）
  - `lr = 1e-4`（更小的学习率）
  - `episodes = 5000`（更多训练轮次）
- 预填充 buffer 避免初期训练不稳定
- 每 50 局测试一次（Breakout 单局时间长）

## 使用方法

### CartPole 训练（向量输入）
```bash
python3 dqn_train.py
```

### Breakout 训练（图像输入）
**方法 1：使用专用脚本**
```bash
python3 dqn_train_breakout.py
```

**方法 2：修改 dqn_train.py**
在 `dqn_train.py` 中注释掉 CartPole，取消注释 Breakout：
```python
# from gym_env import GymEnv
# env = GymEnv('CartPole-v1')
from gym_env import BreakoutEnv
env = BreakoutEnv()
```

## 关键差异

| 特性 | CartPole | Breakout |
|------|----------|----------|
| 输入类型 | 向量 (4,) | 图像 (80, 80) |
| 网络结构 | MLP | CNN |
| 学习率 | 3e-4 | 1e-4 |
| Buffer大小 | 2000 | 10000 |
| Epsilon衰减 | 300 | 10000 |
| 训练轮数 | 1200 | 5000+ |
| 测试频率 | 每10局 | 每50局 |

## 进一步优化建议

1. **Target Network**：添加目标网络并周期性更新（每N步硬更新或软更新）
2. **Double DQN**：用在线网络选动作，目标网络评估价值
3. **Prioritized Replay**：根据 TD-error 优先采样重要样本
4. **Dueling DQN**：分离状态价值和优势函数
5. **Multi-step Returns**：使用 n-step bootstrap
6. **Frame Stacking**：堆叠连续帧以捕捉运动信息（Breakout）
7. **GPU加速**：设置 `device='mps'` (Mac) 或 `device='cuda'` (Nvidia)

## 注意事项

- Breakout 训练需要较长时间（几小时到一天），建议先在 CartPole 上验证代码正确性
- 如遇到内存问题，可减小 `buffer_size` 或 `batch_size`
- CNN 网络参数量较大，建议保存 checkpoint：
  ```python
  torch.save(model.state_dict(), 'checkpoint.pth')
  ```
