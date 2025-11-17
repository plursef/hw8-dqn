import torch
from torch import nn

class QNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, lr):
        super(QNetwork, self).__init__()
        # 判断输入是向量还是图像
        # 如果 input_dim 是 tuple/list，则认为是图像输入 (H, W) 或 (C, H, W)
        self.is_image = isinstance(input_dim, (tuple, list))
        
        if self.is_image:
            # CNN for image input (e.g., Breakout)
            # 假设输入为 (H, W) 单通道灰度图
            h, w = input_dim if len(input_dim) == 2 else input_dim[1:]
            in_channels = 1 if len(input_dim) == 2 else input_dim[0]
            
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            # 计算 conv 输出的 flatten 维度
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, h, w)
                conv_out = self.conv(dummy_input)
                self.conv_out_size = conv_out.view(1, -1).size(1)
            
            self.fc = nn.Sequential(
                nn.Linear(self.conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
        else:
            # MLP for vector input (e.g., CartPole)
            self.seq = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)
    
    def inference(self, obs):
        if self.is_image:
            # obs shape: [B, H, W] or [H, W]
            if obs.dim() == 2:
                obs = obs.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif obs.dim() == 3:
                obs = obs.unsqueeze(1)  # [B, 1, H, W]
            x = self.conv(obs)
            x = x.view(x.size(0), -1)
            q_value = self.fc(x)
        else:
            q_value = self.seq(obs)
        return q_value
    
    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # 新增：用于 learn() 调用
    def forward(self, obs):
        return self.inference(obs)
