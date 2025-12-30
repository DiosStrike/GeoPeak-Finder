# (神经网络定义)

import torch.nn as nn

class ElevationNet(nn.Module):
    """
    神经网络架构：用于根据 (经度, 纬度) 预测海拔高度。
    """
    def __init__(self, n_in: int = 2, n_out: int = 1):
        super().__init__()
        # 3层隐藏层，每层64个神经元，使用 ReLU 激活函数
        self.seq = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_out)
        )
    
    def forward(self, x):
        return self.seq(x)