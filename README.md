# GeoPeak-Finder: Terrain Digital Twin & Optimization

## 项目简介
本项目利用 PyTorch 神经网络对匹兹堡地理空间数据进行拟合，构建地形数字孪生模型。通过多起点梯度上升算法 (SGA) 自动定位并识别区域内的海拔最高点。

## 项目结构
```text
GeoPeak-Finder/
├── src/                # 核心源代码逻辑
│   ├── model.py        # 神经网络架构定义
│   ├── engine.py       # 训练与寻优算法实现
│   ├── utils.py        # 数据处理与加载工具
│   └── visualizer.py   # 地形可视化绘图模块
├── data/               # 原始地形数据文件夹
├── models/             # 存放训练模型权重及结果图
├── .gitignore          # Git 忽略文件配置
├── main.py             # 项目一键启动程序入口
└── requirements.txt    # 项目依赖环境清单

```

## 环境配置

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装项目必要依赖
pip install -r requirements.txt

```

## 运行指南

```bash
# 在激活虚拟环境的状态下运行主程序
python3 main.py

```

## 核心结果

* 识别最高点: Robert Williams Reservoir
* 预测海拔高度: 约 365 米
* 可视化产物: 自动生成的等高线地形图存储于 models/contour_result.png