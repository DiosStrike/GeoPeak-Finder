# GeoPeak-Finder: Terrain Digital Twin & Optimization
### Terrain Digital Twin and Geographic Peak Optimization Identification

---

*[English Version](#english-version) | [中文版本](#中文版本)*

---
<a name="english-version"></a>
# English Version

**Live Demo** 
* **Public Link**: https://geopeak-finder-qumxnbzjdabafzeyjla6cs.streamlit.app

**Project Overview**
GeoPeak-Finder is an AI-driven geospatial project designed to create a Terrain Digital Twin of the Pittsburgh area. By fitting discrete elevation data with PyTorch Neural Networks, the system generates a differentiable surface used for automated peak identification via Multi-start Gradient Ascent (SGA) and advanced environmental risk analysis.

**Project Structure**
```text
GeoPeak-Finder/
├── data/               # Raw geospatial elevation data
├── models/             # Saved model weights (.pth)
├── results/            # Generated analytical outputs
│   ├── 1_basic_gradient_map.png
│   ├── 2_flood_risk_analysis.png
│   └── 3_landslide_risk_analysis.png
├── src/                # Core implementation logic
│   ├── model.py        # PyTorch Neural Network architecture for terrain fitting
│   ├── engine.py       # Training pipelines and SGA optimization algorithms
│   ├── utils.py        # Data loaders and geospatial preprocessing tools
│   └── visualizer.py   # Plotting logic for terrain, risk maps, and trajectories
├── app.py              # Streamlit-based interactive web interface
├── main.py             # Main entry point for training and optimization
└── requirements.txt    # Project dependencies
```

# **Installation & Usage**

1. Setup:
```Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Run Pipeline:
```Bash
python3 main.py
```
3. Interactive UI:
```Bash
streamlit run app.py
```
# **Core Results:**
- Peak Identified: Robert Williams Reservoir (~365m).
- Flood Risk: Automated generation of flood susceptibility maps based on neural terrain gradients and flow accumulation.
- Landslide Risk: Automated generation of landslide susceptibility maps based on neural terrain slope and gradient analysis.

---
<a name="中文版本"></a>

# 中文版本

# GeoPeak-Finder: 地形数字孪生与寻优
### 基于深度学习的地形建模与多起点梯度上升算法实现

---

**在线演示**
* **链接:** https://geopeak-finder-qumxnbzjdabafzeyjla6cs.streamlit.app

**项目简介**
项目概览 GeoPeak-Finder 是一个人工智能驱动的地理空间项目，旨在构建匹兹堡地区的地形数字孪生（Digital Twin）。通过利用 PyTorch 神经网络拟合离散的高程数据，系统生成了一个可微的曲面，用于通过多起点梯度上升算法（SGA）进行自动峰值识别和高级环境风险分析。

**项目结构**
```text
GeoPeak-Finder/
├── data/               # 原始地理空间高程数据
├── models/             # 已保存的模型权重 (.pth)
├── results/            # 生成的分析结果输出
│   ├── 1_basic_gradient_map.png
│   ├── 2_flood_risk_analysis.png
│   └── 3_landslide_risk_analysis.png
├── src/                # 核心实现逻辑
│   ├── model.py        # 用于地形拟合的 PyTorch 神经网络架构
│   ├── engine.py       # 训练流水线与 SGA 寻优算法
│   ├── utils.py        # 数据加载器与地理空间预处理工具
│   └── visualizer.py   # 地形、风险图及寻优轨迹的绘图逻辑
├── app.py              # 基于 Streamlit 的交互式 Web 界面
├── main.py             # 训练与寻优流水线的主入口
└── requirements.txt    # 项目依赖项
```

# **安装与使用**

1. 环境初始化：
```Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. 运行pipeline：
```Bash
python3 main.py
```
3. 交互式界面：
```Bash
streamlit run app.py
```
# **核心成果：**
- 识别的峰值: Robert Williams Reservoir (罗伯特·威廉姆斯水库，约 365 米)。
- 洪涝风险: 基于神经地形梯度与汇水分析，自动生成洪涝易发性风险图。
- 滑坡风险: 基于神经地形坡度与梯度分析，自动生成滑坡灾害易发性风险图。
