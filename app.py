import streamlit as st
from PIL import Image

# Configuration / 页面配置
st.set_page_config(page_title="Pittsburgh Urban Risk Engine")

# --- Sidebar / 侧边栏 ---
st.sidebar.title("Project Details / 项目详情")
st.sidebar.info("""
**Author / 作者**: Tanghao Chen (Dios)
**Core Technologies / 核心技术**: 
- Neural Network Digital Twin / 神经网络数字孪生
- Stochastic Gradient Ascent (SGA) / 随机梯度上升
- Automatic Differentiation (Autograd) / 自动微分
""")

# --- Main Title / 主标题 ---
st.title("Pittsburgh Urban Risk Engine / 匹兹堡城市风险分析引擎")
st.markdown("""
This project utilizes Deep Learning to construct a **Digital Twin** of Pittsburgh's terrain, enabling multi-dimensional environmental risk assessments. 
本项目利用深度学习技术构建了匹兹堡地形的**数字孪生体**，并基于此模型进行多维度的环境风险评估。
""")

# --- Tabs / 选项卡切换 ---
tab1, tab2, tab3 = st.tabs([
    "Terrain Optimization / 基础地形寻优", 
    "Flood Analysis / 洪涝分析", 
    "Landslide Risk / 滑坡风险"
])

with tab1:
    st.header("Highest Peak Localization / 梯度上升寻优")
    image = Image.open('results/1_basic_gradient_map.png')
    st.image(image, caption="Highest altitude point identified via SGA / 基于 SGA 算法定位的城市最高点")
    st.write("""
    Using Stochastic Gradient Ascent, the model accurately localized the Robert Williams Reservoir as the geographic peak within the digital twin.
    利用随机梯度上升算法，模型在数字孪生模型中精准定位了 Robert Williams Reservoir 这一地理制高点。
    """)

with tab2:
    st.header("Watershed & Sink Detection / 汇水点内涝评估")
    image = Image.open('results/2_flood_risk_analysis.png')
    st.image(image, caption="Flood risk zones simulated via surface runoff / 模拟地表径流生成的内涝高风险区")
    st.write("""
    By reversing the gradient direction, the model identifies local minima (Sinks), representing areas with significant water accumulation risk during extreme precipitation.
    通过反转梯度方向，模型识别出了城市中的局部极小值点（Sinks），这些区域在极端降水天气下具有极高的积水风险。
    """)

with tab3:
    st.header("Slope & Landslide Monitoring / 坡度滑坡监测")
    image = Image.open('results/3_landslide_risk_analysis.png')
    st.image(image, caption="High-risk slope boundaries calculated via Autograd / 基于自动微分计算的高风险坡度警戒线")
    st.write("""
    Leveraging PyTorch's Automatic Differentiation, the engine calculates global slope magnitude. Red contours delineate zones exceeding safe steepness thresholds for landslide risk.
    利用 PyTorch 的自动微分功能，引擎计算了地形的全域坡度。红色线条标出了坡度超过安全阈值的高危滑坡区域。
    """)