# 🚇 Intelligent Recognition: 3D Tunnel Pipe Detection System

基于 **PointNet++** 深度学习与 **PCL/Open3D** 几何算法的复杂隧道管道智能提取与参数测算系统。

## ✨ 核心特性 (Features)
- **抗干扰极强**：面对复杂未知的隧道壁与支架背景，结合法向特征（Normals）准确剥离粘连目标。
- **AI + 几何双重校验**：前端使用 PointNet++ 进行点云语义分割，后端采用 RANSAC 进行严苛的圆柱体物理规则拟合。
- **流畅的可视化交互**：基于 PyQt5 + PyVista 构建的 3D 桌面端软件，支持千万级点云数据的流畅渲染。

## 🛠️ 技术栈 (Tech Stack)
- **Deep Learning**: PyTorch, PointNet++
- **Point Cloud Processing**: Open3D, pyransac3d, Numpy
- **GUI & Visualization**: PyQt5, PyVista, pyvistaqt

## 🚀 快速开始 (Quick Start)

### 1. 环境配置
```bash
conda create -n tunnel_env python=3.9
conda activate tunnel_env
pip install -r requirements.txt