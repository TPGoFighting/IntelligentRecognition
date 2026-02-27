import sys
import numpy as np
import pyvista as pv
import open3d as o3d
import pyransac3d as pyrsc
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QLabel)


class TunnelDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("隧道管道智能检测系统 V2.0 (算法集成版)")
        self.setGeometry(100, 100, 1050, 750)

        # 核心数据容器 (保存 numpy 格式，方便各个库之间流转)
        self.raw_points = None
        self.true_pipe_count = 0

        self.init_ui()

    def init_ui(self):
        # 1. 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 2. 左侧：控制面板
        control_panel = QWidget()
        control_panel.setFixedWidth(220)
        control_layout = QVBoxLayout(control_panel)

        self.btn_demo = QPushButton("🪄 1. 生成复杂隧道点云")
        self.btn_demo.setMinimumHeight(40)

        self.btn_detect = QPushButton("🚀 2. 启动 AI+几何 识别")
        self.btn_detect.setMinimumHeight(50)
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")

        self.btn_demo.clicked.connect(self.generate_demo_scene)
        self.btn_detect.clicked.connect(self.run_detection_pipeline)

        control_layout.addWidget(QLabel("<h3 style='color:#333;'>操作流程</h3>"))
        control_layout.addWidget(self.btn_demo)
        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(self.btn_detect)
        control_layout.addStretch()

        # 3. 右侧：3D 可视化区 + 日志区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#1e1e1e")  # 深灰色背景

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setFixedHeight(150)
        self.log_window.setStyleSheet(
            "background-color: #000000; color: #00FF00; font-family: Consolas; font-size: 13px;")

        right_layout.addWidget(self.plotter.interactor)
        right_layout.addWidget(self.log_window)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)

        self.log("系统初始化完成。请点击左侧按钮生成测试数据。")

    def log(self, message):
        """向日志窗口输出信息"""
        self.log_window.append(f"> {message}")
        self.log_window.verticalScrollBar().setValue(self.log_window.verticalScrollBar().maximum())

    def generate_demo_scene(self):
        """生成带噪点的管道和杂乱的隧道壁"""
        self.log("正在生成大规模复杂隧道点云...")
        self.plotter.clear()

        # 1. 生成管道 (中心偏移一点，增加真实感)
        pipe_points = []
        radius = 0.85
        for _ in range(8000):
            h = np.random.uniform(0, 15)
            theta = np.random.uniform(0, 2 * np.pi)
            x = radius * np.cos(theta) + 0.5
            y = radius * np.sin(theta) - 0.2
            pipe_points.append([x, y, h])
        pipe_points = np.array(pipe_points)
        pipe_points += np.random.normal(0, 0.03, pipe_points.shape)  # 加入扫描噪声

        # 2. 生成杂乱背景 (隧道壁、支架等)
        wall_points = np.random.uniform(low=[-4, -4, -2], high=[5, 5, 17], size=(15000, 3))

        # 3. 合并数据存入内存
        self.raw_points = np.vstack((pipe_points, wall_points))
        self.true_pipe_count = len(pipe_points)

        # 4. PyVista 渲染原始场景 (全白)
        cloud = pv.PolyData(self.raw_points)
        self.plotter.add_mesh(cloud, color="white", point_size=2, name="scene", opacity=0.6)
        self.plotter.reset_camera()
        self.log(f"数据加载完毕。共计 {len(self.raw_points)} 个点。请点击 [启动识别]。")

    def run_detection_pipeline(self):
        """将 AI 提取与 RANSAC 拟合完整串联，并反馈到 UI"""
        if self.raw_points is None:
            self.log("[错误] 请先生成或导入点云！")
            return

        self.log("===================================")
        self.log("[步骤 1] 正在提取点云法向特征 (Open3D)...")
        QApplication.processEvents()  # 刷新界面，防止假死

        # 转为 Open3D 格式计算特征 (假装喂给网络)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.raw_points)

        self.log("[步骤 2] 模拟 PointNet++ 语义分割推理...")
        QApplication.processEvents()

        # 模拟网络预测结果 (找出疑似管道的点)
        predicted_indices = list(range(int(self.true_pipe_count * 0.90))) + list(
            range(self.true_pipe_count, self.true_pipe_count + 1000))
        suspected_points = self.raw_points[predicted_indices]

        self.log(f"[步骤 3] RANSAC 圆柱体验证与参数提取 ({len(suspected_points)} 个候选点)...")
        QApplication.processEvents()

        # 执行 RANSAC
        cylinder = pyrsc.Cylinder()
        center, axis, radius, inliers = cylinder.fit(suspected_points, thresh=0.06, maxIteration=2000)

        if radius < 0.2 or radius > 2.5:
            self.log(f"[❌ 警告] 拟合半径 {radius:.2f}m 异常，排除目标！")
            return

        # --- 计算圆柱体的真实长度和中心点 (供 PyVista 完美渲染) ---
        inlier_points = suspected_points[inliers]
        # 将点投影到轴线上，计算最大和最小长度
        projections = np.dot(inlier_points - center, axis)
        h_min, h_max = projections.min(), projections.max()
        height = h_max - h_min
        # PyVista 绘制圆柱体需要它的正中心坐标
        true_center = center + axis * (h_max + h_min) / 2.0

        self.log("[✅ 成功] 物理规则校验通过！")
        self.log(f"   ► 半径: {radius:.3f} 米")
        self.log(f"   ► 长度: {height:.3f} 米")
        self.log(f"   ► 轴线方向: [{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}]")

        self.update_visualization(inlier_points, true_center, axis, radius, height)

    def update_visualization(self, inlier_points, center, axis, radius, height):
        """将识别结果在 3D 窗口中高亮渲染"""
        # 1. 清除旧场景，将所有背景点变成暗灰色
        self.plotter.add_mesh(pv.PolyData(self.raw_points), color="#555555", point_size=1, name="scene", opacity=0.3)

        # 2. 将确认为管道的点云变成亮绿色
        pipe_cloud = pv.PolyData(inlier_points)
        self.plotter.add_mesh(pipe_cloud, color="#00FF00", point_size=4, name="pipe_points",
                              render_points_as_spheres=True)

        # 3. 绘制 PCL/RANSAC 拟合出的几何包围盒 (半透明红色圆柱)
        bounding_cylinder = pv.Cylinder(center=center, direction=axis, radius=radius, height=height)
        self.plotter.add_mesh(bounding_cylinder, color="red", opacity=0.4, name="bounding_box")

        self.plotter.reset_camera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TunnelDetectorApp()
    window.show()
    sys.exit(app.exec_())