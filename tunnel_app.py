import sys
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QFileDialog, QLabel)
from PyQt5.QtCore import Qt


class TunnelDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("检测系统 V1.0 ")
        self.setGeometry(100, 100, 1000, 700)  # x, y, width, height

        # 核心数据容器
        self.current_point_cloud = None

        self.init_ui()

    def init_ui(self):
        # 1. 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 2. 左侧：控制面板
        control_panel = QWidget()
        control_panel.setFixedWidth(200)
        control_layout = QVBoxLayout(control_panel)

        self.btn_load = QPushButton("导入点云 (.pcd/.ply)")
        self.btn_demo = QPushButton("生成演示点云")
        self.btn_detect = QPushButton("开始识别")
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")

        # 绑定按钮事件 (Signals and Slots)
        self.btn_load.clicked.connect(self.load_point_cloud)
        self.btn_demo.clicked.connect(self.generate_demo_data)
        self.btn_detect.clicked.connect(self.mock_detection)

        control_layout.addWidget(QLabel("<b>操作面板</b>"))
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_demo)
        control_layout.addWidget(QLabel("<hr>"))  # 分割线
        control_layout.addWidget(self.btn_detect)
        control_layout.addStretch()  # 把按钮顶到上方

        # 3. 右侧：3D 可视化区 + 日志区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 3.1 核心：PyVista 的 Qt 交互窗口
        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#2b2b2b")  # 护眼深色背景

        # 3.2 底部：日志输出窗口
        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setFixedHeight(120)
        self.log_window.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")

        right_layout.addWidget(self.plotter.interactor)
        right_layout.addWidget(QLabel("<b>系统日志输出:</b>"))
        right_layout.addWidget(self.log_window)

        # 4. 组装主界面
        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)

        self.log("请导入点云或生成演示数据。")

    def log(self, message):
        """向日志窗口添加信息"""
        self.log_window.append(f"[系统] {message}")
        self.log_window.verticalScrollBar().setValue(self.log_window.verticalScrollBar().maximum())

    def load_point_cloud(self):
        """打开文件对话框加载点云 (暂未完全实现文件读取，留出接口)"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择点云文件", "",
                                                   "Point Cloud Files (*.ply *.pcd);;All Files (*)", options=options)
        if file_name:
            self.log(f"加载文件: {file_name}")
            # 这里后续会替换为 pv.read(file_name)
            self.log("文件读取功能将在接入真实数据后激活。")

    def generate_demo_data(self):
        """生成一段带噪声的圆柱体点云，模拟隧道里的管道"""
        try:
            self.log("正在生成带有噪点的数据...")

            # 1. 彻底清理旧内容
            self.plotter.clear()

            # 2. 生成几何体并转换为点云
            # 注意：direction 是圆柱轴向，我们让它沿 X 轴延伸
            cylinder_mesh = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=0.8, height=10)

            # 提取顶点并增加随机噪声
            points = cylinder_mesh.points
            noise = np.random.normal(0, 0.02, points.shape)
            noisy_points = points + noise

            # 3. 创建 PyVista 点云对象
            self.current_point_cloud = pv.PolyData(noisy_points)

            # 4. 添加到渲染器
            # 使用 name 属性可以确保多次点击时替换同一对象，而不是叠加
            self.plotter.add_mesh(
                self.current_point_cloud,
                color="white",
                point_size=3.0,
                render_points_as_spheres=True,
                name="scene_points"
            )

            # 5. 关键：重置相机并强制渲染
            self.plotter.reset_camera()
            self.plotter.render()  # 显式刷新界面

            self.log("演示数据生成完毕。")

        except Exception as e:
            self.log(f"生成失败: {str(e)}")

    def mock_detection(self):
        """模拟深度学习推理和 PCL 几何校验的过程"""
        if self.current_point_cloud is None:
            self.log("请先加载点云数据！")
            return

        self.log("启动深度学习特征提取")
        self.log("正在执行 PCL 几何规则校验")

        # 模拟 1: 把识别出的点云变成高亮绿色
        self.plotter.add_mesh(self.current_point_cloud, color="#00FF00", point_size=4, render_points_as_spheres=True,
                              name="scene")

        # 模拟 2: 用半透明红色圆柱体“框”住识别结果 (相当于 PCL 算出的 Bounding Cylinder)
        bounding_cylinder = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=0.85, height=10.2)
        self.plotter.add_mesh(bounding_cylinder, color="red", opacity=0.3, name="bbox")

        self.log("识别成功")
        self.log("提取参数：中心点:(0,0,0) | 半径: 0.85m | 长度: 10.2m")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TunnelDetectorApp()
    window.show()
    sys.exit(app.exec_())