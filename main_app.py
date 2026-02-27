import sys
import numpy as np
import pyvista as pv
import open3d as o3d
import pyransac3d as pyrsc
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, QFileDialog)


class TunnelDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("隧道管道智能检测系统 V3.0 (真实数据接入版)")
        self.setGeometry(100, 100, 1050, 750)

        self.raw_points = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QWidget()
        control_panel.setFixedWidth(220)
        control_layout = QVBoxLayout(control_panel)

        # 改为真实的导入按钮
        self.btn_load = QPushButton("📂 1. 导入真实点云 (.pcd/.ply)")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")

        self.btn_detect = QPushButton("🚀 2. 启动降采样+几何识别")
        self.btn_detect.setMinimumHeight(50)
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        self.btn_load.clicked.connect(self.load_real_point_cloud)
        self.btn_detect.clicked.connect(self.run_detection_pipeline)

        control_layout.addWidget(QLabel("<h3 style='color:#333;'>操作流程</h3>"))
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(self.btn_detect)
        control_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#1e1e1e")

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setFixedHeight(150)
        self.log_window.setStyleSheet("background-color: #000000; color: #00FF00; font-family: Consolas;")

        right_layout.addWidget(self.plotter.interactor)
        right_layout.addWidget(self.log_window)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)

        self.log("系统已就绪。请点击左侧按钮导入您自己的 .pcd 或 .ply 文件。")

    def log(self, message):
        self.log_window.append(f"> {message}")
        self.log_window.verticalScrollBar().setValue(self.log_window.verticalScrollBar().maximum())

    def load_real_point_cloud(self):
        """打开文件对话框，使用 Open3D 读取真实的点云文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择隧道点云文件", "", "Point Cloud Files (*.ply *.pcd);;All Files (*)", options=options)

        if file_name:
            self.log(f"正在读取文件: {file_name} ...")
            QApplication.processEvents()  # 防止界面卡顿

            try:
                # 1. 使用 Open3D 读取文件
                pcd = o3d.io.read_point_cloud(file_name)
                self.raw_points = np.asarray(pcd.points)
                num_points = len(self.raw_points)

                if num_points == 0:
                    self.log("[错误] 读取失败：文件为空或格式不支持。")
                    return

                self.log(f"✅ 读取成功！原始数据共 {num_points} 个点。")

                # 2. PyVista 渲染显示
                self.plotter.clear()
                cloud = pv.PolyData(self.raw_points)
                self.plotter.add_mesh(cloud, color="white", point_size=2, name="scene", opacity=0.6)
                self.plotter.reset_camera()

            except Exception as e:
                self.log(f"[错误] 读取异常: {str(e)}")

    def run_detection_pipeline(self):
        """在真实数据上运行：降采样 -> RANSAC全局搜索寻找最大圆柱体"""
        if self.raw_points is None:
            self.log("[错误] 请先导入点云！")
            return

        self.log("===================================")
        # 1. 转换为 Open3D 格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.raw_points)

        # 2. 体素降采样 (极其关键：真实场景点太多，必须稀疏化)
        # voxel_size 视你的点云尺度而定。如果单位是米，0.05表示5厘米一个点
        voxel_size = 0.05
        self.log(f"[步骤 1] 正在进行体素降采样 (网格大小: {voxel_size}m)...")
        QApplication.processEvents()

        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_points = np.asarray(downpcd.points)
        self.log(f"   ► 降采样后剩余点数: {len(downsampled_points)} 个")

        # 注意：由于我们还没有接入深度学习来“抠”出管道区域，
        # 我们现在让 RANSAC 在整个隧道背景里“强行”寻找最大的圆柱体。
        self.log("[步骤 2] 全局 RANSAC 盲搜圆柱体 (此过程可能耗时几秒到十几秒)...")
        QApplication.processEvents()

        cylinder = pyrsc.Cylinder()
        # thresh 是拟合的厚度容忍度，真实隧道噪点大，可以适当调大(如0.08)
        center, axis, radius, inliers = cylinder.fit(downsampled_points, thresh=0.08, maxIteration=3000)

        if radius < 0.1 or radius > 3.0:
            self.log(f"[❌ 失败] 算法在场景中找到的最大圆柱形结构半径为 {radius:.2f}m，不符合物理常理。")
            self.log("这说明背景干扰过大，纯传统算法已失效，必须引入深度学习！")
            return

        # 提取结果并计算真实长度
        inlier_points = downsampled_points[inliers]
        projections = np.dot(inlier_points - center, axis)
        h_min, h_max = projections.min(), projections.max()
        height = h_max - h_min
        true_center = center + axis * (h_max + h_min) / 2.0

        self.log("[✅ 成功] 在复杂背景中捕获圆柱体特征！")
        self.log(f"   ► 估算半径: {radius:.3f} 米")
        self.log(f"   ► 估算长度: {height:.3f} 米")

        self.update_visualization(inlier_points, true_center, axis, radius, height)

    def update_visualization(self, inlier_points, center, axis, radius, height):
        self.plotter.add_mesh(pv.PolyData(self.raw_points), color="#555555", point_size=1, name="scene", opacity=0.2)
        pipe_cloud = pv.PolyData(inlier_points)
        self.plotter.add_mesh(pipe_cloud, color="#00FF00", point_size=5, name="pipe_points",
                              render_points_as_spheres=True)
        bounding_cylinder = pv.Cylinder(center=center, direction=axis, radius=radius, height=height)
        self.plotter.add_mesh(bounding_cylinder, color="red", opacity=0.4, name="bounding_box")
        self.plotter.reset_camera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TunnelDetectorApp()
    window.show()
    sys.exit(app.exec_())