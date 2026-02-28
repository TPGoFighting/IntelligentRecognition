import sys
import os
import numpy as np
import pyvista as pv
import open3d as o3d
import pyransac3d as pyrsc
import torch
from pyvistaqt import QtInteractor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit, QLabel, QFileDialog)

# 确保能导入你定义的模型
from models.pointnet2_sem_seg import get_model


class TunnelDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Recognition - 隧道多管道识别系统 V5.5")
        self.setGeometry(100, 100, 1100, 750)
        self.raw_points = None
        self.raw_normals = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 大点云处理配置
        self.max_points = 10000000  # 最大处理点数，超过会自动降采样
        self.voxel_size = 0.05  # 初始体素降采样大小
        self.max_windows = 200  # 最大滑窗数量，防止无限循环
        self.render_point_limit = 50000  # 渲染点数限制，防止图形崩溃

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QWidget()
        control_panel.setFixedWidth(240)
        control_layout = QVBoxLayout(control_panel)

        self.btn_load = QPushButton("📂 1. 导入隧道点云")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")

        self.btn_detect = QPushButton("🚀 2. 启动智能识别")
        self.btn_detect.setMinimumHeight(50)
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")

        # 【修复点】：确保这里的名字与下方定义的一致
        self.btn_load.clicked.connect(self.load_real_point_cloud)
        self.btn_detect.clicked.connect(self.run_intelligent_inference)

        control_layout.addWidget(QLabel("<h3 style='color:#333;'>系统操作台</h3>"))
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
        self.log_window.setFixedHeight(160)
        self.log_window.setStyleSheet("background-color: #000000; color: #00FF00; font-family: Consolas;")

        right_layout.addWidget(self.plotter.interactor)
        right_layout.addWidget(self.log_window)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)
        self.log("系统就绪。")

    def log(self, message):
        self.log_window.append(f"> {message}")
        self.log_window.verticalScrollBar().setValue(self.log_window.verticalScrollBar().maximum())
        QApplication.processEvents()

    def load_real_point_cloud(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "选择点云文件", "", "Point Cloud Files (*.ply *.pcd)")
            if not file_name:
                return

            self.log(f"正在加载点云文件: {os.path.basename(file_name)}")

            # 读取点云
            pcd = o3d.io.read_point_cloud(file_name)
            if len(pcd.points) == 0:
                self.log("[错误] 点云文件为空")
                return

            raw_points = np.asarray(pcd.points)
            self.log(f"原始点数: {len(raw_points)}")

            # 自动降采样：如果点数超过限制，增加体素大小
            voxel_size = self.voxel_size
            if len(raw_points) > self.max_points:
                self.log(f"点云过大，自动降采样...")
                # 根据点数调整体素大小
                ratio = self.max_points / len(raw_points)
                voxel_size = self.voxel_size * (1.0 / ratio) ** (1/3)  # 体积缩放
                voxel_size = max(voxel_size, 0.01)  # 最小体素大小
                self.log(f"调整体素大小: {voxel_size:.3f}")

            # 降采样以提高处理速度
            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                self.log(f"降采样后点数: {len(pcd.points)}")

            self.raw_points = np.asarray(pcd.points)
            # 保存法线（如果存在）
            if pcd.has_normals():
                self.raw_normals = np.asarray(pcd.normals)
            else:
                self.raw_normals = None

            # 渲染点云（限制渲染点数以防止图形崩溃）
            render_points = self.raw_points
            if len(render_points) > self.render_point_limit:
                indices = np.random.choice(len(render_points), self.render_point_limit, replace=False)
                render_points = render_points[indices]
                self.log(f"渲染点数限制为: {self.render_point_limit}")

            self.plotter.clear()
            self.plotter.add_mesh(pv.PolyData(render_points), color="white", point_size=1, name="scene", opacity=0.3)
            self.plotter.reset_camera()
            self.log(f"✅ 点云载入成功，处理点数: {len(self.raw_points)}")

        except Exception as e:
            self.log(f"[错误] 加载点云失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def run_intelligent_inference(self):
        """核心推理逻辑：修复了滑窗越界和 API 警告"""
        if self.raw_points is None:
            self.log("[错误] 未载入数据")
            return

        model_path = "checkpoints/best_pipe_model.pth"
        if not os.path.exists(model_path):
            self.log("[错误] 找不到权重文件")
            return

        # 1. 加载 AI 模型
        model = get_model(num_classes=3).to(self.device)  # 三分类：管道(2)、隧道壁(1)、其他背景(0)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # 2. 预处理
        self.log("预处理：降采样与法向计算...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.raw_points)

        # 如果原始法线存在，使用它们
        if self.raw_normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.raw_normals)
            self.log("使用点云文件中的法线信息")
        else:
            self.log("点云文件中无法线信息，重新计算法线")

        downpcd = pcd.voxel_down_sample(voxel_size=0.05)

        # 如果下采样后没有法线，则计算法线
        if not downpcd.has_normals():
            downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))

        points = np.asarray(downpcd.points)
        normals = np.asarray(downpcd.normals)

        # 3. 滑窗推理
        self.log("AI 正在扫描隧道 (GPU 加速中)...")
        block_size, stride = 3.0, 1.5
        all_labels = np.zeros(len(points))
        counts = np.zeros(len(points))
        xyz_min, xyz_max = points.min(0), points.max(0)

        # 计算滑窗范围，防止无限循环
        z_range = xyz_max[2] - xyz_min[2]
        num_windows = int(np.ceil(z_range / stride))

        # 自动调整步长，限制窗口数量
        if num_windows > self.max_windows:
            old_stride = stride
            stride = max(z_range / self.max_windows, block_size * 0.8)  # 确保有重叠
            num_windows = int(np.ceil(z_range / stride))
            self.log(f"窗口数过多 ({num_windows})，调整步长: {old_stride:.2f} → {stride:.2f}")

        # 生成窗口中心列表
        z_centers = np.linspace(xyz_min[2] + block_size/2, xyz_max[2] - block_size/2, num_windows)

        self.log(f"滑窗配置: {num_windows} 个窗口, 步长 {stride:.2f}m, 区块大小 {block_size:.1f}m")

        processed_windows = 0
        for i, z in enumerate(z_centers):
            try:
                # 进度更新
                if i % 10 == 0 or i == num_windows - 1:
                    self.log(f"  进度: {i+1}/{num_windows} 窗口 ({(i+1)*100/num_windows:.1f}%)")

                mask = (points[:, 2] >= z - block_size/2) & (points[:, 2] < z + block_size/2)
                idx = np.where(mask)[0]

                if len(idx) < 1024:
                    continue

                # 采样固定点数
                if len(idx) >= 4096:
                    sel = np.random.choice(idx, 4096, replace=False)
                else:
                    sel = np.random.choice(idx, 4096, replace=True)

                # 中心化并提取特征
                block_pts = points[sel] - points[sel].mean(0)
                block_feat = np.hstack((block_pts, normals[sel]))

                # 推理
                input_tensor = torch.FloatTensor(block_feat).unsqueeze(0).transpose(2, 1).to(self.device)
                with torch.no_grad():
                    device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
                    with torch.amp.autocast(device_type):
                        pred = model(input_tensor)
                        pred_label = torch.argmax(pred, dim=2).cpu().numpy()[0]

                # 统计管道类别(2)的投票数
                all_labels[sel] += (pred_label == 2).astype(np.int32)
                counts[sel] += 1
                processed_windows += 1

            except Exception as e:
                self.log(f"⚠️  窗口 {i} 处理失败: {str(e)}")
                continue

        self.log(f"滑窗推理完成，成功处理 {processed_windows}/{num_windows} 个窗口")

        # 4. 后续 RANSAC 处理... (保持原样)

        # 4. 【硬核升级】多目标 RANSAC 迭代提取
        self.log("AI 推理完成。开始执行多目标几何提取...")
        self.plotter.remove_actor("scene")  # 隐藏背景
        self.plotter.add_mesh(pv.PolyData(self.raw_points), color="#404040", point_size=1, opacity=0.1, name="bg")

        # 计算管道点：基于投票结果，当管道类别（2）的投票比例超过阈值时认为是管道
        valid_mask = counts > 0
        if not np.any(valid_mask):
            self.log("[错误] 没有有效的推理点")
            return

        # 进一步降低阈值以提高召回率：从0.2降到0.1
        pipe_mask = (all_labels[valid_mask] / counts[valid_mask] > 0.1)
        pipe_points = points[valid_mask][pipe_mask]

        if len(pipe_points) == 0:
            self.log("AI未检测到任何管道")
            return

        self.log(f"AI检测到{len(pipe_points)}个候选管道点")

        found_pipes_count = 0
        remaining_points = pipe_points.copy()

        # 循环提取，直到剩余点数不足以构成一根管
        while len(remaining_points) > 200:
            # 使用简单圆柱拟合算法（pyransac3d有bug）
            # 假设管道沿Z轴方向
            center_xy = remaining_points[:, :2].mean(axis=0)
            radial_distances = np.linalg.norm(remaining_points[:, :2] - center_xy, axis=1)
            radius = np.median(radial_distances)
            center_z = remaining_points[:, 2].mean()
            center = np.array([center_xy[0], center_xy[1], center_z])
            axis = np.array([0, 0, 1])

            # 计算内点
            threshold = 0.08
            distances_to_surface = np.abs(radial_distances - radius)
            inliers = np.where(distances_to_surface < threshold)[0]

            # 物理规则过滤：管道半径应在 0.15m - 0.8m 之间
            if 0.15 < radius < 0.8 and len(inliers) > 50:
                found_pipes_count += 1
                current_pipe_pts = remaining_points[inliers]

                # 渲染每一根管道
                self.add_pipe_to_view(current_pipe_pts, center, axis, radius, found_pipes_count)
                self.log(f"✅ 提取管道 {found_pipes_count}: 半径 {radius:.3f}m")

                # 从候选池中移除这根管的点，继续找下一根
                remaining_points = np.delete(remaining_points, inliers, axis=0)
            else:
                # 如果当前最大的拟合结果都不符合要求，则停止
                break

        if found_pipes_count == 0:
            self.log("几何校验失败，未发现符合物理特征的圆柱体。")
        else:
            self.log(f"任务完成，共发现 {found_pipes_count} 根管道。")

    def add_pipe_to_view(self, pts, center, axis, radius, pipe_id):
        """将单根管道渲染到 3D 视口中"""
        projs = np.dot(pts - center, axis)
        h = projs.max() - projs.min()
        true_center = center + axis * (projs.max() + projs.min()) / 2.0

        # 为每根管随机分配一个颜色
        color = list(np.random.choice(range(256), size=3) / 255.0)

        self.plotter.add_mesh(pv.PolyData(pts), color=color, point_size=4, name=f"pipe_pts_{pipe_id}")
        geom = pv.Cylinder(center=true_center, direction=axis, radius=radius, height=h)
        self.plotter.add_mesh(geom, color=color, opacity=0.4, name=f"pipe_geom_{pipe_id}")
        self.plotter.reset_camera()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TunnelDetectorApp()
    window.show()
    sys.exit(app.exec_())