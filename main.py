import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import open3d as o3d
import pyvista as pv
import torch
from pyvistaqt import QtInteractor
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_loader import load_config
from core.data_generator import UniversalDataGenerator
from core.inference_engine import UniversalInferenceEngine
from core.shape_plugins import get_plugin_class
from models.pointnet2_sem_seg import get_model


class UniversalShapeDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Recognition - Universal Shape Detection System")
        self.setGeometry(100, 100, 1200, 800)

        self.raw_points = None
        self.raw_normals = None
        self.detected_objects = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_points = 10000000
        self.voxel_size = 0.05
        self.max_windows = 200
        self.render_point_limit = 50000

        self.config_path = "config/shape_config.yaml"
        self.config = None
        self.model_path = "models/universal/best_model.pth"
        self.engine = None
        self.engine_signature = None
        self.current_mode = "universal"

        self.training_process = None

        self.default_palette = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.5, 0.0],
            [0.7, 0.2, 1.0],
        ]

        self.init_ui()
        self.load_default_config()
        self.update_status_labels()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        control_panel = QWidget()
        control_panel.setFixedWidth(320)
        control_layout = QVBoxLayout(control_panel)

        control_layout.addWidget(QLabel("<h3 style='color:#333;'>通用控制面板</h3>"))

        control_layout.addWidget(QLabel("<b>配置与模型</b>"))
        self.btn_load_config = QPushButton("加载配置文件 (YAML)")
        self.btn_load_model = QPushButton("加载模型权重 (.pth)")
        self.label_config_status = QLabel("配置: -")
        self.label_model_status = QLabel("模型: -")

        self.btn_load_config.clicked.connect(self.load_config_file)
        self.btn_load_model.clicked.connect(self.load_model_file)

        control_layout.addWidget(self.btn_load_config)
        control_layout.addWidget(self.btn_load_model)
        control_layout.addWidget(self.label_config_status)
        control_layout.addWidget(self.label_model_status)

        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(QLabel("<b>数据与推理</b>"))

        self.btn_load = QPushButton("导入点云文件")
        self.btn_detect_universal = QPushButton("运行通用检测")
        self.btn_detect_legacy = QPushButton("运行隧道检测(旧版)")

        self.btn_load.clicked.connect(self.load_point_cloud)
        self.btn_detect_universal.clicked.connect(self.run_universal_inference)
        self.btn_detect_legacy.clicked.connect(self.run_intelligent_inference)

        self.btn_detect_universal.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_detect_legacy.setStyleSheet("background-color: #607D8B; color: white;")

        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_detect_universal)
        control_layout.addWidget(self.btn_detect_legacy)

        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(QLabel("<b>检测结果</b>"))

        self.btn_export_json = QPushButton("导出检测JSON")
        self.btn_clear_scene = QPushButton("清除场景")

        self.btn_export_json.clicked.connect(self.export_results_json)
        self.btn_clear_scene.clicked.connect(self.clear_scene)

        control_layout.addWidget(self.btn_export_json)
        control_layout.addWidget(self.btn_clear_scene)

        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(QLabel("<b>批量处理</b>"))

        self.btn_batch_inference = QPushButton("运行批量推理")
        self.btn_batch_inference.clicked.connect(self.run_batch_inference)
        control_layout.addWidget(self.btn_batch_inference)

        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(QLabel("<b>数据生成</b>"))

        data_gen_layout = QHBoxLayout()
        self.input_num_train = QLineEdit("100")
        self.input_num_train.setFixedWidth(80)
        self.input_num_test = QLineEdit("20")
        self.input_num_test.setFixedWidth(80)
        self.btn_generate_data = QPushButton("生成数据")
        self.btn_generate_data.clicked.connect(self.generate_training_data)

        data_gen_layout.addWidget(QLabel("训练集:"))
        data_gen_layout.addWidget(self.input_num_train)
        data_gen_layout.addWidget(QLabel("测试集:"))
        data_gen_layout.addWidget(self.input_num_test)
        data_gen_layout.addWidget(self.btn_generate_data)
        control_layout.addLayout(data_gen_layout)

        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(QLabel("<b>训练参数</b>"))

        train_param_layout = QHBoxLayout()
        self.input_epochs = QLineEdit("10")
        self.input_epochs.setFixedWidth(60)
        self.input_batch_size = QLineEdit("8")
        self.input_batch_size.setFixedWidth(60)
        self.input_learning_rate = QLineEdit("0.001")
        self.input_learning_rate.setFixedWidth(80)

        train_param_layout.addWidget(QLabel("训练轮次:"))
        train_param_layout.addWidget(self.input_epochs)
        train_param_layout.addWidget(QLabel("批次:"))
        train_param_layout.addWidget(self.input_batch_size)
        train_param_layout.addWidget(QLabel("学习率:"))
        train_param_layout.addWidget(self.input_learning_rate)
        control_layout.addLayout(train_param_layout)

        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(QLabel("<b>运行参数</b>"))

        runtime_layout = QHBoxLayout()
        self.input_voxel_size = QLineEdit("0.05")
        self.input_voxel_size.setFixedWidth(60)
        self.input_max_windows = QLineEdit("200")
        self.input_max_windows.setFixedWidth(60)
        self.input_render_limit = QLineEdit("50000")
        self.input_render_limit.setFixedWidth(70)
        self.btn_update_params = QPushButton("更新")
        self.btn_update_params.clicked.connect(self.update_runtime_params)

        runtime_layout.addWidget(QLabel("体素大小:"))
        runtime_layout.addWidget(self.input_voxel_size)
        runtime_layout.addWidget(QLabel("窗口数:"))
        runtime_layout.addWidget(self.input_max_windows)
        runtime_layout.addWidget(QLabel("渲染限制:"))
        runtime_layout.addWidget(self.input_render_limit)
        runtime_layout.addWidget(self.btn_update_params)
        control_layout.addLayout(runtime_layout)

        control_layout.addWidget(QLabel("<hr>"))
        control_layout.addWidget(QLabel("<b>工具</b>"))

        self.btn_generate_scene = QPushButton("生成测试场景")
        self.btn_start_training = QPushButton("启动训练脚本")

        self.btn_generate_scene.clicked.connect(self.generate_test_scene_preview)
        self.btn_start_training.clicked.connect(self.start_training_script)

        control_layout.addWidget(self.btn_generate_scene)
        control_layout.addWidget(self.btn_start_training)

        control_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#1e1e1e")

        self.log_window = QTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setFixedHeight(180)
        self.log_window.setStyleSheet("background-color: #000000; color: #00FF00; font-family: Consolas;")

        right_layout.addWidget(self.plotter.interactor)
        right_layout.addWidget(self.log_window)

        main_layout.addWidget(control_panel)
        main_layout.addWidget(right_panel)

        self.log("Initialization complete.")

    def log(self, message: str):
        self.log_window.append(f"> {message}")
        self.log_window.verticalScrollBar().setValue(self.log_window.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _run_with_exception_logging(self, action_name: str, func):
        try:
            return func()
        except Exception as exc:
            self.log(f"[ERROR] {action_name} failed: {exc}")
            self.log(traceback.format_exc())
            return None

    def load_default_config(self):
        if not os.path.exists(self.config_path):
            self.log(f"[WARN] Default config not found: {self.config_path}")
            return

        def _load():
            self.config = load_config(self.config_path)
            self.log(
                f"Config loaded: {self.config_path} | shapes={self.config.shape_names} | "
                f"num_classes={self.config.num_classes}"
            )

        self._run_with_exception_logging("Load default config", _load)

    def update_status_labels(self):
        config_name = Path(self.config_path).name if self.config_path else "-"
        model_name = Path(self.model_path).name if self.model_path else "-"

        self.label_config_status.setText(f"Config: {config_name}")
        self.label_model_status.setText(f"Model: {model_name}")

    def load_config_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select config file", "", "YAML Files (*.yaml *.yml)")
        if not file_name:
            return

        def _load():
            self.config = load_config(file_name)
            self.config_path = file_name
            self.engine = None
            self.engine_signature = None
            self.update_status_labels()
            self.log(f"Config loaded: {file_name}")
            self.log(f"Available shapes: {self.config.shape_names}")

        self._run_with_exception_logging("Load config", _load)

    def _extract_model_num_classes(self, model_path: str):
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        for key, value in state_dict.items():
            if key.endswith("conv2.weight") and hasattr(value, "shape") and len(value.shape) >= 1:
                return int(value.shape[0])

        return None

    def _validate_model_config_compatibility(self, model_path: str):
        if self.config is None:
            self.log("[WARN] Config not loaded. Skip class compatibility check.")
            return

        model_classes = self._extract_model_num_classes(model_path)
        if model_classes is None:
            self.log("[WARN] Cannot infer output classes from model checkpoint.")
            return

        if model_classes != self.config.num_classes:
            raise ValueError(
                f"Model output classes ({model_classes}) != config.num_classes ({self.config.num_classes})"
            )

    def load_model_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select model weights", "", "Model Files (*.pth)")
        if not file_name:
            return

        def _load():
            self._validate_model_config_compatibility(file_name)
            self.model_path = file_name
            self.engine = None
            self.engine_signature = None
            self.update_status_labels()
            self.log(f"Model loaded: {file_name}")

        self._run_with_exception_logging("Load model", _load)

    def _parse_numpy_point_cloud(self, file_name: str):
        ext = Path(file_name).suffix.lower()

        if ext == ".npz":
            data = np.load(file_name)
            points = data["points"] if "points" in data else data["arr_0"]
            normals = data["normals"] if "normals" in data else None
            return np.asarray(points), None if normals is None else np.asarray(normals)

        if ext == ".npy":
            arr = np.load(file_name, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                if arr.shape[1] >= 6:
                    return arr[:, :3], arr[:, 3:6]
                if arr.shape[1] >= 3:
                    return arr[:, :3], None
            raise ValueError("Unsupported .npy format. Expected Nx3 or Nx6 array.")

        if ext == ".txt":
            arr = np.loadtxt(file_name)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] >= 6:
                return arr[:, :3], arr[:, 3:6]
            if arr.shape[1] >= 3:
                return arr[:, :3], None
            raise ValueError("Unsupported .txt format. Expected at least 3 columns.")

        raise ValueError(f"Unsupported numpy/text extension: {ext}")

    def _estimate_normals_if_missing(self):
        if self.raw_points is None:
            return
        if self.raw_normals is not None and len(self.raw_normals) == len(self.raw_points):
            return

        self.log("Normals missing. Estimating normals with Open3D...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.raw_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
        self.raw_normals = np.asarray(pcd.normals)

    def load_point_cloud(self):
        def _load():
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select point cloud file",
                "",
                "Point Cloud Files (*.ply *.pcd *.npz *.npy *.txt)",
            )
            if not file_name:
                return

            self.log(f"Loading point cloud: {os.path.basename(file_name)}")
            ext = Path(file_name).suffix.lower()

            if ext in [".ply", ".pcd"]:
                pcd = o3d.io.read_point_cloud(file_name)
                if len(pcd.points) == 0:
                    raise ValueError("Point cloud file is empty")
                points = np.asarray(pcd.points)
                normals = np.asarray(pcd.normals) if pcd.has_normals() else None
            else:
                points, normals = self._parse_numpy_point_cloud(file_name)
                if len(points) == 0:
                    raise ValueError("Point data is empty")

            self.log(f"Raw points: {len(points)}")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if normals is not None and len(normals) == len(points):
                pcd.normals = o3d.utility.Vector3dVector(normals)

            voxel_size = self.voxel_size
            if len(points) > self.max_points:
                ratio = self.max_points / len(points)
                voxel_size = max(self.voxel_size * (1.0 / ratio) ** (1 / 3), 0.01)
                self.log(f"Point cloud too large. Auto-adjust voxel_size to {voxel_size:.4f}")

            if voxel_size > 0:
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

            self.raw_points = np.asarray(pcd.points)
            self.raw_normals = np.asarray(pcd.normals) if pcd.has_normals() else None

            self._estimate_normals_if_missing()

            self.log(f"Processed points: {len(self.raw_points)}")
            self.log(f"Normals available: {'yes' if self.raw_normals is not None else 'no'}")

            self.render_background_point_cloud()

        self._run_with_exception_logging("Load point cloud", _load)

    def render_background_point_cloud(self):
        if self.raw_points is None:
            return

        render_points = self.raw_points
        if len(render_points) > self.render_point_limit:
            idx = np.random.choice(len(render_points), self.render_point_limit, replace=False)
            render_points = render_points[idx]
            self.log(f"Render points limited to {self.render_point_limit}")

        self.plotter.clear()
        self.plotter.add_mesh(
            pv.PolyData(render_points),
            color=[0.6, 0.6, 0.6],
            point_size=2,
            opacity=0.25,
            name="background_scene",
        )
        self.plotter.reset_camera()

    def _ensure_engine_ready(self):
        if self.config is None:
            raise ValueError("Config is not loaded")
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        signature = (self.model_path, self.config_path, self.device.type)
        if self.engine is None or self.engine_signature != signature:
            self.log("Initializing universal inference engine...")
            self.engine = UniversalInferenceEngine(self.model_path, self.config, self.device.type)
            self.engine_signature = signature

    def run_universal_inference(self):
        def _run():
            self.current_mode = "universal"

            if self.raw_points is None:
                raise ValueError("No point cloud loaded")
            if self.raw_normals is None:
                self._estimate_normals_if_missing()

            self._ensure_engine_ready()

            self.log(
                f"Running universal inference | points={len(self.raw_points)} | "
                f"max_windows={self.config.inference.get('max_windows', self.max_windows)}"
            )

            objects = self.engine.infer(self.raw_points, self.raw_normals)
            self.detected_objects = objects
            self.render_detected_objects(objects)
            self.log_detection_summary(objects)

        self._run_with_exception_logging("Run universal inference", _run)

    def _get_shape_color(self, shape_type: str, idx: int):
        if self.config is not None:
            colors = self.config.visualization.get("shape_colors", {})
            if shape_type in colors:
                return colors[shape_type]

        return self.default_palette[idx % len(self.default_palette)]

    def render_detected_objects(self, objects):
        self.plotter.clear()
        self.render_background_point_cloud()

        for idx, obj in enumerate(objects):
            try:
                plugin_class = get_plugin_class(obj.shape_type)
                shape_config = self.config.get_shape_config(obj.shape_type)
                plugin = plugin_class(
                    {
                        "params": shape_config.params,
                        "fitting": shape_config.fitting,
                        "min_points": shape_config.min_points,
                        "scene_bounds": self.config.scene["bounds"],
                    }
                )
                color = self._get_shape_color(obj.shape_type, idx)
                plugin.visualize(self.plotter, obj.params, obj.points, color=color)
            except Exception as exc:
                self.log(f"[WARN] Render failed for {obj.shape_type}: {exc}")

        self.plotter.reset_camera()

    def log_detection_summary(self, objects):
        self.log(f"Detected objects: {len(objects)}")

        counts = Counter([obj.shape_type for obj in objects])
        if counts:
            for shape_type, count in sorted(counts.items()):
                self.log(f"  {shape_type}: {count}")

        for i, obj in enumerate(objects, start=1):
            self.log(
                f"  #{i} type={obj.shape_type}, confidence={obj.confidence:.3f}, points={obj.num_points}"
            )

    def _to_json_safe(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.float32, np.float64, np.float16)):
            return float(value)
        if isinstance(value, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(value)
        if isinstance(value, dict):
            return {k: self._to_json_safe(v) for k, v in value.items() if k != "inliers"}
        if isinstance(value, (list, tuple)):
            return [self._to_json_safe(v) for v in value]
        return value

    def export_results_json(self):
        def _export():
            if not self.detected_objects:
                self.log("[WARN] No detected objects to export")
                return

            file_name, _ = QFileDialog.getSaveFileName(self, "Export detection results", "", "JSON Files (*.json)")
            if not file_name:
                return

            payload = {
                "num_objects": len(self.detected_objects),
                "objects": [self._to_json_safe(obj.to_dict()) for obj in self.detected_objects],
            }

            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self.log(f"Detection results exported: {file_name}")

        self._run_with_exception_logging("Export JSON", _export)

    def clear_scene(self):
        def _clear():
            self.plotter.clear()
            self.raw_points = None
            self.raw_normals = None
            self.detected_objects = []
            self.log("Scene cleared.")

        self._run_with_exception_logging("Clear scene", _clear)

    def generate_test_scene_preview(self):
        def _generate():
            if self.config is None:
                raise ValueError("Config is not loaded")

            generator = UniversalDataGenerator(self.config)
            num_objects = {shape_name: 1 for shape_name in self.config.shape_names}
            points, normals, labels = generator.generate_scene(num_objects)

            self.raw_points = points
            self.raw_normals = normals
            self.detected_objects = []

            self.render_background_point_cloud()
            self.log(f"Generated test scene with {len(points)} points")

            unique, counts = np.unique(labels, return_counts=True)
            label_stats = {int(k): int(v) for k, v in zip(unique, counts)}
            self.log(f"Label distribution: {label_stats}")

        self._run_with_exception_logging("Generate test scene", _generate)

    def start_training_script(self):
        def _start():
            if self.training_process is not None and self.training_process.state() != QProcess.NotRunning:
                self.log("[WARN] Training process is already running")
                return

            script_path = Path("scripts/train_universal.py")
            if not script_path.exists():
                raise FileNotFoundError(f"Training script not found: {script_path}")

            self.training_process = QProcess(self)
            self.training_process.setProgram(sys.executable)

            # Get training parameters from UI
            epochs = self.input_epochs.text()
            batch_size = self.input_batch_size.text()
            learning_rate = self.input_learning_rate.text()

            args = [str(script_path), "--config", self.config_path]
            if epochs:
                args.extend(["--epochs", epochs])
            if batch_size:
                args.extend(["--batch_size", batch_size])
            if learning_rate:
                args.extend(["--learning_rate", learning_rate])

            self.training_process.setArguments(args)

            self.training_process.readyReadStandardOutput.connect(self._on_training_stdout)
            self.training_process.readyReadStandardError.connect(self._on_training_stderr)
            self.training_process.finished.connect(self._on_training_finished)

            self.training_process.start()
            self.log(f"Training script started: {script_path}")

        self._run_with_exception_logging("Start training script", _start)

    def _on_training_stdout(self):
        data = bytes(self.training_process.readAllStandardOutput()).decode("utf-8", errors="ignore").strip()
        if data:
            for line in data.splitlines():
                self.log(f"[train] {line}")

    def _on_training_stderr(self):
        data = bytes(self.training_process.readAllStandardError()).decode("utf-8", errors="ignore").strip()
        if data:
            for line in data.splitlines():
                self.log(f"[train-err] {line}")

    def _on_training_finished(self, exit_code, exit_status):
        self.log(f"Training finished with exit_code={exit_code}, status={exit_status}")

    def run_intelligent_inference(self):
        """Legacy tunnel-only pipeline (compatibility mode)."""

        def _run_legacy():
            self.current_mode = "legacy_tunnel"

            if self.raw_points is None:
                self.log("[ERROR] No point cloud loaded")
                return

            model_path = "checkpoints/best_pipe_model.pth"
            if not os.path.exists(model_path):
                self.log(f"[ERROR] Legacy model not found: {model_path}")
                return

            # 从检查点中自动确定类别数
            checkpoint = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # 从conv2.weight的形状推断类别数
            num_classes = 3  # 默认值
            for key in state_dict:
                if key.endswith("conv2.weight"):
                    num_classes = state_dict[key].shape[0]
                    break

            self.log(f"Legacy model loaded with {num_classes} classes")

            model = get_model(num_classes=num_classes).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()

            self.log("Legacy preprocessing: downsampling and normal handling...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.raw_points)

            if self.raw_normals is not None and len(self.raw_normals) == len(self.raw_points):
                pcd.normals = o3d.utility.Vector3dVector(self.raw_normals)
            else:
                self.log("Legacy mode: estimating normals")

            downpcd = pcd.voxel_down_sample(voxel_size=0.05)
            if not downpcd.has_normals():
                downpcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
                )

            points = np.asarray(downpcd.points)
            normals = np.asarray(downpcd.normals)

            self.log("Legacy AI scanning...")
            block_size, stride = 3.0, 1.5
            all_labels = np.zeros(len(points))
            counts = np.zeros(len(points))
            xyz_min, xyz_max = points.min(0), points.max(0)

            z_range = xyz_max[2] - xyz_min[2]
            num_windows = int(np.ceil(z_range / stride)) if z_range > 0 else 1
            if num_windows > self.max_windows:
                stride = max(z_range / self.max_windows, block_size * 0.8)
                num_windows = int(np.ceil(z_range / stride)) if z_range > 0 else 1

            if num_windows <= 1:
                z_centers = np.array([(xyz_min[2] + xyz_max[2]) / 2.0])
            else:
                z_centers = np.linspace(xyz_min[2] + block_size / 2, xyz_max[2] - block_size / 2, num_windows)

            processed_windows = 0
            for i, z in enumerate(z_centers):
                if i % 10 == 0 or i == num_windows - 1:
                    self.log(f"Legacy progress: {i + 1}/{num_windows}")

                mask = (points[:, 2] >= z - block_size / 2) & (points[:, 2] < z + block_size / 2)
                idx = np.where(mask)[0]
                if len(idx) < 1024:
                    continue

                sel = np.random.choice(idx, 4096, replace=len(idx) < 4096)
                block_pts = points[sel] - points[sel].mean(0)
                block_feat = np.hstack((block_pts, normals[sel]))

                input_tensor = torch.FloatTensor(block_feat).unsqueeze(0).transpose(2, 1).to(self.device)
                with torch.no_grad():
                    pred = model(input_tensor)
                    pred_label = torch.argmax(pred, dim=2).cpu().numpy()[0]

                all_labels[sel] += (pred_label == 2).astype(np.int32)
                counts[sel] += 1
                processed_windows += 1

            self.log(f"Legacy sliding-window complete: {processed_windows}/{num_windows}")

            valid_mask = counts > 0
            if not np.any(valid_mask):
                self.log("[ERROR] No valid inference points")
                return

            pipe_mask = all_labels[valid_mask] / counts[valid_mask] > 0.1
            pipe_points = points[valid_mask][pipe_mask]
            if len(pipe_points) == 0:
                self.log("Legacy AI detected no pipes")
                return

            self.plotter.clear()
            self.plotter.add_mesh(pv.PolyData(self.raw_points), color=[0.25, 0.25, 0.25], point_size=1, opacity=0.1)

            found_pipes_count = 0
            remaining_points = pipe_points.copy()

            while len(remaining_points) > 200:
                center_xy = remaining_points[:, :2].mean(axis=0)
                radial_distances = np.linalg.norm(remaining_points[:, :2] - center_xy, axis=1)
                radius = np.median(radial_distances)
                center_z = remaining_points[:, 2].mean()
                center = np.array([center_xy[0], center_xy[1], center_z])
                axis = np.array([0.0, 0.0, 1.0])

                distances_to_surface = np.abs(radial_distances - radius)
                inliers = np.where(distances_to_surface < 0.08)[0]

                if 0.15 < radius < 0.8 and len(inliers) > 50:
                    found_pipes_count += 1
                    current_pipe_pts = remaining_points[inliers]
                    self.add_pipe_to_view(current_pipe_pts, center, axis, radius, found_pipes_count)
                    self.log(f"Legacy pipe #{found_pipes_count}: radius={radius:.3f}m")
                    remaining_points = np.delete(remaining_points, inliers, axis=0)
                else:
                    break

            if found_pipes_count == 0:
                self.log("Legacy geometry validation failed. No valid cylinder found.")
            else:
                self.log(f"Legacy completed. Found {found_pipes_count} pipes.")

        self._run_with_exception_logging("Run legacy tunnel inference", _run_legacy)

    def add_pipe_to_view(self, pts, center, axis, radius, pipe_id):
        projs = np.dot(pts - center, axis)
        h = projs.max() - projs.min()
        true_center = center + axis * (projs.max() + projs.min()) / 2.0

        color = list(np.random.choice(range(256), size=3) / 255.0)

        self.plotter.add_mesh(pv.PolyData(pts), color=color, point_size=4, name=f"pipe_pts_{pipe_id}")
        geom = pv.Cylinder(center=true_center, direction=axis, radius=radius, height=h)
        self.plotter.add_mesh(geom, color=color, opacity=0.4, name=f"pipe_geom_{pipe_id}")
        self.plotter.reset_camera()

    def run_batch_inference(self):
        """Run inference on multiple point cloud files in a directory."""
        def _run():
            if self.config is None:
                raise ValueError("Please load configuration first")
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Select input directory
            input_dir = QFileDialog.getExistingDirectory(self, "Select Input Directory")
            if not input_dir:
                return

            # Select output directory
            output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if not output_dir:
                return

            # Get list of point cloud files
            import glob
            extensions = ["*.npz", "*.npy", "*.txt", "*.ply", "*.pcd"]
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(input_dir, ext)))

            if len(files) == 0:
                self.log(f"No point cloud files found in {input_dir}")
                return

            self.log(f"Found {len(files)} files to process")
            self.log(f"Output directory: {output_dir}")

            # Initialize engine if not ready
            self._ensure_engine_ready()

            # Process each file
            success_count = 0
            for file_path in files:
                try:
                    self.log(f"Processing: {os.path.basename(file_path)}")

                    # Load point cloud
                    if file_path.endswith('.npz'):
                        data = np.load(file_path)
                        points = data['points']
                        normals = data['normals'] if 'normals' in data else None
                    elif file_path.endswith('.npy'):
                        data = np.load(file_path)
                        if data.shape[1] == 6:
                            points = data[:, :3]
                            normals = data[:, 3:]
                        else:
                            points = data
                            normals = None
                    elif file_path.endswith('.txt'):
                        data = np.loadtxt(file_path)
                        if data.shape[1] >= 6:
                            points = data[:, :3]
                            normals = data[:, 3:6]
                        else:
                            points = data[:, :3]
                            normals = None
                    else:  # .ply, .pcd
                        pcd = o3d.io.read_point_cloud(file_path)
                        points = np.asarray(pcd.points)
                        normals = np.asarray(pcd.normals) if pcd.has_normals() else None

                    if normals is None:
                        pcd_temp = o3d.geometry.PointCloud()
                        pcd_temp.points = o3d.utility.Vector3dVector(points)
                        pcd_temp.estimate_normals()
                        normals = np.asarray(pcd_temp.normals)

                    # Run inference
                    objects = self.engine.infer(points, normals)

                    # Save results
                    import json
                    output_file = os.path.join(output_dir, f"{Path(file_path).stem}_results.json")
                    results = {
                        'num_objects': len(objects),
                        'objects': [obj.to_dict() for obj in objects]
                    }
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                    self.log(f"  ✓ Saved: {Path(output_file).name}")
                    success_count += 1

                except Exception as e:
                    self.log(f"  ✗ Failed: {os.path.basename(file_path)} - {str(e)}")
                    continue

            self.log(f"Batch processing complete: {success_count}/{len(files)} files processed successfully")

        self._run_with_exception_logging("Batch inference", _run)

    def generate_training_data(self):
        """Generate training data using UniversalDataGenerator."""
        def _generate():
            if self.config is None:
                raise ValueError("Please load configuration first")

            # Get parameters from UI
            try:
                num_train = int(self.input_num_train.text())
                num_test = int(self.input_num_test.text())
            except ValueError:
                raise ValueError("Please enter valid numbers for train/test counts")

            # Select output directory
            output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Generated Data")
            if not output_dir:
                return

            self.log(f"Generating training data: {num_train} train scenes, {num_test} test scenes")
            self.log(f"Output directory: {output_dir}")

            # Create generator
            generator = UniversalDataGenerator(self.config)

            # Define object counts per scene (3-5 of each type for better balance)
            objects_per_scene = {}
            for shape_name in self.config.shape_names:
                objects_per_scene[shape_name] = (3, 5)  # 3-5 objects of each type

            # Generate training data
            train_dir = os.path.join(output_dir, 'train')
            os.makedirs(train_dir, exist_ok=True)
            self.log(f"Generating {num_train} training scenes...")
            generator.generate_dataset(num_train, objects_per_scene, train_dir)

            # Generate test data
            test_dir = os.path.join(output_dir, 'test')
            os.makedirs(test_dir, exist_ok=True)
            self.log(f"Generating {num_test} test scenes...")
            generator.generate_dataset(num_test, objects_per_scene, test_dir)

            self.log(f"Data generation complete!")
            self.log(f"Training data: {train_dir}")
            self.log(f"Test data: {test_dir}")

        self._run_with_exception_logging("Generate training data", _generate)

    def update_runtime_params(self):
        """Update runtime parameters from UI inputs."""
        try:
            self.voxel_size = float(self.input_voxel_size.text())
            self.max_windows = int(self.input_max_windows.text())
            self.render_point_limit = int(self.input_render_limit.text())

            self.log(f"Runtime parameters updated:")
            self.log(f"  Voxel size: {self.voxel_size}")
            self.log(f"  Max windows: {self.max_windows}")
            self.log(f"  Render limit: {self.render_point_limit}")
        except ValueError as e:
            self.log(f"[ERROR] Invalid parameter value: {e}")


TunnelDetectorApp = UniversalShapeDetectorApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UniversalShapeDetectorApp()
    window.show()
    sys.exit(app.exec_())
