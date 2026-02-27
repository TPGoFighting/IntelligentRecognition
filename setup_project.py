import os


def create_project_structure():
    print("🚀 正在构建 IntelligentRecognition 项目骨架...")

    # 1. 定义需要创建的文件夹结构
    folders = [
        "data/raw",
        "data/processed",
        "models",
        "core",
        "checkpoints"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 创建目录: {folder}/")

    # 2. 定义需要创建的空文件 (或初始化文件)
    files = {
        "models/__init__.py": "# 使 models 成为 Python 包\n",
        "core/__init__.py": "# 使 core 成为 Python 包\n",
        "requirements.txt": "numpy\ntorch\nopen3d\npyvista\npyvistaqt\nPyQt5\npyransac3d\n",
        ".gitignore": "data/\ncheckpoints/\n__pycache__/\n*.pyc\n.idea/\n.vscode/\n*.npy\n*.pcd\n*.ply\n",
        "README.md": "# Intelligent Recognition\n基于 PointNet++ 与 PCL 几何算法的隧道管道智能提取系统。\n",
        "main.py": "# 系统的可视化主入口\n",
        "train.py": "# 系统的训练脚本\n",
        "prepare_dataset.py": "# 数据预处理脚本\n"
    }

    for filepath, content in files.items():
        # 如果文件不存在，则创建并写入初始内容
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"📄 创建文件: {filepath}")
        else:
            print(f"⚠️ 文件已存在，跳过: {filepath}")

    print("\n✅ 项目骨架搭建完成！您可以开始将代码填入对应的文件中了。")


if __name__ == "__main__":
    create_project_structure()