"""
检查训练进度和状态
"""
import os
from pathlib import Path
import json

def check_data_generation():
    """检查数据生成进度"""
    train_dir = Path("data/cylinder_enhanced/train")
    test_dir = Path("data/cylinder_enhanced/test")

    print("="*60)
    print("数据生成状态")
    print("="*60)

    if train_dir.exists():
        train_files = list(train_dir.glob("scene_*.npz"))
        print(f"训练数据: {len(train_files)}/500 场景")
    else:
        print("训练数据: 未开始")

    if test_dir.exists():
        test_files = list(test_dir.glob("scene_*.npz"))
        print(f"测试数据: {len(test_files)}/100 场景")
    else:
        print("测试数据: 未开始")

def check_training():
    """检查训练进度"""
    model_dir = Path("models/cylinder_model")

    print("\n" + "="*60)
    print("训练状态")
    print("="*60)

    if not model_dir.exists():
        print("训练: 未开始")
        return

    # 检查模型文件
    best_model = model_dir / "best_model.pth"
    latest_model = model_dir / "latest_model.pth"
    history_file = model_dir / "training_history.json"

    if best_model.exists():
        print(f"[OK] Best model: {best_model}")

        # 读取模型信息
        import torch
        try:
            ckpt = torch.load(best_model, map_location='cpu', weights_only=False)
            print(f"  - Epoch: {ckpt.get('epoch', 'N/A')}")
            print(f"  - Best accuracy: {ckpt.get('best_acc', 'N/A'):.4f}")
            print(f"  - Val accuracy: {ckpt.get('val_acc', 'N/A'):.4f}")
            print(f"  - Train accuracy: {ckpt.get('train_acc', 'N/A'):.4f}")
        except Exception as e:
            print(f"  - Cannot read model info: {e}")
    else:
        print("Best model: Not generated")

    if latest_model.exists():
        print(f"[OK] Latest model: {latest_model}")
    else:
        print("Latest model: Not generated")

    if history_file.exists():
        print(f"[OK] Training history: {history_file}")

        # 读取训练历史
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)

            epochs = len(history.get('train_loss', []))
            print(f"  - Completed epochs: {epochs}")

            if epochs > 0:
                print(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
                print(f"  - Final train acc: {history['train_acc'][-1]:.4f}")
                print(f"  - Final val loss: {history['val_loss'][-1]:.4f}")
                print(f"  - Final val acc: {history['val_acc'][-1]:.4f}")
        except Exception as e:
            print(f"  - Cannot read history: {e}")
    else:
        print("Training history: Not generated")

def main():
    print("\n圆柱体识别训练进度检查\n")

    check_data_generation()
    check_training()

    print("\n" + "="*60)
    print("提示")
    print("="*60)
    print("- 数据生成通常需要5-10分钟")
    print("- 训练100个epochs通常需要30-60分钟（使用GPU）")
    print("- 可以随时运行此脚本检查进度")
    print("\n使用 Ctrl+C 可以中断训练（会保存最新检查点）")

if __name__ == "__main__":
    main()
