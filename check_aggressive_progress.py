"""
监控激进优化训练进度
"""
import os
from pathlib import Path
import json
import torch

def check_progress():
    print("\n" + "="*60)
    print("圆柱体检测 - 激进优化训练进度")
    print("="*60)

    # 检查数据生成
    train_dir = Path("data/cylinder_aggressive/train")
    test_dir = Path("data/cylinder_aggressive/test")

    print("\n[数据生成]")
    if train_dir.exists():
        train_files = list(train_dir.glob("scene_*.npz"))
        print(f"  训练场景: {len(train_files)}/800")
        if len(train_files) > 0:
            # 检查第一个场景的数据平衡
            import numpy as np
            data = np.load(train_files[0])
            labels = data['labels']
            unique, counts = np.unique(labels, return_counts=True)
            bg_ratio = counts[0] / counts.sum() * 100
            cyl_ratio = counts[1] / counts.sum() * 100 if len(counts) > 1 else 0
            print(f"  数据平衡: 背景{bg_ratio:.1f}% vs 圆柱体{cyl_ratio:.1f}%")
    else:
        print("  训练场景: 未开始")

    if test_dir.exists():
        test_files = list(test_dir.glob("scene_*.npz"))
        print(f"  测试场景: {len(test_files)}/150")
    else:
        print("  测试场景: 未开始")

    # 检查模型训练
    model_dir = Path("models/cylinder_aggressive")

    print("\n[模型训练]")
    if not model_dir.exists():
        print("  状态: 未开始")
        return

    best_model = model_dir / "best_model.pth"
    latest_model = model_dir / "latest_model.pth"
    history_file = model_dir / "training_history.json"

    if best_model.exists():
        try:
            ckpt = torch.load(best_model, map_location='cpu', weights_only=False)
            epoch = ckpt.get('epoch', 'N/A')
            best_acc = ckpt.get('best_acc', 0)
            val_acc = ckpt.get('val_acc', 0)
            train_acc = ckpt.get('train_acc', 0)

            print(f"  当前Epoch: {epoch}/150")
            print(f"  最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
            print(f"  验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"  训练准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")

            # 对比旧模型
            old_acc = 0.3354
            improvement = (best_acc - old_acc) / old_acc * 100
            print(f"\n  对比旧模型:")
            print(f"    旧模型准确率: {old_acc:.4f} ({old_acc*100:.2f}%)")
            print(f"    提升幅度: {improvement:+.1f}%")

            # 评估
            if best_acc >= 0.70:
                print(f"\n  ✓ 已达到目标准确率 (70%+)")
            elif best_acc >= 0.60:
                print(f"\n  → 接近目标，继续训练...")
            elif best_acc >= 0.50:
                print(f"\n  → 有进步，但还需提升...")
            else:
                print(f"\n  → 准确率仍然较低，可能需要进一步优化")

        except Exception as e:
            print(f"  无法读取模型: {e}")
    else:
        print("  状态: 训练中...")

    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            epochs_done = len(history.get('train_loss', []))
            print(f"\n  已完成轮数: {epochs_done}/150")

            if epochs_done > 0:
                print(f"  最新训练损失: {history['train_loss'][-1]:.4f}")
                print(f"  最新验证损失: {history['val_loss'][-1]:.4f}")
        except:
            pass

    print("\n" + "="*60)

if __name__ == "__main__":
    check_progress()
