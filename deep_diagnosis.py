"""
深度诊断：检查为什么准确率卡在33%
"""
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def check_data_quality():
    """检查训练数据质量"""
    print("="*60)
    print("1. 检查训练数据质量")
    print("="*60)

    # 加载几个训练样本
    train_dir = Path('data/cylinder_aggressive/train')
    files = list(train_dir.glob('scene_*.npz'))[:5]

    for i, f in enumerate(files):
        data = np.load(f)
        points = data['points']
        labels = data['labels']

        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n场景 {i+1}:")
        print(f"  总点数: {len(points)}")
        for label, count in zip(unique, counts):
            label_name = {0: '背景', 2: '圆柱体'}.get(label, f'未知({label})')
            print(f"  {label_name}: {count} ({count/len(points)*100:.1f}%)")

        # 检查标签是否正确
        if len(unique) == 1:
            print(f"  ⚠️ 警告：只有一个类别！")

        # 检查点云范围
        print(f"  点云范围: X[{points[:,0].min():.1f}, {points[:,0].max():.1f}] "
              f"Y[{points[:,1].min():.1f}, {points[:,1].max():.1f}] "
              f"Z[{points[:,2].min():.1f}, {points[:,2].max():.1f}]")

def check_model_output():
    """检查模型输出"""
    print("\n" + "="*60)
    print("2. 检查模型输出")
    print("="*60)

    from models.pointnet2_sem_seg import get_model

    model_path = Path('models/cylinder_aggressive/best_model.pth')
    if not model_path.exists():
        print("模型不存在，跳过")
        return

    # 加载模型
    model = get_model(num_classes=3)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载一个测试样本
    data = np.load('data/cylinder_aggressive/test/scene_0000.npz')
    points = data['points']
    normals = data['normals']
    labels = data['labels']

    # 采样4096个点
    indices = np.random.choice(len(points), 4096, replace=len(points)<4096)
    sample_points = points[indices]
    sample_normals = normals[indices]
    sample_labels = labels[indices]

    # 归一化
    center = sample_points.mean(axis=0)
    sample_points = sample_points - center

    # 组合特征
    features = np.concatenate([sample_points, sample_normals], axis=1)
    features_tensor = torch.FloatTensor(features).unsqueeze(0).transpose(2, 1)  # (1, 6, 4096)

    # 前向传播
    with torch.no_grad():
        outputs = model(features_tensor)  # (1, 3, 4096)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

    # 分析输出
    print(f"\n输入:")
    print(f"  特征形状: {features_tensor.shape}")
    print(f"  真实标签分布:")
    unique, counts = np.unique(sample_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    Label {label}: {count} ({count/len(sample_labels)*100:.1f}%)")

    print(f"\n输出:")
    print(f"  输出形状: {outputs.shape}")
    print(f"  预测标签分布:")
    preds_np = preds.cpu().numpy().flatten()
    unique, counts = np.unique(preds_np, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    Label {label}: {count} ({count/len(preds_np)*100:.1f}%)")

    print(f"\n概率统计:")
    probs_np = probs.cpu().numpy()[0]  # (3, 4096)
    for i in range(3):
        print(f"  Class {i}: mean={probs_np[i].mean():.4f}, max={probs_np[i].max():.4f}, min={probs_np[i].min():.4f}")

    # 计算准确率
    correct = (preds_np == sample_labels).sum()
    acc = correct / len(sample_labels)
    print(f"\n准确率: {acc:.4f} ({acc*100:.2f}%)")

    # 检查是否总是预测同一个类别
    if len(unique) == 1:
        print(f"\n⚠️ 严重问题：模型总是预测类别 {unique[0]}！")
        print("   这说明模型没有学到任何有用的特征。")
        return False

    return True

def check_loss_function():
    """检查损失函数"""
    print("\n" + "="*60)
    print("3. 检查损失函数和类别权重")
    print("="*60)

    from core.config_loader import load_config

    config = load_config('config/cylinder_aggressive_config.yaml')
    weights = config.get_class_weights()

    print(f"类别权重: {weights}")
    print(f"  背景权重: {weights[0]}")
    print(f"  未标记权重: {weights[1]}")
    print(f"  圆柱体权重: {weights[2]}")

    # 模拟损失计算
    print(f"\n如果模型总是预测背景（label 0）:")
    print(f"  对于背景点：损失 = {weights[0]:.2f} × log(prob)")
    print(f"  对于圆柱体点：损失 = {weights[2]:.2f} × log(prob)")

    print(f"\n如果模型总是预测圆柱体（label 2）:")
    print(f"  对于背景点：损失 = {weights[0]:.2f} × log(prob)")
    print(f"  对于圆柱体点：损失 = {weights[2]:.2f} × log(prob)")

def main():
    print("\n深度诊断：为什么准确率卡在33%？\n")

    check_data_quality()
    model_ok = check_model_output()
    check_loss_function()

    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)

    if not model_ok:
        print("\n发现严重问题：")
        print("1. 模型总是预测同一个类别")
        print("2. 这说明模型没有学到有用的特征")
        print("\n可能的原因：")
        print("- 数据生成有问题（圆柱体特征不明显）")
        print("- 模型架构不适合这个任务")
        print("- 学习率太高或太低")
        print("- 需要更简单的场景来训练")
        print("\n建议：")
        print("1. 检查数据生成代码")
        print("2. 尝试极简场景（1个圆柱体，少量背景）")
        print("3. 降低学习率到0.0001")
        print("4. 使用更简单的模型")

if __name__ == "__main__":
    main()
