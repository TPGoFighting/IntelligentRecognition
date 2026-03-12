import torch
import numpy as np

def check_model_weights(path):
    print(f"检查模型权重: {path}")
    state_dict = torch.load(path, map_location='cpu', weights_only=True)

    total_params = 0
    nan_params = 0
    inf_params = 0

    for name, param in state_dict.items():
        num = param.numel()
        total_params += num
        nan_count = torch.isnan(param).sum().item()
        inf_count = torch.isinf(param).sum().item()
        nan_params += nan_count
        inf_params += inf_count

        if nan_count > 0 or inf_count > 0:
            print(f"  层 {name}: {param.shape}, NaN: {nan_count}, Inf: {inf_count}")

    print(f"总参数: {total_params}")
    print(f"NaN参数: {nan_params}")
    print(f"Inf参数: {inf_params}")

    if nan_params == 0 and inf_params == 0:
        print("权重正常，无NaN/Inf")
    else:
        print("警告: 权重包含NaN/Inf值")

if __name__ == "__main__":
    check_model_weights("checkpoints/best_pipe_model.pth")