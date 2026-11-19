import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 设备数量:", torch.cuda.device_count())
    print("当前使用的 CUDA 设备:", torch.cuda.current_device())
    print("CUDA 设备名称:", torch.cuda.get_device_name(torch.cuda.current_device()))
