import torch
import subprocess

def check_torch_cuda():
    if torch.cuda.is_available():
        print(f"Torch CUDA 可用，GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Torch CUDA 不可用")

def check_nvidia_smi():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("nvidia-smi 可用:")
            print(result.stdout)
        else:
            print("nvidia-smi 不可用:")
            print(result.stderr)
    except FileNotFoundError:
        print("未找到 nvidia-smi")

if __name__ == "__main__":
    check_torch_cuda()
    print("-")
    check_nvidia_smi()
