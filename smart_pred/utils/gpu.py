import torch


def test_gpu_available():
    x = torch.rand(5, 3)
    print(f"Tensor torch.rand(5, 3):\n {x}")
    print(f"Is GPU available: {torch.cuda.is_available()}")
    # 获取GPU的型号与数量
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU device count: {torch.cuda.device_count()}")


if __name__ == "__main__":
    test_gpu_available()