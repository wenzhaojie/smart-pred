import torch


def test():
    x = torch.rand(5, 3)
    print(x)
    print(torch.cuda.is_available())


if __name__ == "__main__":
    test()