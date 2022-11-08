import torch

# a = torch.ones(896, 512)
a = torch.Tensor([[1, 1], [1, 1]])
print(a)
print(a.shape)
T = 1
print(a.view(T, -1))