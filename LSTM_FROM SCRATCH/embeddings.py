import torch as T
import torch.nn as nn

embed = nn.Embedding(5, 4)

print(embed.weight)
