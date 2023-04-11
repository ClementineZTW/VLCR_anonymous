import torch
from torch import nn


class SVTRPatchEmbed(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.blocks = nn.Sequential(
            nn.Conv2d(input_dim,output_dim//2,3,2,1),
            nn.BatchNorm2d(output_dim//2),
            nn.GELU(),
            nn.Conv2d(output_dim//2,output_dim,3,2,1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )
    def forward(self,x):
        x = self.blocks(x)
        x = x.flatten(2).transpose(1,2)
        return x


if __name__ == '__main__':
    x = torch.randn((10,3,32,128))
    model = SVTRPatchEmbed(3,192)
    res = model(x)
    print(res.shape)