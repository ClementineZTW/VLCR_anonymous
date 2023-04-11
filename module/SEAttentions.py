import torch
from torch import nn

class SqueezeExcitation(nn.Module):
    def __init__(self,num_channel,mlp_ratio=4):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channel,num_channel*mlp_ratio),
            nn.ReLU(),
            nn.Linear(num_channel*mlp_ratio,num_channel),
            nn.Sigmoid()
        )
    def forward(self,x):
        avg = self.avgpool(x).squeeze(-1)
        weight = self.fc(avg).unsqueeze(-1)
        x = x*weight
        #print('This is SE Module.')
        return x
    def forward_with_weight(self,x):
        avg = self.avgpool(x).squeeze(-1)
        weight = self.fc(avg).unsqueeze(-1)
        x = x*weight
        #print('This is SE Module.')
        return x,weight

if __name__ == '__main__':
    x = torch.randn((10,256,192))
    model = SqueezeExcitation(256)
    print(model(x).shape)