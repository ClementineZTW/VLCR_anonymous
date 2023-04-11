import torch
from torch import nn

class MultiCELoss(nn.Module):
    def __init__(self,num,length):
        super().__init__()
        self.num = num
        self.length = length
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
    def forward(self,x,target):
        #x:[N,T*num,C], target:[N,T]
        T = self.length
        tgt = target.contiguous().view(-1)
        total_loss = 0
        for i in range(self.num):
            xi = x[:,i*T:(i+1)*T,:]
            total_loss += self.loss(xi.reshape(-1,xi.shape[-1]),tgt)
        return total_loss

if __name__ == '__main__':
    x = torch.randn((10,27*3,38),dtype=torch.float32)
    target = [1]+[0]*26
    target = torch.LongTensor(target).repeat(10,1)
    loss = MultiCELoss(3,27)
    print(loss(x,target))