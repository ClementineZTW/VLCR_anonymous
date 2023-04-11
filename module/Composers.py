import torch
from torch import nn


class Composer(nn.Module):
    def __init__(self,input_dim,couple_dim,num_class):
        super().__init__()
        self.input_dim = input_dim
        self.couple_dim = couple_dim
        self.num_couples = num_class ** 2
        #self.single_char = nn.Parameter(torch.randn((1,self.num_class,input_dim)))
        #self.couple_char = nn.Parameter(torch.randn((1,self.couple_types,couple_dim)))
        self.single_linear = nn.Linear(input_dim,num_class)
        
        self.couple_conv = nn.Conv2d(input_dim,couple_dim,(2,1),1,0)
        self.couple_linear = nn.Linear(couple_dim,self.num_couples)

        self.upscale_proj = nn.Linear(input_dim,couple_dim)
        self.register_buffer('zero',torch.zeros(1,1,input_dim))

    def forward(self,x):
        (n,l,d) = x.shape
        zero = self.zero.repeat(n,1,1)
        zero_x =  torch.cat((zero,x),dim=1).unsqueeze(2)
        x_zero = torch.cat((x,zero),dim=1).unsqueeze(2)
        map = torch.cat((zero_x,x_zero),dim=2).permute(0,3,2,1)
        map = self.couple_conv(map).squeeze(2).transpose(1,2)
        couple = self.couple_linear(map)

        single = torch.softmax(self.single_linear(x),dim=-1)
        print(single.shape)

        #p[i] = max(p[i-1]*single[i],p[i-2]*couple[i])
        #couple[i] = x_[i] ++ x_[i-1]


        #[1] Accept
        

        #[2] Reject

        #let's use DP!!



if __name__ == '__main__':
    x = torch.randn((10,27,192))
    model = Composer(192,384,38)
    x = model(x)
    single,couple = x
    print(single.shape,couple.shape)
