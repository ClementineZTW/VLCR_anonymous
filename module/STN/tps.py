import torch
from torch import nn
from module.STN.tps_spatial_transformer import TPS


class TPS(nn.Module):
    def __init__(self,tps_inputsize=[32,64],
                tps_outputsize=[32,100],
                tps_margin=[0.05,0.05],
                stn_activation='none',
                num_control_points=20)
        self.tps = 