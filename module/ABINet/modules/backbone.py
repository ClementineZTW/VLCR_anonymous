import torch
from torch import nn
from .model import _default_tfmer_cfg
from .resnet import resnet45
from .transformer import (PositionalEncoding,
                                 TransformerEncoder,
                                 TransformerEncoderLayer)


class ResTranformer(nn.Module):
    def __init__(self, load_from_DeiT=False):
        super().__init__()
        self.resnet = resnet45()

        self.d_model = _default_tfmer_cfg['d_model']

        nhead = _default_tfmer_cfg['nhead']

        self.DeiT_cfg = {
            'embed_dim':self.d_model,
            'num_heads':nhead,
        }
        d_inner = _default_tfmer_cfg['d_inner']
        dropout = _default_tfmer_cfg['dropout']
        activation = _default_tfmer_cfg['activation']
        num_layers = 2

        self.pos_encoder = PositionalEncoding(self.d_model, max_len=8*32)
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=nhead, 
                dim_feedforward=d_inner, dropout=dropout, activation=activation)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

    def forward(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).permute(2, 0, 1)
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature)
        #feature = feature.permute(1, 2, 0).view(n, c, h, w)
        feature = feature.transpose(0,1)
        return feature


if __name__ == '__main__':
    x = torch.randn((10,3,32,128))
    model = ResTranformer(None)
    print(model(x).shape)