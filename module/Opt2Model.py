import torch
from torch import nn
from torch.cuda.amp import autocast
from module import Models
from module.Predators import PredatorTiny
from module.Finders import FinderTiny

class WarpModel(nn.Module):
    def __init__(self,model,amp):
        super().__init__()
        self.model = model
        self.amp = amp
        if self.amp:
            print('USE AMP')
    def forward(self,x,is_eval=False,*args,**kwargs):
        if self.amp:
            with autocast():
                res = self.model(x,*args,**kwargs)
        else:
            res = self.model(x,*args,**kwargs)
        if is_eval:
            return [0,res]
        else:
            return [res]


class WarpPredator(nn.Module):
    def __init__(self,model,amp):
        super().__init__()
        self.model = model
        self.amp = amp
        if self.amp:
            print('USE AMP')
    def forward(self,x,is_eval=False,*args,**kwargs):
        if self.amp:
            with autocast():
                res = self.model(x,is_eval,*args,**kwargs)
        else:
            res = self.model(x,is_eval,*args,**kwargs)
        if is_eval:
            return [0,res]
        else:
            return [res]

def CreateModel(opt):
    if opt.CRM:
        s = opt.supervised_blocks
        s = [int (char) for char in s.split(',')]
        model =  Models.CRMModel(opt.backbone_type,
                                opt.batch_max_length+2,
                                opt.num_class,
                                opt.load_from_DeiT,
                                opt.definition_string,
                                s,
                                opt.decoder_embed_dim,
                                opt.decoder_num_heads,
                                opt.with_cls_token,
                                opt.random_mask,
                                opt.share_parameters,
                                opt.SE,
                                opt.SVTRPatchEmbed,
                                )
        return WarpModel(model,opt.amp)

    elif opt.Predator:
        if opt.backbone_type == 'Tiny':
            model = PredatorTiny()
        else:
            print('Predator donot support backbone'+opt.backbone_type)
        return WarpPredator(model,out.amp)
    elif opt.Finder:
        if opt.backbone_type == 'Tiny':
            model = FinderTiny()
        else:
            print('Finder donot support backbone'+opt.backbone_type)
        return WarpModel(model,opt.amp)
    else:
        print('No Model Selected!!!')



