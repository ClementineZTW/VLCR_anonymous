import torch
from torch import nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed
from module.PatchEmbeds import SVTRPatchEmbed


class DeiT_STR(VisionTransformer):
    DeiT_cfg = None
    Deit_checkpoint_url = None

    def __init__(self,load_from_DeiT=True,svtr_patch_embed=False):
        super().__init__(**self.DeiT_cfg)
        self.load_from_DeiT = load_from_DeiT
        self.svtr_patch_embed = svtr_patch_embed
        if self.svtr_patch_embed:
            assert load_from_DeiT == False
        self.load() if load_from_DeiT else None
        self.reset_patch_embed()
        

    def load(self):
        checkpoint = torch.hub.load_state_dict_from_url(
                url=self.Deit_checkpoint_url,
                map_location="cpu", check_hash=True
            )
        self.load_state_dict(checkpoint["model"])
        print('Load DeiT OK.')


    def reset_patch_embed(self,patch_size=4,img_size=(32,128)):
        num_patches = img_size[0]*img_size[1]//patch_size//patch_size
        if self.svtr_patch_embed:
            self.patch_embed = SVTRPatchEmbed(3,self.DeiT_cfg['embed_dim'])
        else:            
            self.patch_embed = PatchEmbed(img_size,patch_size,embed_dim=self.embed_dim)
            
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,self.embed_dim))    
    
    def forward(self,x):        
        return super().forward_features(x)


class DeiT_STR_Tiny(DeiT_STR):
    DeiT_cfg = {
        'patch_size':16,
        'embed_dim':192,
        'depth':12,
        'num_heads':3,
        'mlp_ratio':4,
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
    }
    Deit_checkpoint_url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'


class DeiT_STR_Base(DeiT_STR):
    DeiT_cfg = {
        'patch_size':16,
        'embed_dim':768,
        'depth':12,
        'num_heads':12,
        'mlp_ratio':4,
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
    }
    Deit_checkpoint_url = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'


class DeiT_STR_Tiny_Short(DeiT_STR):
    DeiT_cfg = {
        'patch_size':16,
        'embed_dim':192,
        'depth':9,
        'num_heads':3,
        'mlp_ratio':4,
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
    }
    Deit_checkpoint_url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'


if __name__ == '__main__':
    device = 'cpu'
    model = DeiT_STR_Tiny(False,True)
    x = torch.randn((10,3,32,128))
    res = model(x)
    print(res.shape)

