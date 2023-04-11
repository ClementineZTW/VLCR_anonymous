import torch
from torch import nn
from module.DeiTs import DeiT_STR_Tiny,DeiT_STR_Base,DeiT_STR_Tiny_Short
from module.ABINet.modules.backbone import ResTranformer
from module.Decoders import Decoder
from module.SEAttentions import SqueezeExcitation


class CRMModel(nn.Module):
    def __init__(self,
                backbone_type,
                seq_max_length,
                num_classes,
                load_from_DeiT,
                definition_string,
                supervised_blocks,
                embed_dim = None,
                num_heads = None,
                with_cls_token = False,
                random_mask = False,
                share_parameters = False,
                SE = False,
                svtr_patch_embed = False,
                ):
        super().__init__()
        self.backbone_type = backbone_type
        self.seq_max_length = seq_max_length
        self.with_cls_token = with_cls_token
        self.random_mask = random_mask
        self.share_parameters = share_parameters
        self.SE = SE
        self.svtr_patch_embed = svtr_patch_embed


        assert backbone_type in ['Tiny','Base','ABINet','TinyShort']
        backbone_map = {
            'Tiny':DeiT_STR_Tiny,
            'Base':DeiT_STR_Base,
            'ABINet':ResTranformer,
            'TinyShort':DeiT_STR_Tiny_Short
        }
        self.backbone = backbone_map[backbone_type](load_from_DeiT,svtr_patch_embed)
        
        if embed_dim == None:
            embed_dim = self.backbone.DeiT_cfg['embed_dim']
        if num_heads == None:
            num_heads = self.backbone.DeiT_cfg['num_heads']

        self.inner_norm = nn.LayerNorm(embed_dim)

        if self.with_cls_token:
            assert backbone_type != 'ABINet', 'ABINet automatically support with cls_token, this config is for deits.'
            self.query_proj = nn.Linear(2*embed_dim,embed_dim)
        else:
            self.query_proj = nn.Linear(embed_dim,embed_dim)

        if backbone_type != 'ABINet':
            if SE:
                self.feat_proj = SqueezeExcitation(257)
            else:
                self.feat_proj = nn.Linear(embed_dim,embed_dim)
            self.decoder = Decoder(self.backbone.embed_dim,num_heads,num_classes,definition_string,supervised_blocks,random_mask,share_parameters)
            query = nn.Parameter(torch.randn((1,seq_max_length,self.backbone.embed_dim),dtype=torch.float))
            self.register_parameter('query',query)

        else:
            if SE:
                self.feat_proj = SqueezeExcitation(257)
            else:                
                self.feat_proj = nn.Linear(self.backbone.d_model,embed_dim)
            self.decoder = Decoder(self.backbone.d_model,num_heads,num_classes,definition_string,supervised_blocks,random_mask,share_parameters)
            query = nn.Parameter(torch.randn((1,seq_max_length,self.backbone.d_model),dtype=torch.float))
            self.register_parameter('query',query)




    def forward(self,x):
        x = self.backbone(x)
        (n,l,d) = x.shape

        x = self.inner_norm(x)

        if self.backbone_type != 'ABINet':
            if self.with_cls_token:
                cls_token = x[:,0,:].unsqueeze(1).repeat(1,self.seq_max_length,1)
                query = self.query.repeat(n,1,1)
                query = torch.cat((query,cls_token),dim=2)
                query = self.query_proj(query)

                feat = x[:,1:,:]
                feat = self.feat_proj(feat)
            else:
                query = self.query_proj(self.query)
                query = query.repeat(n,1,1)
                feat = x
                feat = self.feat_proj(feat)
        else:
            cls_token = x[:,0,:].unsqueeze(1)
            query = self.query + cls_token
            query = self.query_proj(query)

            feat = x[:,1:,:]
            feat = self.feat_proj(feat)



        res = self.decoder(query,feat)

        return res
    def forward_with_weight(self,x):
        x = self.backbone(x)
        (n,l,d) = x.shape

        x = self.inner_norm(x)

        if self.backbone_type != 'ABINet':
            if self.with_cls_token:
                cls_token = x[:,0,:].unsqueeze(1).repeat(1,self.seq_max_length,1)
                query = self.query.repeat(n,1,1)
                query = torch.cat((query,cls_token),dim=2)
                query = self.query_proj(query)

                feat = x[:,1:,:]
                feat = self.feat_proj(feat)
            else:
                query = self.query_proj(self.query)
                query = query.repeat(n,1,1)
                feat = x
                feat,weight = self.feat_proj.forward_with_weight(feat)
        else:
            cls_token = x[:,0,:].unsqueeze(1)
            query = self.query + cls_token
            query = self.query_proj(query)

            feat = x[:,1:,:]
            feat = self.feat_proj(feat)



        res = self.decoder(query,feat)

        return res,weight
'''
        self.feat_proj = nn.Linear(embed_dim if backbone_type!='ABINet' else self.backbone.d_model,embed_dim)
        
        if backbone_type != 'ABINet':
            self.decoder = Decoder(self.backbone.embed_dim,num_heads,num_classes,definition_string,supervised_blocks,random_mask,share_parameters)
        else:
            self.decoder = Decoder(self.backbone.DeiT_cfg['embed_dim'],num_heads,num_classes,definition_string,supervised_blocks,random_mask,share_parameters)
        if backbone_type != 'ABINet':
            query = nn.Parameter(torch.randn((1,seq_max_length,self.backbone.embed_dim),dtype=torch.float))
        else:
            query = nn.Parameter(torch.randn((1,seq_max_length,self.backbone.DeiT_cfg['embed_dim']),dtype=torch.float))
        self.register_parameter('query',query)
'''

if __name__ == '__main__':
    model = CRMModel('TinyShort',27,38,False,'V3V',[0,1,2],random_mask=False,share_parameters=False,with_cls_token=False,SE=True,svtr_patch_embed=True)
    print(model)
    x = torch.randn((10,3,32,128))
    res = model.forward_with_weight(x)
    print(res[0].shape,res[1].shape)
    #print(model.decoder)
